# -*- coding: utf-8 -*-
"""Code to initialize the architecture of a Domain-Adaptive Recurrent Neural Network (DA-RNN) for driving manoeuver
anticipation. 

From the paper "Robust and Subject-Independent Driving Manoeuvre Anticipation through Domain-Adversarial 
Recurrent Neural Networks", by Tonutti M, Ruffaldi E, et al.

Author: Michele Tonutti
Date: 2019-02-09
"""
from math import exp

from keras.callbacks import EarlyStopping
from keras.layers import (
    Input,
    LSTM,
    Dense,
    concatenate,
    Dropout,
    GRU,
    Masking
)
from keras.layers.wrappers import TimeDistributed
from keras.models import (
    Model
)
from keras.engine import Layer
import keras.backend as K
from tensorflow.python.framework import ops
import tensorflow as tf

BIAS_INITIALIZER = 'ones'
KERNEL_INITIALIZER = 'VarianceScaling'
OPTIMIZER = 'adam'

UNITS_DENSE = 128
UNITS_GRU = 128
UNITS_LSTM = 64

RECURRENT_DROPOUT_STRENGTH = 0.5
DROPOUT_STRENGTH = 0.6
LAMBDA_REVERSAL_STRENGTH = 0.31


class DomainAdaptiveRNN:
    """Class to create the architecture of a Domain Adaptive RNN for maneuver anticipation.
    
    Parameters
    ----------
    n_timesteps : int
        Number of timesteps in each sample.
    n_feats_head : int
        Number of features related to head movements.
    n_feats_outside : int
        Number of features from the camera outside the car (street).
    n_feats_gaze : int
        Number of features related to gaze movements.
    dense_units : int, optional
        Number of units in the dense layers.
    gru_units : int, optional
        Number of units in the GRU layers.
    lstm_units : int, optional
        Number of units in the LSTM layers.
    lambda_reversal_strength : float, optional
        Constant controlling the ratio of the domain classifier loss to action classifier loss 
        (lambda = L_class / L_domain)
        A higher lambda will increase the influence of the domain classifier, rewarding domain-invariant features. 
        A lower lambda will increase the influence of the manoeuver anticipation, rewarding correct classification.
    kernel_initializer : str, optional
        Initializer for the kernel of recurrent layers.
    bias_initializer: str, optional
        Initializer for the bias of recurrent layers.
    dropout : float, optional
        Strength of regular dropout in Dense layers.
    rec_dropout: float, optional
        Strength of recurrent dropout in recurrent layers.
    """

    def __init__(self, n_timesteps, n_feats_head, n_feats_outside, n_feats_gaze,
                 dense_units=UNITS_DENSE, gru_units=UNITS_GRU, lstm_units=UNITS_LSTM,
                 lambda_reversal_strength=LAMBDA_REVERSAL_STRENGTH, kernel_initializer=KERNEL_INITIALIZER,
                 bias_initializer=BIAS_INITIALIZER, dropout=DROPOUT_STRENGTH, rec_dropout=RECURRENT_DROPOUT_STRENGTH):
        self.n_timesteps = n_timesteps
        self.n_feats_head = n_feats_head
        self.n_feats_outside = n_feats_outside
        self.n_feats_gaze = n_feats_gaze

        self.dense_units = dense_units
        self.gru_units = gru_units
        self.lstm_units = lstm_units
        self.lambda_reversal = lambda_reversal_strength
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.rec_dropout = rec_dropout

        self.model = None

    def create_loss_weights(self):
        """Create loss weights that increase exponentially with time.
        
        Returns
        -------
        type : list
            A list containing a weight for each timestep.
        """
        weights = []
        for t in range(self.n_timesteps):
            weights.append(exp(-(self.n_timesteps - t)))
        return weights

    def create_architecture(self):
        """Create the model architecture.
        
        Returns
        -------
        type : keras.Model
            The initialized model object.
        """
        # Inputs
        input_shape_head = (self.n_timesteps, self.n_feats_head)
        input_inside = Input(shape=input_shape_head, name='input_head')

        input_shape_outside = (self.n_timesteps, self.n_feats_outside)
        input_outside = Input(shape=input_shape_outside, name='input_outside')

        input_shape_gaze = (self.n_timesteps, self.n_feats_gaze)
        input_gaze = Input(shape=input_shape_gaze, name='input_gaze')

        # Recurrent section for head movement features
        lstm_out_inside = TimeDistributed(Masking(mask_value=0), name='dense_head')(input_inside)
        lstm_out_inside = LSTM(
            self.lstm_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='lstm_head'
        )(lstm_out_inside)
        lstm_out_inside = Dropout(self.dropout)(lstm_out_inside)

        # Recurrent section for outside features (context)
        lstm_out_outside = TimeDistributed(Masking(mask_value=0), name='dense_outside')(input_outside)
        lstm_out_outside = LSTM(
            self.lstm_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='lstm_outside'
        )(lstm_out_outside)
        lstm_out_outside = Dropout(self.dropout)(lstm_out_outside)

        # Recurrent section for gaze (eye-tracking) features
        lstm_out_gaze = TimeDistributed(Masking(mask_value=0), name='dense_gaze')(input_gaze)
        lstm_out_gaze = LSTM(
            self.lstm_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='lstm_gaze'
        )(lstm_out_gaze)
        lstm_out_gaze = Dropout(self.dropout)(lstm_out_gaze)

        # Action fusion - merge gaze and head features
        lstm_first_concat = concatenate([lstm_out_inside, lstm_out_gaze])
        lstm_first_concat = GRU(
            self.gru_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
            recurrent_dropout=self.rec_dropout,
            bias_initializer=self.bias_initializer,
            name='action_fusion'
        )(lstm_first_concat)
        lstm_first_concat = Dropout(self.dropout)(lstm_first_concat)

        # Action-context fusion - merge action features (gaze + head) with context features (from outside)
        lstm_second_concat = concatenate([lstm_first_concat, lstm_out_outside])
        aux_output = TimeDistributed(
            Dense(self.dense_units, activation='tanh',
                  kernel_initializer=self.kernel_initializer),
            name='action_context_fusion'
        )(lstm_second_concat)
        aux_output = Dropout(self.dropout)(aux_output)

        # Action classification - classify the manouever
        main_output = TimeDistributed(
            Dense(self.dense_units, activation='softmax', kernel_initializer=self.kernel_initializer),
            name='aux_classifier'
        )(aux_output)

        # Domain Adaptation section - classify the domain
        flip_layer = GradientReversal(self.lambda_reversal)
        dann_in = flip_layer(aux_output)
        dann_out = Dense(
            units=2, activation='softmax', kernel_initializer=self.kernel_initializer, name='domain_classifier'
        )(dann_in)

        self.model = Model(inputs=[input_inside, input_outside, input_gaze], outputs=[main_output, dann_out])

        return self.model

    def compile_model(self, loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=None, loss_weights=None):
        """Compile the model.
        
        Parameters
        ----------
        loss : str or custom loss function, optional
            Loss function to use for the training. Categorical crossentropy by default.
        optimizer : str or custom optimizer object, optional
            Optimizer to use for the training. Adam by default.
        metrics : list
            Metric to use for the training. Can be a custom metric function.
        loss_weights: dict
            Dictionary of loss weights. The items of the dictionary can be lists, with one weight per timestep.

        Returns
        -------
        type : keras.Model
            The compiled model.
        """
        if metrics is None:
            metrics = ['accuracy']
        if loss_weights is None:
            weights = self.create_loss_weights()
            loss_weights = {'domain_classifier': weights, 'aux_classifier': weights}

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, sample_weight_mode='temporal',
                           loss_weights=loss_weights)

        return self.model

    def fit_model(self, X_train_head, X_train_outside, X_train_gaze,
                  y_train, y_train_domain, batch_size=128, epochs=1000, patience=30):
        """Fit the model on a training set.
        
        The training sets must be divided in batches that contain both the source and target domain, 
        in order to perform adaptation.
                    
        Parameters
        ----------
        X_train_head : np.ndarray
            Head features. Shape = (n_samples, n_timestamps, n_features)
        X_train_outside : np.ndarray
            Context features. Shape = (n_samples, n_timestamps, n_features)
        X_train_gaze : np.ndarray
            Gaze features. Shape = (n_samples, n_timestamps, n_features)
        y_train : np.ndarray
            Action labels, encoded as integers.
        y_train_domain : np.ndarray
            Binary domain labels, encoded as integers.
        batch_size : int, optional
            Size of the batches for training. Default = 128.
        epochs : int, optional
            Number of epochs to run the training. Default = 1000.
        patience: int, optional
            Number of epochs to wait without an improvement in the validation loss for early stopping. Default = 30.
            
        Returns
        -------
        type : keras.History
            The history of the trained model as a keras.History object.
        """

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        model_history = self.model.fit(
            x=[X_train_head, X_train_outside,
               X_train_gaze],
            y=[y_train, y_train_domain],
            epochs=epochs, batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping])

        return model_history

    def predict_proba(self, X_test_head, X_test_outside, X_test_gaze):
        """Predict probabilities on a test set.
        
        The test set can come from either the source or target domain.
        
        Parameters
        ----------
        X_test_head : np.ndarray
            Head features. Shape = (n_samples, n_timestamps, n_features)
        X_test_outside : np.ndarray
            Context features. Shape = (n_samples, n_timestamps, n_features)
        X_test_gaze : np.ndarray
            Gaze features. Shape = (n_samples, n_timestamps, n_features)

        Returns
        -------
        type : np.ndarray
            Predicted probabilities.
        """
        proba = self.model.predict_proba([X_test_head, X_test_outside, X_test_gaze], batch_size=2)[0]

        return proba


def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    # Create architecture with a custom number of features and timesteps.
    da_rnn = DomainAdaptiveRNN(n_timesteps=150, n_feats_head=20, n_feats_outside=10, n_feats_gaze=6)
    da_rnn.create_architecture()
    model = da_rnn.compile_model()
    print(model.summary())
