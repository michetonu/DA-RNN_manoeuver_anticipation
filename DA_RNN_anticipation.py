# -*- coding: utf-8 -*-
from math import exp

from keras.callbacks import EarlyStopping
from keras.models import (
    Model
)
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
from flip_gradient_tf import GradientReversal

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
    """Domain adaptive RNN"""
    def __init__(self, n_timesteps, n_feats_head, n_feats_outside, n_feats_gaze,
                 dense_units=UNITS_DENSE, gru_units=UNITS_GRU, lstm_units=UNITS_LSTM,
                 lambda_reversal_strength=LAMBDA_REVERSAL_STRENGTH, kernel_initializer=KERNEL_INITIALIZER,
                 bias_initializer=BIAS_INITIALIZER, dropout=DROPOUT_STRENGTH, rec_dropout=RECURRENT_DROPOUT_STRENGTH):

        self.n_timesteps = n_timesteps,
        self.n_feats_inside = n_feats_head
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

    @staticmethod
    def create_loss_weights(n_timesteps):
        """Create loss weights that increase exponentially with time."""
        d = []
        for t in range(n_timesteps):
            d.append(exp(-(n_timesteps - t)))
        return d

    def create_architecture(self):
        """Create the model architecture."""
        # Recurrent section for inside features
        input_shape_inside = (self.n_timesteps, self.n_feats_inside)
        input_inside = Input(shape=input_shape_inside)
        lstm_out_inside = TimeDistributed(Masking(mask_value=0))(input_inside)

        lstm_out_inside = LSTM(self.lstm_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
                               recurrent_dropout=self.rec_dropout,
                               bias_initializer=self.bias_initializer)(lstm_out_inside)
        lstm_out_inside = Dropout(self.dropout)(lstm_out_inside)

        # Recurrent section for outside features
        input_shape_outside = (self.n_timesteps, self.n_feats_outside)
        input_outside = Input(shape=input_shape_outside)
        lstm_out_outside = TimeDistributed(Masking(mask_value=0))(input_outside)
        lstm_out_outside = LSTM(self.lstm_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
                                recurrent_dropout=self.rec_dropout, bias_initializer=self.bias_initializer)(lstm_out_outside)
        lstm_out_outside = Dropout(self.dropout)(lstm_out_outside)

        # Recurrent section for eye-tracking
        input_shape_gaze = (self.n_timesteps, self.n_feats_gaze)
        input_gaze = Input(shape=input_shape_gaze)
        lstm_out_gaze = TimeDistributed(Masking(mask_value=0))(input_gaze)
        lstm_out_gaze = LSTM(self.lstm_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
                             recurrent_dropout=self.rec_dropout, bias_initializer=self.bias_initializer)(lstm_out_gaze)
        lstm_out_gaze = Dropout(self.dropout)(lstm_out_gaze)

        # Action fusion
        lstm_first_concat = concatenate([lstm_out_inside, lstm_out_gaze])
        lstm_first_concat = GRU(self.gru_units, kernel_initializer=self.kernel_initializer, return_sequences=True,
                                recurrent_dropout=self.rec_dropout, bias_initializer=self.bias_initializer)(lstm_first_concat)
        lstm_first_concat = Dropout(self.dropout)(lstm_first_concat)

        # Action-context fusion
        lstm_second_concat = concatenate([lstm_first_concat, lstm_out_outside])
        aux_output = TimeDistributed(Dense(self.dense_units, activation='tanh', kernel_initializer=self.kernel_initializer))(
            lstm_second_concat)
        aux_output = Dropout(self.dropout)(aux_output)

        # Action classification
        main_output = TimeDistributed(Dense(self.dense_units, activation='softmax', kernel_initializer=self.kernel_initializer,
                                            name='aux_classifier'))(
            aux_output)

        # Domain Adaptation section
        flip_layer = GradientReversal(self.lambda_reversal)
        dann_in = flip_layer(aux_output)
        dann_out = Dense(2, activation='softmax', kernel_initializer=self.kernel_initializer, name='domain_classifier')(
            dann_in)

        self.model = Model(inputs=[input_inside, input_outside, input_gaze], outputs=[main_output, dann_out])

        return self.model

    def compile_model(self, loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=None, loss_weights=None):
        """Compile the model."""
        if metrics is None:
            metrics = ['precision']
        if loss_weights is None:
            weights = self.create_loss_weights(self.n_timesteps)
            loss_weights = {'domain_classifier': weights, 'aux_classifier': weights}

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics,
                           sample_weight_mode='temporal', loss_weights=loss_weights)
        return self.model

    def fit_model(self, X_train_head, X_train_outside, X_train_gaze,
                  y_train, y_train_domain, batch_size=128, epochs=1000, patience=30):
        """Fit the model."""

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        model_history = self.model.fit(
            x=[X_train_head, X_train_outside,
               X_train_gaze],
            y=[y_train, y_train_domain],
            epochs=epochs, batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping])

        return model_history

    def predict(self, X_test_head, X_test_outside, X_test_gaze):
        """Predict probabilities on a test set."""
        proba = self.model.predict([X_test_head, X_test_outside, X_test_gaze], batch_size=2)[0]

        return proba

if __name__ == "__main__":

    # Create architecture with random number of features and timesteps.
    da_rnn = DomainAdaptiveRNN(n_timesteps=150, n_feats_head=20, n_feats_outside=10, n_feats_gaze=6)
    da_rnn.create_architecture()
    model = da_rnn.compile_model()
