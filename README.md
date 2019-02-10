# DA-RNN for Manoeuver Anticipation

Domain-Adaptive Recurrent Neural Network for driving manoeuver anticipation, built in Keras. 
Architecture used for the models in the paper "Robust and Subject-Independent Driving Manoeuvre Anticipation through Domain-Adversarial Recurrent Neural Networks" by Tonutti M., Ruffaldi E., et al. (2019)

## Abstract
Through deep learning and computer vision techniques, driving manoeuvres can be predicted accurately a few seconds in advance. Even though adapting a learned model to new drivers and different vehicles is key for robust driver-assistance systems, this problem has received little attention so far. This work proposes to tackle this challenge through domain adaptation, a technique closely related to transfer learning. A proof of concept for the application of a Domain-Adversarial Recurrent Neural Network (DA-RNN) to multi-modal time series driving data is presented, in which domain-invariant features are learned by maximizing the loss of an auxiliary domain classifier. Our implementation is evaluated using a leave-one-driver-out approach on individual drivers from the Brain4Cars dataset, as well as using a new dataset acquired through driving simulations, yielding an average increase in performance of 30\% and 114\% respectively compared to no adaptation. We also show the importance of fine-tuning sections of the network to optimise the extraction of domain-independent features. The results demonstrate the applicability of the approach to driver-assistance systems as well as training and simulation environments.

## Domain adaptive RNN 
![](https://user-images.githubusercontent.com/18726750/52519677-60db2280-2c5f-11e9-8e16-0c0812e8712c.png)

## LSTM-GRU section
![](https://user-images.githubusercontent.com/18726750/52519678-62a4e600-2c5f-11e9-986d-bcba3542fd24.png)
