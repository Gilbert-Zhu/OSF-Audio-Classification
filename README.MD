# OSF cough audio dataset classification 

This repository provides several machine learning models for [OSF](https://osf.io/tmkud/wiki/home/) cough audio data classification using Keras with tensorflow backend. The dataset can be found [here](https://osf.io/tmkud/files/).

## Models:

1. Using MFCC arrays as input
MFCC_OSF.ipynb (61% accuracy)

2. Transfer learning with pretrained VGG16 network based on ImageNet
TransferLearning_OSF_Melspectrograms.ipynb (83% accuracy)
TransferLearning_OSF_spectrograms.ipynb (80% accuracy)
TransferLearning_OSF_wavelet.ipynb (85% accuracy)

3. Transfer learning with pretrained VGGish network based on Google [AudioSet](https://research.google.com/audioset/index.html)
VGGish-TransferLearning_OSF_Melspectrograms.ipynb (74% accuracy)

Due to timing I didn't do much tuning on hyperparameters. Based on the current results, transfer learning using pretrained VGG16 network has the best performance in terms of acuuracy and confusion matrix, although it takes the longest training time. I believe all the models would perform better given more time for tuning.  

## Reference:

* [Transfer Learning with VGG16](https://github.com/adanRivas/CNN-Audio-Classifier-with-Keras-Tensorflow)

* [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)

* [VGGish to Keras](https://github.com/DTaoo/VGGish)
