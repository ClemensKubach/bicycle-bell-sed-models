# bicycle-bell-sed-models

**Author: Clemens Kubach**

This is a collection of the models for my bachelor thesis about detecting atomatically the sound event of cyclists using their bicycle bell.

You can use this package with:
```
pip3 install git+https://github.com/ClemensKubach/bicycle-bell-sed-models.git
```

There are 3 model configurations:
- The pre-trained YAMNet base model without any transfer learning. The resulting probability values for the class "Bicycle bell" is directly taken out of the results of all 521 classes.
- An extended pre-trained YAMNet model with transfer learning using embeddings of the base model.
- A model based on an CRNN architecture without any pre-training.

YAMNet is documented here:
https://github.com/tensorflow/models/tree/master/research/audioset/yamnet


Written in Python 3.9.7.