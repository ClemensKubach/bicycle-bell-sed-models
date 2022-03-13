import tensorflow.keras as keras
import tensorflow as tf
import csv
import numpy as np

def visualizeArchitecture(model):
  return keras.utils.plot_model(model, show_shapes=True, to_file=f'{model.name}.png')
