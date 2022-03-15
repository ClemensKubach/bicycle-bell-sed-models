import tensorflow.keras as keras
import os

def visualizeArchitecture(model):
  return keras.utils.plot_model(model, expand_nested=True, show_shapes=True, to_file=os.path.join('visualizations', f'{model.name}.png'))
