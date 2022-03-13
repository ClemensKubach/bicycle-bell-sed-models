import tensorflow.keras as keras

def visualizeArchitecture(model):
  return keras.utils.plot_model(model, show_shapes=True, to_file=f'{model.name}.png')
