from enum import Enum
import tensorflow as tf
import tensorflow.keras as keras
from bicycle_bell_sed_models.utils.features import pad_waveform, waveform_to_log_mel_spectrogram_patches

class YAMNET_OUT(int):
  SCORES: int = 0
  EMBEDDINGS: int = 1
  SPECTROGRAM: int = 2

  def nameOfId(id):
    if id == 0: return 'scores'
    if id == 1: return 'embeddings'
    if id == 2: return 'spectrogram'

class YAMNetWrapper(keras.layers.Wrapper):
  def __init__(self, layer, outputType: YAMNET_OUT, name='yamnet_wrapper', **kwargs):
    super(YAMNetWrapper, self).__init__(layer, name=name, **kwargs)
    self.outputTypes = sorted(outputType) if isinstance(outputType, list) else [outputType]
    self.layer = layer

  def call(self, input):
    # better would be self.layer(input)
    result = tf.map_fn(lambda batchItem: {YAMNET_OUT.nameOfId(i): self.layer(batchItem)[i] for i in self.outputTypes}, input, fn_output_signature={YAMNET_OUT.nameOfId(i): tf.float32 for i in self.outputTypes}) #self.layer(batchItem)[1], input) #{YAMNET_OUT.nameOfId(i): self.layer(batchItem)[i] for i in self.outputTypes}, input)
    return result

class ReduceMeanLayer(keras.layers.Layer):
  def __init__(self, axis=0, name='reduce_mean', **kwargs):
    super(ReduceMeanLayer, self).__init__(name=name, **kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)

class ReduceMaxLayer(keras.layers.Layer):
  def __init__(self, axis=0, name='reduce_max', **kwargs):
    super(ReduceMaxLayer, self).__init__(name=name, **kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_max(input, axis=self.axis)

class ArgMaxLayer(keras.layers.Layer):
  def __init__(self, axis=0, name='arg_max', **kwargs):
    super(ArgMaxLayer, self).__init__(name=name, **kwargs)
    self.axis = axis

  def call(self, input):
    return tf.cast(tf.argmax(input, axis=self.axis), tf.float32)

class ReductionOptions(Enum):
  REDUCE_MEAN = ReduceMeanLayer
  REDUCE_MAX = ReduceMaxLayer
  LSTM_CELL = keras.layers.LSTM
  AVERAGE_POOLING_1D = keras.layers.GlobalAveragePooling1D
  ARGMAX = ArgMaxLayer
  DENSE_CELL = keras.layers.Dense

class ReduceTimeWrapper(keras.layers.Wrapper):
  def __init__(self, mode: ReductionOptions, layer_axis=1, num_classes=1, name='time_dissolved', **kwargs):
    self.mode = mode
    if mode == ReductionOptions.REDUCE_MEAN:
      layer = ReduceMeanLayer(axis=layer_axis, name='reduce_mean_output')

    elif mode == ReductionOptions.REDUCE_MAX:
      layer = ReduceMaxLayer(axis=layer_axis, name='reduce_max_output')

    elif mode == ReductionOptions.LSTM_CELL:
      layer = keras.layers.LSTM(num_classes, return_sequences=False, recurrent_activation='softmax', name='lstm_output')

    elif mode == ReductionOptions.AVERAGE_POOLING_1D:
      layer = keras.layers.GlobalAveragePooling1D(data_format='channels_last', name='avg_pooling_1d_output')

    elif mode == ReductionOptions.ARGMAX:
      layer = ArgMaxLayer(axis=layer_axis, name='arg_max_output')

    elif mode == ReductionOptions.DENSE_CELL:
      layer = keras.layers.Dense(units=num_classes, activation='sigmoid', name='dense_output')

    else:
      raise NotImplementedError('Not a member of ReductionOptions!')
    super(ReduceTimeWrapper, self).__init__(layer, name=name, **kwargs)
  
  def call(self, input):
   result = self.layer(input) #tf.map_fn(lambda batchItem: self.layer(batchItem), input)
   return result

class PadWaveformLayer(keras.layers.Layer):
  def __init__(self, params, name='padded_wave', **kwargs):
    super(PadWaveformLayer, self).__init__(name=name, **kwargs)
    self.params = params

  def call(self, input):
    return tf.vectorized_map(lambda batchItem: pad_waveform(batchItem, self.params), input)

class LogMelSpectrogramTransformLayer(keras.layers.Layer):
  def __init__(self, params, name='log_mel_spectrogram_transform', **kwargs):
    super(LogMelSpectrogramTransformLayer, self).__init__(name=name, **kwargs)
    self.params = params

  def call(self, input):
    return tf.vectorized_map(lambda batchItem: waveform_to_log_mel_spectrogram_patches(batchItem, self.params), input)

class ClassOutput(keras.layers.Layer):
  def __init__(self, name='class_output', **kwargs):
    super(ClassOutput, self).__init__(name=name, **kwargs)

class LogMelSpectrogramOutput(keras.layers.Layer):
  def __init__(self, name='log_mel_spectrogram_output', **kwargs):
    super(LogMelSpectrogramOutput, self).__init__(name=name, **kwargs)
