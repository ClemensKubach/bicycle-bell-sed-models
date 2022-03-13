import tensorflow as tf
import tensorflow.keras as keras
from bicycle_bell_sed_models.utils.custom_layers import LogMelSpectrogramTransformLayer, PadWaveformLayer, ReduceTimeWrapper, ReductionOptions
from bicycle_bell_sed_models.utils.params import Params

def _conv_net(input, params: Params):
  mels = params.mel_bands
  kernel_size = 32 # 3?
  filters = mels
  pool_size = mels - kernel_size + 1
  convNet = keras.layers.TimeDistributed(keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))(input)
  convNet = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(convNet)
  convNet = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=pool_size))(convNet)
  convNet = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(convNet)
  convNet = keras.layers.TimeDistributed(keras.layers.Flatten())(convNet)
  return convNet

def _lstm_net(input, params: Params):
  mels = params.mel_bands
  lstmNet = keras.layers.LSTM(units=mels, go_backwards=True, activation='tanh', dropout=0.1, return_sequences=True, # dropout=0.3
                            recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True)(input)
  lstmNet = keras.layers.LSTM(units=mels, go_backwards=True, activation='tanh', dropout=0.1, return_sequences=True, # dropout=0.3
                            recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True)(lstmNet)
  return lstmNet

def _fully_connected_net(input, params: Params):
  mels = params.mel_bands
  fcNet = keras.layers.TimeDistributed(keras.layers.Dense(units=mels, activation='relu'))(input)
  fcNet = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(fcNet)
  fcNet = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))(fcNet)
  fcNet = keras.layers.Flatten()(fcNet)
  return fcNet

def _reduce_time(input, params: Params):
  single_out = ReduceTimeWrapper(ReductionOptions.REDUCE_MEAN, name='time_reduction')(input)
  return single_out

def _crnn_core(features, params: Params):
  input = keras.layers.Reshape((params.patch_frames, params.patch_bands, 1), input_shape=(params.patch_frames, params.patch_bands))(features) # (frameNum, mels, 1)
  convNet = _conv_net(input, params)
  lstmNet = _lstm_net(convNet, params)
  fcNet = _fully_connected_net(lstmNet, params)
  wave_classification = _reduce_time(fcNet, params)
  return wave_classification

#@tf.function
def crnn():
  params = Params()
  # wave is gegen als 16k mono normed float vor
  wave = keras.layers.Input(shape=(None,), batch_size=None, dtype=tf.float32, name='mono_16k_wav_input')
  wave_padded = PadWaveformLayer(params)(wave)
  log_mel_spectrogram, features = LogMelSpectrogramTransformLayer(params)(wave_padded)
  # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]
  # features has shape [<# patches>, <# STFT frames in an patch>, params.mel_bands]
  wave_classification = _crnn_core(features, params)
  
  model = keras.Model(name='crnn', inputs=wave, outputs=[wave_classification, log_mel_spectrogram])
  return model