import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow as tf
from bicycle_bell_sed_models.utils.custom_layers import YAMNET_OUT, ClassOutput, LogMelSpectrogramOutput, ReduceTimeWrapper, ReductionOptions, YAMNetWrapper
from bicycle_bell_sed_models.utils.params_yamnet import Params

def _yamnet_pretrained_net(input, params: Params, yamnetOutputType, yamnet_model_handle='https://tfhub.dev/google/yamnet/1'):
  """ returns embeddings extracted from yamnet """
  yamnet_wrapper_dict = YAMNetWrapper(hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet'), yamnetOutputType)(input)
  return yamnet_wrapper_dict['embeddings'], yamnet_wrapper_dict['spectrogram']

def _lstm_net(input, params: Params):
  lstm = keras.layers.LSTM(512, return_sequences=True, name='lstm')(input) # many parameters but very good
  return lstm

def _fully_connected_net(input, params: Params):
  fc = keras.layers.TimeDistributed( keras.layers.Dense(512, activation='relu', name='dense_1'), name='time_distributed_1' )(input)
  fc = keras.layers.TimeDistributed( keras.layers.Dense(params.num_classes , name='dense_2'), name='time_distributed_2' )(fc) # softmax could be wrong - delete activation param
  return fc

def _dissolve_time(input, params: Params):
  single_out = ReduceTimeWrapper(ReductionOptions.REDUCE_MEAN)(input)
  single_out = keras.layers.Reshape((), name='reshape')(single_out)
  return single_out

#@tf.function
def yamnet_lstm_fc():
  """
  Returns an uncompiled keras model of the extended yamnet.

  Model Input:
  - Waveform 16000 Hz mono channel audio with shape (None,)

  Model Output:
  - tuple of (wave_classification, log_mel_spectrogram)
    - wave_classification is a scalar
    - log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands] 
  """
  params = Params()
  wave = keras.layers.Input(shape=(None,), batch_size=None, dtype=tf.float32, name='wav_16000_mono_input')
  embeddings, log_mel_spectrogram = _yamnet_pretrained_net(wave, params, yamnetOutputType=[YAMNET_OUT.EMBEDDINGS, YAMNET_OUT.SPECTROGRAM])
  embeddings = keras.layers.Layer(name='yamnet_embeddings')(embeddings)
  lstm_out = _lstm_net(embeddings, params)
  time_dist_classification = _fully_connected_net(lstm_out, params)
  wave_classification = _dissolve_time(time_dist_classification, params)

  class_output = ClassOutput()(wave_classification)
  log_mel_spectrogram_output = LogMelSpectrogramOutput()(log_mel_spectrogram)
  model = keras.Model(name='yamnet_lstm_fc', inputs=wave, outputs=[class_output, log_mel_spectrogram_output])
  return model