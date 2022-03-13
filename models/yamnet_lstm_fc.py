import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow as tf
from utils.custom_layers import YAMNET_OUT, ReduceTimeWrapper, ReductionOptions, YAMNetWrapper
from utils.params import Params

def _yamnet_pretrained_net(input, params: Params, yamnetOutputType, yamnet_model_handle='https://tfhub.dev/google/yamnet/1'):
  """ returns embeddings extracted from yamnet """
  yamnet_wrapper_dict = YAMNetWrapper(hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet'), yamnetOutputType)(input)
  yamnet_embeddings = yamnet_wrapper_dict['embeddings']
  log_mel_spectrogram = yamnet_wrapper_dict['spectrogram']
  return yamnet_embeddings, log_mel_spectrogram

def _lstm_net(input, params: Params):
  lstm = keras.layers.LSTM(512, return_sequences=True, name='lstm')(input) # many parameters but very good
  return lstm

def _fully_connected_net(input, params: Params):
  fc = keras.layers.TimeDistributed( keras.layers.Dense(512, activation='relu', name='dense_1'), name='time_distributed_1' )(input)
  fc = keras.layers.TimeDistributed( keras.layers.Dense(params.num_classes , name='dense_2'), name='time_distributed_2' )(fc) # softmax could be wrong - delete activation param
  return fc

def _reduce_time(input, params: Params):
  single_out = ReduceTimeWrapper(ReductionOptions.REDUCE_MEAN, name='time_reduction')(input)
  return single_out

#@tf.function
def yamnet_lstm_fc():
  params = Params()
  wave = keras.layers.Input(shape=(None,), batch_size=None, dtype=tf.float32, name='mono_16k_wav_input')
  embeddings, log_mel_spectrogram = _yamnet_pretrained_net(wave, params, yamnetOutputType=[YAMNET_OUT.EMBEDDINGS, YAMNET_OUT.SPECTROGRAM])
  lstm_out = _lstm_net(embeddings, params)
  time_dist_classification = _fully_connected_net(lstm_out, params)
  wave_classification = _reduce_time(time_dist_classification, params)
  model = keras.Model(name='yamnet_lstm_fc', inputs=wave, outputs=[wave_classification, log_mel_spectrogram])
  return model