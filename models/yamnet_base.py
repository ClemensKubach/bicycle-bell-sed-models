import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from utils.custom_layers import YAMNET_OUT, ReduceTimeWrapper, ReductionOptions, YAMNetWrapper

def _yamnet_pretrained_net(input, yamnetOutputType, yamnet_model_handle='https://tfhub.dev/google/yamnet/1'):
  """ returns embeddings extracted from yamnet """
  yamnet_wrapper_dict = YAMNetWrapper(hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet'), yamnetOutputType)(input)
  yamnet_scores = yamnet_wrapper_dict['scores']
  log_mel_spectrogram = yamnet_wrapper_dict['spectrogram']
  return yamnet_scores, log_mel_spectrogram

def _reduce_time(input):
  single_out = ReduceTimeWrapper(ReductionOptions.REDUCE_MEAN, name='time_reduction')(input)
  return single_out

BICYCLE_BELL_CLASS_INDEX = 198

#@tf.function
def yamnet_base():
  wave = keras.layers.Input(shape=(None,), batch_size=None, dtype=tf.float32, name='mono_16k_wav_input')
  scores, log_mel_spectrogram = _yamnet_pretrained_net(wave, yamnetOutputType=[YAMNET_OUT.SCORES, YAMNET_OUT.SPECTROGRAM])
  wave_classification = _reduce_time(scores)
  model = keras.Model(name='yamnet_base', inputs=wave, outputs=[wave_classification, log_mel_spectrogram])
  return model