import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
from bicycle_bell_sed_models.utils.custom_layers import YAMNET_OUT, ClassOutput, LogMelSpectrogramOutput, ReduceTimeWrapper, ReductionOptions, YAMNetWrapper
from bicycle_bell_sed_models.utils.params_yamnet import Params

def _yamnet_pretrained_net(input, yamnetOutputType, yamnet_model_handle='https://tfhub.dev/google/yamnet/1'):
  """ returns embeddings extracted from yamnet """
  BICYCLE_BELL_CLASS_INDEX = tf.constant(198)
  yamnet_wrapper_dict = YAMNetWrapper(hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet'), yamnetOutputType)(input)
  return yamnet_wrapper_dict['scores'][:, :, BICYCLE_BELL_CLASS_INDEX], yamnet_wrapper_dict['spectrogram']

def _dissolve_time(input):
  single_out = ReduceTimeWrapper(ReductionOptions.REDUCE_MEAN)(input)
  return single_out


#@tf.function
def yamnet_base():
  """
  Returns an uncompiled keras model of the base yamnet.

  Model Input:
  - Waveform 16000 Hz mono channel audio with shape (None,)

  Model Output:
  - tuple of (wave_classification, log_mel_spectrogram)
    - wave_classification is a scalar
    - log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands] 
  """
  params = Params()
  wave = keras.layers.Input(shape=(None,), batch_size=None, dtype=tf.float32, name=f'wav_{int(params.sample_rate)}_mono_input')
  scores, log_mel_spectrogram = _yamnet_pretrained_net(wave, yamnetOutputType=[YAMNET_OUT.SCORES, YAMNET_OUT.SPECTROGRAM])
  scores = keras.layers.Layer(name='scores')(scores)
  wave_classification = _dissolve_time(scores)

  class_output = ClassOutput()(wave_classification)
  log_mel_spectrogram_output = LogMelSpectrogramOutput()(log_mel_spectrogram)
  model = keras.Model(name='yamnet_base', inputs=wave, outputs=[class_output, log_mel_spectrogram_output])
  return model