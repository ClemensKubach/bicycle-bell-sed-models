import tensorflow as tf
import tensorflow.keras as keras
from bicycle_bell_sed_models.utils.custom_layers import ClassOutput, LogMelSpectrogramOutput, LogMelSpectrogramTransformLayer, PadWaveformLayer, ReduceTimeWrapper, ReductionOptions
from bicycle_bell_sed_models.utils.params_crnn import Params


def _conv_net(input, params: Params):
    mels = params.mel_bands
    kernel_size = 32
    filters = mels
    pool_size = mels - kernel_size + 1
    convNet = keras.layers.TimeDistributed(
        keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', name='conv1d'), name='time_distributed_conv1d')(input)
    convNet = keras.layers.TimeDistributed(keras.layers.BatchNormalization(name='batch_normalized_1'), name='time_distributed_batch_normalized_1')(convNet)
    convNet = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(pool_size=pool_size, name='max_pooling_1d'), name='time_distributed_max_pooling_1d')(convNet)
    convNet = keras.layers.TimeDistributed(keras.layers.Dropout(0.3, name='dropout_1'), name='time_distributed_dropout_1')(convNet)
    convNet = keras.layers.TimeDistributed(keras.layers.Flatten(name='flatten_1'), name='time_distributed_flatten_1')(convNet)
    return convNet


def _lstm_net(input, params: Params):
    mels = params.mel_bands
    lstmNet = keras.layers.LSTM(units=mels, go_backwards=True, activation='tanh', return_sequences=True,
                                dropout=0.3,
                                recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, name='lstm_1')(input)
    lstmNet = keras.layers.LSTM(units=mels, go_backwards=True, activation='tanh', return_sequences=True,
                                dropout=0.3,
                                recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, name='lstm_2')(lstmNet)
    return lstmNet


def _fully_connected_net(input, params: Params):
    mels = params.mel_bands
    fcNet = keras.layers.TimeDistributed(keras.layers.Dense(units=mels, activation='relu', name='dense_1'), name='time_distributed_dense_1')(input)
    fcNet = keras.layers.TimeDistributed(keras.layers.BatchNormalization(name='batch_normalized_2'), name='time_distributed_batch_normalized_2')(fcNet)
    fcNet = keras.layers.TimeDistributed(keras.layers.Dense(1, activation=params.classifier_activation, name='dense_frame_output'), name='time_distributed_dense_frame_output')(fcNet)
    fcNet = keras.layers.Flatten(name='flatten_2')(fcNet)
    return fcNet

def _dissolve_time(input, params: Params):
    single_out = ReduceTimeWrapper(ReductionOptions.REDUCE_MEAN, layer_axis=1)(input)
    return single_out


def _crnn_core(log_mel_spec, params: Params):
    # without patching
    input = keras.layers.Reshape((-1, params.mel_bands, 1),
                                 input_shape=(-1, params.mel_bands), name='reshape')(log_mel_spec)
    convNet = _conv_net(input, params)
    lstmNet = _lstm_net(convNet, params)
    fcNet = _fully_connected_net(lstmNet, params)
    wave_classification = _dissolve_time(fcNet, params)
    return wave_classification


# @tf.function
def crnn() -> keras.Model:
    """
  Returns an uncompiled keras model of the CRNN.

  Model Input:
  - Waveform 16000 Hz mono channel audio with shape (None,)

  Model Output:
  - wave_classification is a scalar

  inactive multi-output version: 
  - tuple of (wave_classification, log_mel_spectrogram)
    - wave_classification is a scalar
    - log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands] 

  """
    params = Params()
    wave = keras.layers.Input(shape=(None,), batch_size=None, dtype=tf.float32, name=f'wav_{int(params.sample_rate)}_mono_input')
    wave_padded = PadWaveformLayer(params)(wave)
    log_mel_spectrogram, features = LogMelSpectrogramTransformLayer(params)(wave) #(wave_padded)
    # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]
    # features has shape [<# patches>, <# STFT frames in an patch>, params.mel_bands]
    wave_classification = _crnn_core(log_mel_spectrogram, params)

    class_output = ClassOutput()(wave_classification)
    log_mel_spectrogram_output = LogMelSpectrogramOutput()(log_mel_spectrogram)
    #model = keras.Model(name='crnn', inputs=wave, outputs=[class_output, log_mel_spectrogram_output])
    model = keras.Model(name='crnn', inputs=wave, outputs=class_output)
    return model
