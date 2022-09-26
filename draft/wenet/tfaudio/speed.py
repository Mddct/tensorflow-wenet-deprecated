import random

import tensorflow as tf
from wenet.tfaudio.resample import eager_resample


def eager_speed(waveform: tf.Tensor, sr: tf.Tensor,
                speeds: tf.Tensor) -> tf.Tensor:
    """"Why use speed option instead of tempo -s in SoX for speed perturbation"
    https://groups.google.com/forum/#!topic/kaldi-help/8OOG7eE4sZ8

    """

    # index = tf.squeeze(tf.random.categorical(tf.expand_dims(speeds, 0), 1),
    # [1] )
    # speed = tf.gather(speeds, index)[0]
    # speed = random.choice(speeds.numpy())

    resample_rate = tf.cast(tf.cast(sr, dtype=tf.float32) *
                            tf.cast(1.0 / speed, dtype=tf.float32),
                            dtype=tf.int32)

    speed_waveform = eager_resample(waveform=waveform,
                                    orig_freq=sr,
                                    new_freq=resample_rate,
                                    lowpass_filter_width=5)
    return speed_waveform


def speed_fn(waveform: tf.Tensor, sr: tf.Tensor, speeds: tf.Tensor):
    return tf.py_function(eager_speed, [waveform, sr, speeds], waveform.dtype)
