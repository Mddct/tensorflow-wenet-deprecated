import tensorflow as tf
from wenet.tfaudio.cc.ops import gen_x_op
from wenet.tfaudio.python.resample import eager_resample


# channel first
def eager_speed(waveform: tf.Tensor, sr: tf.Tensor,
                speeds: tf.Tensor) -> tf.Tensor:
    """"Why use speed option instead of tempo -s in SoX for speed perturbation"
    https://groups.google.com/forum/#!topic/kaldi-help/8OOG7eE4sZ8

    """

    # equal random.choince([0.9, 1.0, 1.1])
    distributed = tf.ones([1, tf.shape(speeds)[0]])
    index = tf.random.categorical(distributed, 1)[0][0]
    speed = speeds[index]

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


def speed_fn_v1(waveform: tf.Tensor, sr: tf.Tensor, speeds: tf.Tensor):
    return tf.py_function(eager_speed, [waveform, sr, speeds], waveform.dtype)


# channel last
@tf.function
def speed_fn_v2(waveform: tf.Tensor, sr: tf.Tensor,
                speeds: tf.Tensor) -> tf.Tensor:

    distributed = tf.ones([1, tf.shape(speeds)[0]])
    index = tf.random.categorical(distributed, 1)[0][0]
    speed = tf.gather(speeds, index)
    # print(speed)
    if speed == 1.0:
        return waveform
    resample_rate = tf.cast(tf.cast(sr, dtype=tf.float32) * speed,
                            dtype=tf.int32)
    return tf.squeeze(
        gen_x_op.speed_op(waveform, sr, resample_rate, lowpass_filter_width=5))


@tf.function
def speed_fn_v3(waveform: tf.Tensor, sr: tf.Tensor,
                speeds: tf.Tensor) -> tf.Tensor:
    distributed = tf.ones([1, tf.shape(speeds)[0]])
    index = tf.random.categorical(distributed, 1)[0][0]
    speed = tf.gather(speeds, index)
    if speed == 1.0:
        return waveform
    resample_rate = tf.cast(tf.cast(sr, dtype=tf.float32) * speed,
                            dtype=tf.int32)
    return tfio.audio.resample(waveform, tf.cast(sr, tf.int64),
                               tf.cast(resample_rate, tf.int64))


speed = speed_fn_v3

# print(speed_fn(tf.ones(100, 1), sr=16000, speeds=[0.9, 1.0, 1.1]))
