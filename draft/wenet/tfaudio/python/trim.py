import tensorflow as tf
from wenet.tfaudio.python import utils


def _signal_to_frame_nonsilent(
    waveform,
    frame_length=40 * 16,
    frame_shift=20 * 16,
    top_db=60,
):
    # Compute the MSE for the signal
    mse = utils.rms(waveform,
                    frame_length=frame_length,
                    frame_shift=frame_shift)  # [channel, mse_value]

    # Convert to decibels and slice out the mse channel
    db = utils.amplitude_to_db(mse,
                               ref=tf.reduce_max(mse, axis=-1),
                               top_db=top_db)
    return db > -top_db


def trim(
    waveform,
    top_db=60,
    frame_length=2048,
    frame_shift=512,
):
    """Tri leading and trailing silence from an audio signal.

    """

    frame_length = tf.convert_to_tensor(frame_length, dtype=tf.int64)
    frame_shift = tf.convert_to_tensor(frame_shift, dtype=tf.int64)
    non_silent = _signal_to_frame_nonsilent(
        waveform=waveform,
        frame_length=frame_length,
        frame_shift=frame_length,
        top_db=top_db,
    )

    forward = tf.cast(non_silent, tf.int8)
    backward = tf.reverse(forward, axis=[-1])

    starts = tf.argmax(forward, axis=-1)
    ends = tf.cast(tf.shape(forward)[-1], dtype=tf.int64) - tf.argmax(
        backward, axis=-1)[0],

    # TODO: wrong cal
    start = tf.cast(starts[0], dtype=tf.int64) * frame_length
    end = tf.cast(ends[0], dtype=tf.int64) * frame_length
    return waveform[:, start:end]


def trim_greedy(input, axis, epsilon, name=None):
    """
    Trim the noise from beginning and end of the audio.
    Args:
      input: An audio Tensor.
      axis: The axis to trim.
      epsilon: The max value to be considered as noise.
      name: A name for the operation (optional).
    Returns:
      A tensor of start and stop with shape `[..., 2, ...]`.
    """
    shape = tf.shape(input, out_type=tf.int64)
    length = shape[axis]

    nonzero = tf.math.greater(input, epsilon)
    check = tf.reduce_any(nonzero, axis=axis)

    forward = tf.cast(nonzero, tf.int8)
    reverse = tf.reverse(forward, [axis])

    start = tf.where(check, tf.argmax(forward, axis=axis), length)
    stop = tf.where(check, tf.argmax(reverse, axis=axis),
                    tf.constant(0, tf.int64))
    stop = length - stop

    return input[start[0]:stop[0], :]
    # return tf.stack([start, stop], axis=axis)


# raw = tf.io.read_file("/users/mddct/1.wav")
# wav, sr = tf.audio.decode_wav(raw)
# # wav = tf.transpose(wav, [1, 0])
# trim_wav = trim_greedy(wav, axis=0, epsilon=0.1)

# out = tf.audio.encode_wav(trim_wav, sample_rate=16000)

# tf.io.write_file("out.wav", out)
# # print(out)
