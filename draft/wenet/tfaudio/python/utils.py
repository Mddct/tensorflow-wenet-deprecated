import tensorflow as tf


def rms(samples: tf.Tensor,
        frame_length: tf.Tensor,
        frame_shift: tf.Tensor,
        methods="samples") -> tf.Tensor:
    """Compute root-mean-square(RMS) value for each audio samples

    samples: tf.Tensor, shape [channels, samples]
    length: tf.Tensor,  dtype=tf.int64
    methods samples or mel 
    """
    assert methods in ["samples", "mel"]
    frames = tf.signal.frame(samples,
                             frame_length=frame_length,
                             frame_step=frame_shift,
                             pad_end=False)
    # Calculate power
    power = tf.reduce_mean(tf.math.abs(frames), axis=-1)

    return tf.math.sqrt(power)


def amplitude_to_db(magnitude, ref=1.0, amin=1e-5, top_db=80.0):
    """if top_db == -200, no top_db
    """

    magnitude = tf.math.abs(magnitude)

    ref_value = tf.math.abs(ref)

    power = tf.math.square(magnitude)

    return power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)


def log10(x):
    return tf.divide(tf.math.log(x), tf.math.log(tf.constant(10,
                                                             dtype=x.dtype)))


def power_to_db(power, ref, amin=1e-10, top_db=80.0):

    ref = tf.convert_to_tensor(ref, dtype=tf.float32)
    amin = tf.convert_to_tensor(amin, dtype=tf.float32)
    top_db = tf.convert_to_tensor(top_db, dtype=tf.float32)
    # tf.assert_greater(top_db, 0.0)

    magnitude = tf.math.abs(power)
    ref_value = tf.math.abs(ref)

    log_spec = tf.math.multiply(10.0, log10(tf.maximum(amin, magnitude)))
    log_spec -= tf.math.multiply(10.0, log10(tf.maximum(amin, ref_value)))

    return tf.where(log_spec > top_db, top_db, log_spec)
