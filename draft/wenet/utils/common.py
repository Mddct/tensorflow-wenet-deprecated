import tensorflow as tf


def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.
    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16
    Args:
      tensor_type: a dtype to determine the type.
    Returns:
      a large negative number.
    """
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9


def mask_softmax_v1(input, mask):
    inputs = tf.where(mask, -float('inf'), inputs)

    return tf.nn.softmax(input, axis=-1)  # (batch, head, time1, time2)


def mask_softmax_v2(input, mask):
    assert mask is not None
    adder = (1.0 - tf.cast(mask, inputs.dtype)) * (_large_compatible_negative(
        inputs.dtype))

    # Since we are adding it to the raw scores before the softmax, this
    # is effectively the same as removing these entirely.
    inputs += adder

    return tf.nn.softmax(input, axis=-1)


mask_softmax = mask_softmax_v2
