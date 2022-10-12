from typing import Tuple

import tensorflow as tf

IGNORE_ID = -1


def reverse_pad_list(ys_pad: tf.Tensor,
                     ys_lens: tf.Tensor,
                     pad_value: int = -1) -> tf.Tensor:
    """Reverse padding for the list of tensors.
    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tokenmax).
    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])
    """
    maxlen = tf.shape(ys_pad)[1]
    index = tf.expand_dims(tf.range(0, maxlen), axis=0)  # [1,B]

    index = tf.expand_dims(ys_lens, axis=1) - 1 - index  # [B,B]

    squence_mask = tf.sequence_mask(ys_lens, maxlen=maxlen)
    index = tf.where(squence_mask, index, 0)
    r_ys_pad = tf.gather(params=ys_pad, indices=index, axis=1, batch_dims=1)

    return tf.where(squence_mask, r_ys_pad, pad_value)


def add_sos_eos(ys_pad: tf.Tensor, ys_lens: tf.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Add <sos> and <eos> labels.
    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding
    Returns:
        ys_in (tf.Tensor) : (B, Lmax + 1)
        ys_out (tf.Tensor) : (B, Lmax + 1)
    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=tf.int64)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    ys_pad_shape = tf.shape(ys_pad)
    bs = ys_pad_shape[0]
    sos_tensor = tf.constant(sos, dtype=ys_pad.dtype, shape=[bs, 1])
    eos_tensor = tf.constant(eos, dtype=ys_pad.dtype, shape=[bs, 1])

    ys_pad = tf.where(ys_pad == ignore_id, eos, ys_pad)
    # TODO: for now,  assume ignore id always in tail of uttrance
    ys_in = tf.concat([sos_tensor, ys_pad], axis=1)
    ys_out = tf.concat([ys_pad, eos_tensor], axis=1)

    ys_out = tf.where(
        tf.sequence_mask(ys_lens + 1, maxlen=ys_pad_shape[1] + 1), ys_out,
        ignore_id)
    return ys_in, ys_out


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
    input = tf.where(mask, -float('inf'), input)

    return tf.nn.softmax(input, axis=-1)  # (batch, head, time1, time2)


def mask_softmax_v2(input, mask):
    assert mask is not None
    adder = (1.0 - tf.cast(mask, input.dtype)) * (_large_compatible_negative(
        input.dtype))

    # Since we are adding it to the raw scores before the softmax, this
    # is effectively the same as removing these entirely.
    input += adder

    return tf.nn.softmax(input, axis=-1)


mask_softmax = mask_softmax_v1
