import tensorflow as tf


def subsequent_mask(size: tf.Tensor) -> tf.Tensor:
    input = tf.ones((size, size), dtype=tf.bool)
    return tf.linalg.band_part(input, -1, 0)


@tf.function(input_signature=[
    tf.TensorSpec([], dtype=tf.int32),
    tf.TensorSpec([], dtype=tf.int32),
    tf.TensorSpec([], dtype=tf.int32),
])
def subsequent_chunk_mask(
    size: tf.Tensor,
    chunk_size: tf.Tensor,
    num_left_chunks: tf.Tensor,
):

    index = tf.range(0, size)

    mask_seq = tf.minimum(((index // chunk_size) + 1) * chunk_size, size)
    mask_1 = tf.sequence_mask(mask_seq, maxlen=size)

    def limit_left_fn():
        left_mask = tf.maximum(
            (index // chunk_size - num_left_chunks) * chunk_size, 0)

        mask_2 = tf.sequence_mask(left_mask, maxlen=size)
        return mask_1 & (~mask_2)

    return tf.cond(
        tf.not_equal(num_left_chunks, -1),
        limit_left_fn,
        lambda: mask_1,
    )


def add_optional_chunk_mask(input: tf.Tensor, mask: tf.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int):

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.bool)
    ])
    def get_mask(xs: tf.Tensor, masks: tf.Tensor):
        xs_shape = tf.shape(xs)
        if use_dynamic_chunk:
            max_len = xs_shape[1]
            if decoding_chunk_size < 0:
                chunk_size = max_len
                num_left_chunks = tf.constant(-1, dtype=xs_shape.dtype)
            elif decoding_chunk_size > 0:
                chunk_size = tf.convert_to_tensor(decoding_chunk_size,
                                                  dtype=xs_shape.dtype)
                num_left_chunks = tf.convert_to_tensor(
                    num_decoding_left_chunks, dtype=tf.int64)
            else:
                chunk_size = tf.random.uniform(shape=[],
                                               minval=1,
                                               maxval=max_len,
                                               dtype=xs_shape.dtype)
                num_left_chunks = tf.constant(-1, dtype=xs_shape.dtype)

                if chunk_size > max_len // 2:
                    chunk_size = max_len
                else:
                    chunk_size = chunk_size % 25 + 1
                    if use_dynamic_left_chunk:
                        max_left_chunks = (max_len - 1) // chunk_size
                        num_left_chunks = tf.random.uniform(
                            shape=[],
                            minval=0,
                            maxval=max_left_chunks,
                            dtype=xs_shape.dtype)

        elif static_chunk_size > 0:
            num_left_chunks = tf.convert_to_tensor(num_decoding_left_chunks,
                                                   dtype=xs_shape.dtype)
            chunk_size = tf.convert_to_tensor(static_chunk_size)
        else:
            return masks

        chunk_masks = subsequent_chunk_mask(x_shape[1], chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = tf.expand_dims(chunk_masks, 0)  #[1,L, L]
        chunk_masks = masks & chunk_masks  # (B, L, L)

        return chunk_masks

    return get_mask(input, mask)


def make_pad_mask(lengths: tf.Tensor, max_len=None):
    return ~tf.sequence_mask(lengths, maxlen=max_len)


def make_no_pad_mask(lengths: tf.Tensor, max_len):
    return tf.sequence_mask(lengths)


def get_next_cache_start(required_cache_size: tf.Tensor,
                         attention_key_size: tf.Tensor):
    if required_cache_size < 0:
        next_cache_start = tf.constant(0, dtype=required_cache_size.dtype)
    elif required_cache_size == 0:
        next_cache_start = attention_key_size
    else:
        next_cache_start = tf.maximum(attention_key_size - required_cache_size,
                                      0)
    return next_cache_start
