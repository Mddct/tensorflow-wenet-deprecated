"""Encoder definition."""
from typing import Optional, Tuple

import tensorflow as tf
from typeguard import check_argument_types
from wenet.transformer.attention import (MultiHeadedAttention,
                                         RelPositionMultiHeadedAttention)
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.embedding import (NoPositionalEncoding,
                                         PositionalEncoding,
                                         RelPositionalEncoding)
from wenet.transformer.encoder_layer import (ConformerEncoderLayer,
                                             TransformerEncoderLayer)
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import (Conv2dSubsampling4,
                                           Conv2dSubsampling6,
                                           Conv2dSubsampling8,
                                           LinearNoSubsampling)
from wenet.utils.mask import (add_optional_chunk_mask, get_next_cache_start,
                              make_pad_mask)


class BaseEncoder(tf.keras.layers.Layer):

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "abs_pos",
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: Optional[tf.keras.layers.Layer] = None,
            use_dynamic_left_chunk: bool = False,
            bias_regularizer=tf.keras.regularizers.l2(1e-6),
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[tf.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            eps=1e-5,
        )
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

    def call(
        self,
        xs: tf.Tensor,
        xs_lens: tf.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Embed positions in tensor.
        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: tf.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        masks = tf.expand_dims(tf.sequence_mask(xs_lens), axis=2)  # (B, T, 1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # offset = [0, 0....] for training
        fake_offset = tf.ones(tf.shape(xs)[0], dtype=tf.int64)
        xs, pos_emb, masks = self.embed(xs, masks, fake_offset, training=True)
        mask_pad = masks  # (B, T/subsample_rate, 1)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs,
                                          chunk_masks,
                                          pos_emb,
                                          mask_pad,
                                          training=True)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk_batch(self):
        # TOO
        pass

    # helper function for export savedmodel : one uttrance
    def forward_chunk(
        self,
        xs: tf.Tensor,
        xs_lens: tf.Tensor,
        offset: tf.Tensor,
        required_cache_size: tf.Tensor,
        att_cache: tf.Tensor,
        cnn_cache: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """ Forward just one chunk
        Args:
            xs (tf.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (tf.Tensor): [B] current offset in encoder output time stamp
            required_cache_size (tf.Tensor): [] cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (tf.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
                if cache_t1 == 0, means fake cache, first chunk
            cnn_cache (tf.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`
        Returns:
            tf.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            tf.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            tf.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """
        chunk_mask = make_pad_mask(xs_lens)  # [B, T]
        chunk_mask = tf.expand_dims(chunk_mask, axis=2)  # [B, T, 1]
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, chunk_mask = self.embed(xs,
                                             chunk_mask,
                                             offset,
                                             training=False)
        att_cache_shape = tf.shape(att_cache)
        elayers, cache_t1 = att_cache_shape[0], att_cache_shape[2]
        chunk_size = tf.shape(xs)[1]
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size,
                                               apply_dropout=False,
                                               training=False)

        next_cache_start = get_next_cache_start(attention_key_size,
                                                required_cache_size)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                # we don't need a mask for self attention
                None,
                pos_emb,
                att_cache=att_cache[i:i + 1],
                cnn_cache=cnn_cache[i],
                training=False,
            )
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(tf.squeeze(new_cnn_cache, axis=0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        r_att_cache = tf.concat(r_att_cache, axis=0)
        r_cnn_cache = tf.concat(r_cnn_cache, axis=0)

        return (xs, r_att_cache, r_cnn_cache)


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: Optional[tf.keras.layers.Layer] = None,
        use_dynamic_left_chunk: bool = False,
    ):
        """ Construct TransformerEncoder
        See Encoder for the meaning of each parameter.
        """
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk)
        self.encoders = [
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ]


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: Optional[tf.keras.layers.Layer] = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
    ):
        """Construct ConformerEncoder
        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk)

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation_type,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel,
                                  activation_type, cnn_module_norm, causal)

        self.encodrs = [
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ]
