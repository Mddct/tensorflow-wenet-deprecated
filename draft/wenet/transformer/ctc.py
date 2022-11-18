import tensorflow as tf
from typeguard import check_argument_types


class CTCDense(tf.keras.layers.Layer):
    """CTC module"""

    def __init__(
            self,
            odim: int,
            dropout_rate: float = 0.1,
            reduce: bool = False,
            bias_regularizer=tf.keras.regularizers.l2(1e-6),
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            **kwargs,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super(CTCDense, self).__init__(**kwargs)
        _ = reduce
        self.dropout_rate = dropout_rate
        self.proj = tf.keras.layers.Dense(
            units=odim,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Calculate CTC loss.
        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        Returns:
            ys_hat : [batch, odim]
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        if training:
            inputs = tf.nn.dropout(inputs, self.dropout_rate)
        return self.proj(inputs)

    def log_softmax(self, inputs: tf.Tensor) -> tf.Tensor:
        """log_softmax of frame activations
        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            tf.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return tf.nn.log_softmax(self.proj(inputs), axis=-1)

    def argmax(self, inputs: tf.Tensor) -> tf.Tensor:
        """argmax of frame activations
        Args:
            tf.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            tf.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return tf.argmax(self.proj(inputs), axis=-1)
