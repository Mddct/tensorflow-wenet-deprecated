import tensorflow as tf
from typeguard import check_argument_types


class CTCDense(tf.keras.layers.Layer):
    """CTC module"""

    def __init__(
            self,
            odim: int,
            encoder_output_size: int,
            dropout_rate: float = 0.0,
            reduce: bool = True,
            bias_regularizer=tf.keras.regularizers.l2(1e-6),
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.ctc_lo = tf.keras.Sequential([
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.Input(shape=[None, eprojs]),
            tf.keras.layers.Dense(
                odim,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
            )
        ])

        # reduction_type = "sum" if reduce else "none"

    def call(self, hs_pad: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Calculate CTC loss.
        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        Returns:
            ys_hat : [batch, odim]
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(hs_pad, training=training)
        return ys_hat
        # ys_hat: (B, L, D) -> (L, B, D)
        # ys_hat = tf.nn.log_softmax(ys_hat, axis=2)
        # loss = tf.nn.ctc_loss(
        #     ys_pad,
        #     ys_hat,
        #     ys_lens,
        #     hlens,
        #     logits_time_major=False,
        #     blank_index=0,  # wenet default blank is 0
        # )

        # return loss

    def log_softmax(self,
                    hs_pad: tf.Tensor,
                    training: bool = False) -> tf.Tensor:
        """log_softmax of frame activations
        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            tf.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return tf.nn.log_softmax(self.ctc_lo(hs_pad, training=training),
                                 axis=-1)

    def argmax(self, hs_pad: tf.Tensor, training: bool = False) -> tf.Tensor:
        """argmax of frame activations
        Args:
            tf.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            tf.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return tf.argmax(self.ctc_lo(hs_pad, training=training), axis=2)
