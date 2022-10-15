import tensorflow as tf
from typeguard import check_argument_types


class CTC(tf.keras.layers.Layer):
    """CTC module"""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
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
            tf.keras.layers.Dense(odim)
        ])

        reduction_type = "sum" if reduce else "none"

    def call(self,
             hs_pad: tf.Tensor,
             hlens: tf.Tensor,
             ys_pad: tf.Tensor,
             ys_lens: tf.Tensor,
             training: bool = True) -> tf.Tensor:
        """Calculate CTC loss.
        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        Returns:
            loss_ctc: [batch]
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(hs_pad, training=training)
        # ys_hat: (B, L, D) -> (L, B, D)
        # ys_hat = tf.nn.log_softmax(ys_hat, axis=2)
        loss = tf.nn.ctc_loss(
            ys_pad,
            ys_hat,
            ys_lens,
            hlens,
            logits_time_major=False,
            blank_index=0,  # wenet default blank is 0
        )

        return loss

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
