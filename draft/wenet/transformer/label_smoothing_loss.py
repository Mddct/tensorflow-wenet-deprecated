
import tensorflow as tf

## NOTE: tempoary
# LabelSmoothingLoss = partial(tf.keras.losses.CategoricalCrossentropy,
#                              from_logits=True)



class LabelSmoothingLoss(tf.keras.losses.Loss):

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name="LabelSmoothingLoss"):
        super().__init__(reduction, name)

        self.smoothing = smoothing
        self.reduction = reduction
        self.padding_idx = padding_idx
        self.size_ = size

        self.low_confidence = self.smoothing / (self.size - 1)
        self.confidence = 1 - self.smoothing

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: [B, L]
            y_pred: logit [B, L, V]

        Returns:
            losses [B]
        """

        ignore = tf.expand_dims(y_true == self.padding_idx, axis=2) # [B, L, 1]
        y_true = tf.one_hot(y_true,
                   depth=self.size_,
                   on_value=self.confidence
                   off_value=self.low_confidence) # [B, L, V]
        y_pred = tf.nn.log_softmax(y_pred)
        output = y_true * tf.exp(y_true - y_pred)# [B, L, V]
        output = tf.where(ignore, output, 0) # [B, L, V]

        output = tf.reduce_sum(tf.reduce_sum(output, axis=-1) , axis=-1)# [B]

        # NOTE: distributed strateggy need sum average all global size
        return output
