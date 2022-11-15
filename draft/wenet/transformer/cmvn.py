import tensorflow as tf


class GlobalCMVN(tf.keras.layers.Layer):

    def __init__(self,
                 mean: tf.Tensor,
                 istd: tf.Tensor,
                 norm_var: bool = True,
                 trainable=False,
                 name='CMVN',
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.norm_var = norm_var
        # self.mean = mean
        # self.istd = istd
        self.istd = tf.Variable(istd, dtype=istd.dtype)
        self.mean = tf.Variable(mean, dtype=mean.dtype)

    def call(self, inputs):
        x = inputs
        x = tf.subtract(x, self.mean)
        if self.norm_var:
            x = tf.multiply(x, self.istd)
        return x
