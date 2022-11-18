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
        super(GlobalCMVN, self).__init__(trainable, name, dtype, dynamic,
                                         **kwargs)

        self.norm_var = norm_var
        self.istd = tf.Variable(istd,
                                dtype=istd.dtype,
                                trainable=trainable,
                                name='istd')
        self.mean = tf.Variable(mean,
                                dtype=mean.dtype,
                                trainable=trainable,
                                name='mean')

    def call(self, inputs):
        inputs = tf.subtract(inputs, self.mean)
        if self.norm_var:
            inputs = tf.multiply(inputs, self.istd)
        return inputs
