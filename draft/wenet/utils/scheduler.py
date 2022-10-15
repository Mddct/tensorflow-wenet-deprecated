"""Gradient Accummlate for training TF2 custom training loop.
Copy from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py.
"""

import tensorflow as tf


class GradientAccumulator(object):
    """Gradient accumulation utility.
    When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and
    without synchronization. Users should then call ``.gradients``, scale the
    gradients if required, and pass the result to ``apply_gradients``.
    """

    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_steps = None

    @property
    def step(self):
        """Number of accumulated steps."""
        if self._accum_steps is None:
            self._accum_steps = tf.Variable(
                tf.constant(0, dtype=tf.int64),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )

        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(gradient.value() if gradient is not None else gradient
                    for gradient in self._gradients)

    def __call__(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self._gradients:
            _ = self.step  # Create the step variable.
            self._gradients.extend([
                tf.Variable(
                    tf.zeros_like(gradient),
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                ) if gradient is not None else gradient
                for gradient in gradients
            ])
        if len(gradients) != len(self._gradients):
            raise ValueError("Expected %s gradients, but got %d" %
                             (len(self._gradients), len(gradients)))

        for accum_gradient, gradient in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient, read_value=False)

        self._accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient), read_value=False)


# class NoamLR(tf.keras.optimizers.schedules.LearningRateSchedule):
#     """lr *= model_size ** -0.5
#              * min(step ** -0.5, step * warmup_step ** -1.5)
#     """

#     def __init__(self, d_model, warmup_steps=4000.0, max_lr=None):
#         super().__init__()

#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#         self.warmup_steps = float(warmup_steps)
#         self.max_lr = max_lr

#     def __call__(self, step):
#         step = tf.cast(step, dtype=tf.float32)
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps**-1.5)

#         lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#         if self.max_lr is not None:
#             return tf.math.minimum(self.max_lr, lr)
#         return lr

#     def get_config(self):
#         config = {
#             'd_model': self.d_model,
#             'warmup_steps': self.warmup_steps,
#         }

#         return config

# class WarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):
#     """lr *= warmup_step ** 0.5
#              * min(step ** -0.5, step * warmup_step ** -1.5)
#     """

#     def __init__(self,
#                  initial_learning_rate,
#                  warmup_steps=2500.0,
#                  max_lr=None):
#         super().__init__()

#         self.warmup_steps = float(warmup_steps)
#         self.warmup_steps_tensor = tf.cast(warmup_steps, tf.float32)
#         self.initial_learning_rate = initial_learning_rate
#         self.max_lr = max_lr

#     def __call__(self, step):
#         step = tf.cast(step, dtype=tf.float32)
#         learning_rate = self.initial_learning_rate
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps**-1.5)

#         lr = learning_rate * tf.math.rsqrt(
#             self.warmup_steps_tensor) * tf.math.minimum(arg1, arg2)

#         return lr

#     def get_config(self):
#         config = {
#             'warmup_steps': self.warmup_steps,
#         }
#         config = {}

#         return config


class TransformerLearningRateSchedule(
        tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, initial_learning_rate, hidden_size, warmup_steps):
        """Initialize configuration of the learning rate schedule.

        Args:
          initial_learning_rate: A float, the initial learning rate.
          hidden_size: An integer, the model dimension in the hidden layers.
          warmup_steps: An integer, the number of steps required for linear warmup.
        """
        super(TransformerLearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.hidden_size = hidden_size
        self.warmup_steps = warmup_steps
        self.warmup_steps_tensor = tf.cast(warmup_steps, tf.float32)

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.

        Args:
        global_step: An integer, the current global step used for learning rate
          calculation.

        Returns:
          A float, the learning rate needs to be used for current global step.
        """
        with tf.name_scope('transformer_learning_rate_schedule'):
            global_step = tf.cast(global_step, tf.float32)
            learning_rate = self.initial_learning_rate
            learning_rate *= (self.hidden_size**-0.5)
            # Apply linear warmup
            learning_rate *= tf.minimum(1.0,
                                        global_step / self.warmup_steps_tensor)
            # Apply rsqrt decay
            learning_rate /= tf.sqrt(
                tf.maximum(global_step, self.warmup_steps_tensor))
            return learning_rate

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'hidden_size': self.hidden_size,
            'warmup_steps': self.warmup_steps,
        }


# learning_rate = NoamLR(d_model)

# optimizer = tf.keras.optimizers.Adam(learning_rate,
#                                      beta_1=0.9,
#                                      beta_2=0.98,
#                                      epsilon=1e-9)
