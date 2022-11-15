from typing import Optional
import orbit
import tensorflow as tf


class AsrTrainer(orbit.StandardTrainer):

    def __init__(
        self,
        train_dataset,
        model,
        optimizer,
        global_batch_size,
        strategy: Optional[tf.distribute.Strategy] = None,
        metrics=None,
        trainer_options=None,
    ) -> None:

        self.strategy = strategy if strategy is not None else tf.distribute.get_strategy(
        )
        with self.strategy.scope():
            self.optimizer = optimizer
            self.model = model
            self.global_step = self.optimizer.iterations

        self.global_batch_size = global_batch_size

        if metrics is None:
            self.metrics = {}
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            self.metrics = {'loss': metrics}

        super(AsrTrainer, self).__init__(
            train_dataset=train_dataset,
            options=trainer_options,
        )

    def train_loop_begin(self):
        for _, metric in self.metrics.items():
            metric.reset_states()

    def train_step(self, iterator):

        def train_fn(inputs):

            with tf.GradientTape() as tape:
                # feats, feats_length, labels, labels_length = inputs
                # labels = tf.cast(labels, dtype=tf.int32)
                # labels_length = tf.cast(labels_length, dtype=tf.int32)
                _, _, labels, labels_length = inputs
                encoder_out, encoder_out_lens, decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad = self.model(
                    inputs)
                loss_dict = self.model.compute_loss(
                    encoder_out,
                    encoder_out_lens,
                    labels,
                    labels_length,
                    decoder_out,
                    ys_out_pad,
                    r_decoder_out,
                    r_ys_out_pad,
                )
                loss = tf.reduce_sum(
                    loss_dict['loss']) / self.global_batch_size
                # regularization loss
                regularization_loss = tf.add_n(self.model.losses)
                loss = loss + tf.nn.scale_regularization_loss(
                    regularization_loss)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    list(zip(gradients, self.model.trainable_variables)))

                for name in self.metrics.keys():
                    self.metrics[name].update_state(
                        tf.reduce_sum(loss_dict[name]) /
                        self.global_batch_size)

        self.strategy.run(train_fn, args=(next(iterator), ))

    def train_loop_end(self):

        with self.strategy.scope():
            # Export the metrics.
            metrics = {
                name: metric.result() / self.optimizer.iterations.numpy()
                for name, metric in self.metrics.items()
            }
            if isinstance(self.optimizer.lr,
                          tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = self.optimizer.lr(self.optimizer.iterations)
            else:
                current_lr = self.optimizer.lr
            metrics['learnint_rate'] = current_lr

        return metrics


# def train(
#         model,
#         dataset,
#         optimizer,
#         strategy: tf.distribute.Strategy,
#         checkpoint_dir: str,
#         checkpoint_path_or_latest: Optional[str] = None,
#         global_batch_size: int = 1,
#         num_steps: int = tf.int64.max,
#         max_to_keep: int = 1,
#         # checkpoint_interval: int = 100,
#         log_interval: int = 100):
#     """
#     Args:
#         checkpoint_path: str: latest or {ckpt_name}-step eg: model-1
#     """

#     steps = tf.Variable(0, dtype=tf.int64)
#     checkpoint = tf.train.Checkpoint(model=model, steps=steps)

#     checkpoint_manager = tf.train.CheckpointManager(
#         checkpoint,
#         directory=checkpoint_dir,
#         max_to_keep=max_to_keep if not is_chief(strategy) else 1,
#         # checkpoint_interval=checkpoint_interval,
#     )
#     if checkpoint_path_or_latest is not None:
#         if checkpoint_path_or_latest == 'latest':
#             checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#         else:
#             checkpoint.restore(checkpoint_path_or_latest).expect_partial()
#         steps.assign_add(checkpoint.steps)

#     train_summary_writer = tf.summary.create_file_writer(
#         os.path.join(checkpoint_dir, "tensorboard"))

#     log(f"train | step: {steps.numpy(): 6d} | training until step {num_steps}..."
#         )

#     stepTimer = StepTimer(steps)
#     iterator = iter(dataset)

#     try:
#         with tf.experimental.async_scope():
#             while True:

#                 loss_dict = train_step(
#                     iterator,
#                     model,
#                     optimizer,
#                     global_batch_size,
#                     strategy,
#                 )

#                 with train_summary_writer.as_default():
#                     steps_scalar = steps.numpy()
#                     for name, value_tensor in loss_dict.items():
#                         if value_tensor is not None:
#                             tf.summary.scalar(name,
#                                               value_tensor.numpy(),
#                                               step=steps_scalar)

#                 if steps_scalar % log_interval == 0:
#                     train_output_str = format_output({
#                         name: value_tensor.numpy()
#                         for name, value_tensor in loss_dict.items()
#                         if value_tensor is not None
#                     })
#                     steps_per_second = stepTimer.steps_per_second(steps)

#                     # lr = optimizer.get_config()['learning_rate']
#                     lr = optimizer.lr(optimizer.iterations).numpy()
#                     log(f"train | step: {steps_scalar: 6d} | "
#                         f"steps/sec: {steps_per_second: 6.1f} | "
#                         f"output: {train_output_str} | "
#                         f"learnint_rate: {lr}")

#                 steps.assign_add(1)
#                 # checkpoint_manager.save(checkpoint_number=steps)
#     except (StopIteration, tf.errors.OutOfRangeError):
#         log(f"The dataset iterator is exhausted after {steps.nump*()} steps.")
#         if not is_chief(strategy):
#             tf.io.gfile.rmtree(checkpoint_dir)

# #     # clstrategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])

# # # def step_fn():
# # #     return {'loss': tf.constant([1]), 'ctc': tf.constant([2]), 'att': tf.constant([3])}

# # # per_replica_result = strategy.run(step_fn)
# # # print(per_replica_result)

# # # {'loss': PerReplica:{
# # #   0: <tf.Tensor: shape=(1,), dtype=int32, numpy=array([1], dtype=int32)>,
# # #   1: <tf.Tensor: shape=(1,), dtype=int32, numpy=array([1], dtype=int32)>
# # # }, 'ctc': PerReplica:{
# # #   0: <tf.Tensor: shape=(1,), dtype=int32, numpy=array([2], dtype=int32)>,
# # #   1: <tf.Tensor: shape=(1,), dtype=int32, numpy=array([2], dtype=int32)>
# # # }, 'att': PerReplica:{
# # #   0: <tf.Tensor: shape=(1,), dtype=int32, numpy=array([3], dtype=int32)>,
# # #   1: <tf.Tensor: shape=(1,), dtype=int32, numpy=array([3], dtype=int32)>
# # # }}
