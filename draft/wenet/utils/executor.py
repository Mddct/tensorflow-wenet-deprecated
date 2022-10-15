import orbit
import tensorflow as tf


class AsrTrainer(orbit.StandardTrainer):

    def __init__(self,
                 train_dataset,
                 model,
                 optimizer,
                 global_batch_size,
                 strategy=tf.distribute.MirroredStrategy,
                 metrics=None,
                 traininer_options=None) -> None:

        self.strategy = strategy
        with self.strategy.scope():
            # NOTE: loss_fn == None, model.call will return loss
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
            options=traininer_options,
        )

    def train_loop_begin(self):
        for _, metric in self.metrics.items():
            metric.reset_states()

    def train_step(self, iterator):

        def train_fn(inputs):
            with tf.GradientTape() as tape:
                feats, feats_length, labels, labels_length = inputs
                loss_dict = self.model(
                    feats,
                    feats_length,
                    labels,
                    labels_length,
                    training=True,
                )
                loss_dict = self.model(feats, feats_length, labels,
                                       labels_length)
                gradients = tape.gradient(
                    tf.reduce_sum(loss_dict['loss']) / self.global_batch_size,
                    self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    list(zip(gradients, self.model.trainable_variables)))
                # Update metrics
                for name in self.metrics.keys():
                    self.metrics[name].update_state(
                        tf.reduce_sum(
                            tf.reduce_sum(loss_dict[name]) /
                            self.global_batch_size))

        self.strategy.run(train_fn, args=(next(iterator), ))

    def train_loop_end(self):
        with self.strategy.scope():
            # Export the metrics.
            metrics = {
                name: metric.result()
                for name, metric in self.metrics.items()
            }
            if isinstance(self.optimizer.lr,
                          tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = self.optimizer.lr(self.optimizer.iterations)
            else:
                current_lr = self.optimizer.lr
            metrics['learnint_rate'] = current_lr

        return metrics


# @tf.function(experimental_relax_shapes=True)
# def train_step(iterator, dist_model, optimizer, global_batch_size, strategy):
#     """Training step function."""

#     def step_fn(inputs):
#         """Per-Replica step function."""
#         feats, feats_length, labels, labels_length = inputs
#         with tf.GradientTape() as tape:
#             loss_dict = dist_model(feats, feats_length, labels, labels_length)
#             train_loss = tf.reduce_sum(loss_dict['loss']) / global_batch_size
#             grads = tape.gradient(train_loss, dist_model.trainable_variables)
#             optimizer.apply_gradients(
#                 zip(grads, dist_model.trainable_variables))

#             loss_ctc = None
#             if loss_dict['loss_ctc'] is not None:
#                 loss_ctc = tf.reduce_sum(
#                     loss_dict['loss_ctc']) / global_batch_size
#             loss_att = None
#             if loss_dict['loss_att'] is not None:
#                 loss_att = tf.reduce_sum(
#                     loss_dict['loss_att']) / global_batch_size

#         return {'loss': train_loss, 'loss_ctc': loss_ctc, 'loss_att': loss_att}

#     per_replica_losses_dict = strategy.run(step_fn, args=(next(iterator), ))

#     return {
#         'loss':
#         strategy.reduce(tf.distribute.ReduceOp.SUM,
#                         per_replica_losses_dict['loss'],
#                         axis=None),
#         'loss_ctc':
#         strategy.reduce(tf.distribute.ReduceOp.SUM,
#                         per_replica_losses_dict['loss_ctc'],
#                         axis=None)
#         if per_replica_losses_dict['loss_ctc'] is not None else None,
#         'loss_att':
#         strategy.reduce(tf.distribute.ReduceOp.SUM,
#                         per_replica_losses_dict['loss_att'],
#                         axis=None)
#         if per_replica_losses_dict['loss_att'] is not None else None,
#     }

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
