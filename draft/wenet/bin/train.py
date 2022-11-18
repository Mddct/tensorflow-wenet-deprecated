import os

from tensorflow.python.eager.context import num_gpus

from wenet.utils.distribute_utils import get_distribution_strategy

import orbit
import tensorflow as tf
import yaml
from absl import app, flags
from wenet.dataset.dataset import Dataset
from wenet.utils.executor import AsrTrainer
from wenet.utils.file_utils import distributed_write_filepath, is_chief
from wenet.utils.init_model import init_model
from wenet.utils.scheduler import TransformerLearningRateSchedule
from wenet.utils.file_utils import read_symbol_table

FLAGS = flags.FLAGS

flags.DEFINE_string('config', default=None, required=True, help='config file')
flags.DEFINE_enum('data_type',
                  default='raw',
                  enum_values=['raw', 'shard'],
                  help='train and cv data type')
flags.DEFINE_string('train_data',
                    default=None,
                    required=True,
                    help='train data file')
flags.DEFINE_string('cv_data',
                    default=None,
                    required=False,
                    help='cv data file')

flags.DEFINE_string('model_dir',
                    default=None,
                    required=True,
                    help='save model dir')
flags.DEFINE_string('checkpoint', default=None, help='checkpoint model')

flags.DEFINE_string('tensorboard_dir',
                    default='tensorboard',
                    help='tensorboard log dir')
flags.DEFINE_string('cmvn', default=None, help='global cmvn file')
flags.DEFINE_string('symbol_table',
                    default=None,
                    required=True,
                    help='model unit symbol table for training')
flags.DEFINE_integer('prefetch', default=100, help='prefetch number')
flags.DEFINE_integer('max_to_keep', default=100, help='max to keep checkpoint')
flags.DEFINE_integer('checkpoint_interval',
                     default=100,
                     help='the minimum step interval between two checkpoints.')
flags.DEFINE_enum(
    'dist_strategy',
    default='mirrored',
    enum_values=[
        "one_device",
        "mirrored",
        "parameter_server",
        "multi_worker_mirrored",
    ],
    help="""MultiWorkerMirroredStrategy: mutli machine multi gpus\n
    MirrordStrategy: one machine multi gpus""")
flags.DEFINE_string('trained_model_path',
                    default="",
                    help="partial init model, often from pretrain model")


def main(argv):
    # Set random seed
    tf.random.set_seed(777)
    with open(FLAGS.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    num_gpus = 0
    if os.environ['CUDA_VISIBLE_DEVICES'] is not None:
        num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    strategy = get_distribution_strategy(
        FLAGS.dist_strategy,
        num_gpus=num_gpus,
    )
    dataset_conf = configs['dataset_conf']

    # TOO: global_batch_size from TF_CONFIG
    global_batch_size = dataset_conf['batch_conf'][
        'batch_size'] * strategy.num_replicas_in_sync

    words, ids = read_symbol_table(symbol_table_path)
    train_dataset, vocab_size = Dataset(
        dataset_conf,
        FLAGS.symbol_table,
        FLAGS.train_data,
        global_batch_size,
        FLAGS.prefetch,
        FLAGS.data_type,
        strategy,
    )
    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        raise NotImplementedError('only fbank support now')

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = FLAGS.cmvn
    configs['is_json_cmvn'] = True

    ctc_weight = configs['model_conf']['ctc_weight']
    # Init asr model from configs
    with strategy.scope():
        # model
        model = init_model(configs)

        # scheduler
        learning_rate = TransformerLearningRateSchedule(
            configs['optim_conf']['lr'],
            # TODO
            256.0,
            configs['scheduler_conf']['warmup_steps'],
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
            clipvalue=configs['grad_clip'],
        )
        # metrics
        # TODO: wer metrics
        metrics = {'loss': tf.keras.metrics.Sum('loss', dtype=tf.float32)}
        if ctc_weight != 0.0:
            metrics['loss_ctc'] = tf.keras.metrics.Sum('loss_ctc',
                                                       dtype=tf.float32)
        if ctc_weight != 1.0:
            metrics['loss_att'] = tf.keras.metrics.Sum('loss_att',
                                                       dtype=tf.float32)
        # checkpoint
        checkpoint_dir = distributed_write_filepath(FLAGS.model_dir, strategy)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ## init from partial other model
        init_fn = None
        if FLAGS.trained_model_path != "":
            # TODO assume model_dir should be empty
            init_fn = lambda: checkpoint.restore(FLAGS.trained_model_path
                                                 ).expect_partial()
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            checkpoint_dir,
            max_to_keep=FLAGS.max_to_keep if is_chief(strategy) else 0,
            checkpoint_interval=FLAGS.checkpoint_interval
            if is_chief(strategy) else None,
            step_counter=checkpoint.optimizer.iterations,
            init_fn=init_fn,  # future to init partial weight
        )
        trainer = AsrTrainer(
            train_dataset,
            model,
            optimizer,
            global_batch_size=global_batch_size,
            strategy=strategy,
            metrics=metrics,
        )
        controller = orbit.Controller(
            trainer=trainer,
            steps_per_loop=10,
            global_step=trainer.optimizer.iterations,
            checkpoint_manager=checkpoint_manager,
            summary_interval=configs['log_interval'],
            summary_dir=os.path.join(checkpoint_dir, "tensorboard"),
        )
        controller.train(configs['max_steps'])

    # # TODO: train and continous evaluate
    # train(model,
    #       train_dataset,
    #       optimizer,
    #       strategy=strategy,
    #       checkpoint_dir=distributed_write_filepath(FLAGS.model_dir, strategy),
    #       checkpoint_path_or_latest=FLAGS.checkpoint,
    #       global_batch_size=global_batch_size,
    #       max_to_keep=FLAGS.max_to_keep,
    #       log_interval=configs['log_interval'])


if __name__ == '__main__':
    app.run(main)
