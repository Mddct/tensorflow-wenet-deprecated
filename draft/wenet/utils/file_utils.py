import os
from typing import List, Tuple

import tensorflow as tf


def read_symbol_table(symbol_table_file) -> Tuple[List, List]:
    word = []
    ids = []
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            word.append(arr[0])
            ids.append(int(arr[1]))
    return word, ids


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def _is_chief(task_type, task_id, cluster_spec):
    return (task_type is None or task_type == 'chief'
            or (task_type == 'worker' and task_id == 0
                and 'chief' not in cluster_spec.as_dict()))


def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def _write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)


def distributed_write_filepath(filepath, strategy):
    if isinstance(strategy, tf.distribute.MirroredStrategy):
        return filepath
    task_type, task_id, cluster_spec = (
        strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id,
        strategy.cluster_resolver.cluster_spec())

    write_checkpoint_dir = _write_filepath(filepath, task_type, task_id,
                                           cluster_spec)
    return write_checkpoint_dir


def is_chief(strategy):
    if isinstance(strategy, tf.distribute.MirroredStrategy):
        return True
    task_type, task_id, cluster_spec = (
        strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id,
        strategy.cluster_resolver.cluster_spec())
    return _is_chief(task_type, task_id, cluster_spec)
