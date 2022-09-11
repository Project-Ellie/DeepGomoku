import os
import tempfile
from pathlib import Path
from typing import List, Iterable, Callable

import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from aegomoku.game_data import read_training_data

feature_description = {
    's': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'p': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'v': tf.io.FixedLenFeature([], tf.float32, default_value=0),
}


def read_tfrecord(example):

    record = tf.io.parse_single_example(example, feature_description)
    s = tf.io.parse_tensor(record['s'], out_type=tf.uint8)
    p = tf.cast(tf.io.parse_tensor(record['p'], out_type=tf.uint8), tf.float32) / 256.
    v = record['v']
    return s, p, v


def load_dataset(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def serialize_example(state, policy, value):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        's': tf.train.Feature(bytes_list=tf.train.BytesList(value=[state])),
        'p': tf.train.Feature(bytes_list=tf.train.BytesList(value=[policy])),
        'v': tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    }

    e = tf.train.Example(features=tf.train.Features(feature=feature))
    return e.SerializeToString()


def to_tfrecords(source, target_dir=None, condition: Callable = None) -> List[str]:
    """
    Creates tfrecord files for each source file.
    If source is a list of file names that don't reside in the same directory, special care needs to be taken
    that the filename stems are unique to avoid collisions in the resulting tfrecord filenames.

    :param condition: a function that takes a record n example and returns a boolean
    :param source: directory or list of file names with pickled game data
    :param target_dir: directory for the created tfrecord files, one for each input file
    :returns a list of the names of the created tfrecord files
    """
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp())

    if os.path.isdir(source):
        filenames = [(Path(source).resolve() / file).as_posix() for file in list(source.rglob("*.pickle"))]
    elif isinstance(source, Iterable):
        filenames = [Path(file).resolve().as_posix() for file in source]
    else:
        raise ValueError(f"Can't interpret {source} as source. Expected directory name or list of filenames")

    outfiles = []
    for filename in tqdm(filenames):

        name = Path(filename).stem
        examples, _ = read_training_data(filename, condition)
        outfile = os.path.join(target_dir, '.'.join([name, 'tfrecords']))
        outfiles.append(outfile)
        with tf.io.TFRecordWriter(outfile) as writer:
            for state, policy, value in examples:
                state = tf.io.serialize_tensor(state.astype(np.uint8)).numpy()
                policy = tf.io.serialize_tensor(np.array(np.array(policy)*255).astype(np.uint8)).numpy()
                example = serialize_example(state, policy, value)
                writer.write(example)
    return outfiles
