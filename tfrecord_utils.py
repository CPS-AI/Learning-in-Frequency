from functools import partial

import tensorflow as tf
import numpy as np

def decode_feature(feature, wide, feature_dim):
    feature = tf.expand_dims(feature, 0)
    feature = tf.reshape(feature, shape=(wide, feature_dim))
    return feature

def read_tfrecord(example, wide, feature_dim, out_dim):
    example_description = {
        'label': tf.io.FixedLenFeature([out_dim], tf.float32),
        'example': tf.io.FixedLenFeature([wide*feature_dim], tf.float32),
    }
    example = tf.io.parse_single_example(example, example_description)
    feature = decode_feature(example['example'], wide, feature_dim)
    label = example['label']
    return feature, label

def load_dataset(tfrec_path, wide, feature_dim, out_dim):
    dataset = tf.data.TFRecordDataset(tfrec_path)
    dataset = dataset.map(
        partial(read_tfrecord, 
            wide = wide, 
            feature_dim = feature_dim,
            out_dim=out_dim
            ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return dataset

def get_dataset(tfrec_path, batch_size, wide, feature_dim, out_dim, shuffle_sample=True):
    dataset = load_dataset(tfrec_path, wide, feature_dim, out_dim)
    if shuffle_sample:
        dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset