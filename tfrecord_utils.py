from functools import partial
import tensorflow as tf
import numpy as np

def read_tfrecord(example, wide, sensor_num, sensor_channel, out_dim):
    example_description = {
        'label': tf.io.FixedLenFeature([out_dim], tf.float32),
        'example': tf.io.FixedLenFeature([wide, sensor_num, sensor_channel], tf.float32)
    }
    example = tf.io.parse_single_example(example, example_description)
    return example['example'], example['label']

def load_dataset(tfrec_path, wide, sensor_num, sensor_channel, out_dim):
    dataset = tf.data.TFRecordDataset(tfrec_path)
    dataset = dataset.map(
        partial(read_tfrecord, 
            wide = wide, 
            sensor_num = sensor_num,
            sensor_channel = sensor_channel,
            out_dim=out_dim
            ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return dataset

def get_dataset(tfrec_path, 
    batch_size, 
    wide, 
    sensor_num, 
    sensor_channel, 
    out_dim, 
    shuffle_sample=True):
    dataset = load_dataset(tfrec_path, wide, sensor_num, sensor_channel, out_dim)
    if shuffle_sample:
        dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset