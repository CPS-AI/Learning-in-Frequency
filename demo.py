import os
import numpy as np
import tensorflow as tf

from tfrecord_utils import get_dataset
from stfnet import STFNet

LAYER_TYPE = "conv"
BATCH_SIZE = 32
SERIES_SIZE = 512
SENSOR_CHANNEL = 3
SENSOR_NUM = 2
CLASS_NUM = 10

EPOCH_NUM = 10000000
SAVE_EPOCH_NUM = 500

TRAIN_TFRECORD = os.path.join("tfrecords", "speech", "train.tfrecord")
TEST_TFRECORD = os.path.join("tfrecords", "speech", "train.tfrecord")

# TRAIN_TFRECORD = os.path.join("tfrecords", "hhar", "train.tfrecord")
# TEST_TFRECORD = os.path.join("tfrecords", "hhar", "eval.tfrecord")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    dataset_train = get_dataset(
        TRAIN_TFRECORD, 
        BATCH_SIZE, 
        SERIES_SIZE, 
        SENSOR_NUM, 
        SENSOR_CHANNEL,
        CLASS_NUM
    )

    dataset_eval = get_dataset(
        TEST_TFRECORD,
        BATCH_SIZE,
        SERIES_SIZE,
        SENSOR_NUM,
        SENSOR_CHANNEL,
        CLASS_NUM,
        shuffle_sample=False,
    )

    model = STFNet(LAYER_TYPE, CLASS_NUM, SENSOR_NUM, SENSOR_CHANNEL, BATCH_SIZE)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.9, beta_2=0.99),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    checkpoint_path = "demo/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=SAVE_EPOCH_NUM * BATCH_SIZE,
    )

    model.fit(
        dataset_train,
        # steps_per_epoch=10,
        epochs=EPOCH_NUM,
        validation_data=dataset_eval,
        callbacks=[cp_callback]
    )    

if __name__ == "__main__":
    main()
