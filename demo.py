import os
import numpy as np 
import tensorflow as tf 

from tfrecord_utils import get_dataset
from stfnet_new import STFNet

# model = STFNet(LAYER_TYPE, CLASS_NUM, SENSOR_NUM, SENSOR_CHANNEL, BATCH_SIZE)

LAYER_TYPE = 'filter'
BATCH_SIZE = 32
SERIES_SIZE = 512
SENSOR_CHANNEL = 3
SENSOR_NUM = 2
CLASS_NUM = 10

EPOCH_NUM = 10000000
SAVE_EPOCH_NUM = 500

# TRAIN_TFRECORD = os.path.join("tfrecords", "train_0-17siebel_all_speaker.tfrecord")
# TEST_TFRECORD = os.path.join("tfrecords", "eval_0-17siebel_all_speaker.tfrecord")

TRAIN_TFRECORD = os.path.join("tfrecords", "New10Digits", "train.tfrecord")
TEST_TFRECORD = os.path.join("tfrecords", "New10Digits", "eval.tfrecord")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
	dataset_train = get_dataset(
		TRAIN_TFRECORD, 
		BATCH_SIZE, 
		SERIES_SIZE, 
		SENSOR_CHANNEL*SENSOR_NUM, 
		CLASS_NUM)
	dataset_eval = get_dataset(
		TEST_TFRECORD, 
		BATCH_SIZE, 
		SERIES_SIZE, 
		SENSOR_CHANNEL*SENSOR_NUM, 
		CLASS_NUM, 
		shuffle_sample=False)

	batch_feature_train, batch_label_train = next(iter(dataset_train))
	batch_feature_train = tf.reshape(batch_feature_train, 
		[BATCH_SIZE, SERIES_SIZE, SENSOR_NUM*SENSOR_CHANNEL])

	batch_feature_eval, batch_label_eval = next(iter(dataset_eval))
	batch_feature_eval = tf.reshape(batch_feature_eval, 
		[BATCH_SIZE, SERIES_SIZE, SENSOR_NUM*SENSOR_CHANNEL])

	print("batch_feature_train.shape = {}, batch_label_train.shape = {}".format(
		batch_feature_train.get_shape().as_list(),
		batch_label_train.get_shape().as_list()))

	model = STFNet(LAYER_TYPE, CLASS_NUM, SENSOR_NUM, SENSOR_CHANNEL, BATCH_SIZE)

	loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


	model.compile(
				optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2,
												beta_1=0.9,
												beta_2=0.99), 
				loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
				metrics=[tf.keras.metrics.CategoricalAccuracy()])
	
	checkpoint_path = "demo/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
											save_weights_only=True,
											verbose=1,
											save_freq=SAVE_EPOCH_NUM*BATCH_SIZE)

	model.fit(batch_feature_train, 
			batch_label_train, 
			epochs=EPOCH_NUM,
			validation_data=[batch_feature_eval, 
							batch_label_eval],
			callbacks=[cp_callback])


if __name__ == '__main__':
	main()