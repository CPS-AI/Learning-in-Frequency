import os
import numpy as np 
import tensorflow as tf 

from tfrecord_utils import get_dataset
from stfnet import STFNet

BATCH_SIZE = 32
FFT_LIST = [16, 32, 64, 128]
KERNEL_LEN_LIST = [3, 3, 3, 3]
MODE = 'filter'
SERIES_SIZE = 512
SENSOR_AXIS = 3
SENSOR_NUM = 2
CLASS_NUM = 10

EPOCH_NUM = 100000

TRAIN_TFRECORD = os.path.join("tfrecords", "train_0-17siebel_all_speaker.tfrecord")
TEST_TFRECORD = os.path.join("tfrecords", "eval_0-17siebel_all_speaker.tfrecord")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
	dataset_train = get_dataset(
		TRAIN_TFRECORD, 
		BATCH_SIZE, 
		SERIES_SIZE, 
		SENSOR_AXIS*SENSOR_NUM, 
		CLASS_NUM)
	dataset_eval = get_dataset(
		TEST_TFRECORD, 
		BATCH_SIZE, 
		SERIES_SIZE, 
		SENSOR_AXIS*SENSOR_NUM, 
		CLASS_NUM, 
		shuffle_sample=False)

	batch_feature_train, batch_label_train = next(iter(dataset_train))
	batch_feature_train = tf.reshape(batch_feature_train, 
		[BATCH_SIZE, SERIES_SIZE, SENSOR_NUM*SENSOR_AXIS])

	batch_feature_eval, batch_label_eval = next(iter(dataset_eval))
	batch_feature_eval = tf.reshape(batch_feature_eval, 
		[BATCH_SIZE, SERIES_SIZE, SENSOR_NUM*SENSOR_AXIS])

	print("batch_feature_train.shape = {}, batch_label_train.shape = {}".format(
		batch_feature_train.get_shape().as_list(),
		batch_label_train.get_shape().as_list()))

	model = STFNet(BATCH_SIZE, FFT_LIST, KERNEL_LEN_LIST, MODE)

	model.compile(
				optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
												beta_1=0.9,
												beta_2=0.99), 
				loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
				metrics=[tf.keras.metrics.CategoricalAccuracy()])
	
	model.fit(batch_feature_train, 
			batch_label_train, 
			epochs=EPOCH_NUM,
			validation_data=[batch_feature_eval, 
							batch_label_eval])

	# print(model.predict(batch_feature_train))

if __name__ == '__main__':
	main()