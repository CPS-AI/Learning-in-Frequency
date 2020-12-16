import numpy as np
import math

import tensorflow as tf

from stflayer import STFLayer

class STFNet(tf.keras.Model):
	def __init__(self, 
				batch_size,
				fft_list, 
				kernel_len_list, 
				mode,
				class_num = 10,
				sensor_axis=3,
				c_out=64,
				act_domain='time',
				dropout_rate=0.2,
				reuse=True,
				name='STFNet'):
		super(STFNet, self).__init__(name=name)
		self.batch_size = batch_size
		self.fft_list = fft_list
		self.kernel_len_list = kernel_len_list
		self.mode = mode
		self.class_num = class_num
		self.sensor_axis = 3
		self.c_out = 64
		self.act_domain = 'time'
		self.dropout_rate = dropout_rate
		self.reuse = True

		self.acc_layer1 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.sensor_axis, 
								self.c_out,
								self.act_domain, 
								self.reuse,
								self.mode,
								"acc_layer1"
								)
		self.acc_dropout1 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="acc_dropout1")
		self.acc_layer2 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.c_out, 
								self.c_out,
								self.act_domain, 
								self.reuse,
								self.mode,
								"acc_layer2"
								)
		self.acc_dropout2 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="acc_dropout2")
		self.acc_layer3 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.c_out, 
								int(self.c_out/2),
								self.act_domain, 
								self.reuse,
								self.mode,
								"acc_layer3"
								)
		self.acc_dropout3 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, int(self.c_out/2)],
								name="acc_dropout3")

		self.gyr_layer1 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.sensor_axis, 
								self.c_out,
								self.act_domain, 
								self.reuse,
								self.mode,
								"gyr_layer1"
								)
		self.gyr_dropout1 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="gyr_dropout1")
		self.gyr_layer2 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.c_out, 
								self.c_out,
								self.act_domain, 
								self.reuse,
								self.mode,
								"gyr_layer2"
								)
		self.gyr_dropout2 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="gyr_dropout2")
		self.gyr_layer3 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.c_out, 
								int(self.c_out/2),
								self.act_domain, 
								self.reuse,
								self.mode,
								"gyr_layer3"
								)
		self.gyr_dropout3 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, int(self.c_out/2)],
								name="gyr_dropout3")

		self.fusion_layer1 = STFLayer(self.fft_list, 
								self.kernel_len_list,
								self.c_out, 
								self.c_out,
								self.act_domain, 
								self.reuse,
								self.mode,
								"fusion_layer1"
								)
		self.fusion_dropout1 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="gyr_dropout1")
		self.fusion_layer2 = STFLayer(self.fft_list, 
						self.kernel_len_list,
						self.c_out, 
						self.c_out,
						self.act_domain, 
						self.reuse,
						self.mode,
						"fusion_layer2"
						)
		self.fusion_dropout2 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="gyr_dropout2")
		self.fusion_layer3 = STFLayer(self.fft_list, 
						self.kernel_len_list,
						self.c_out, 
						self.c_out,
						self.act_domain, 
						self.reuse,
						self.mode,
						"fusion_layer3"
						)
		self.fusion_dropout3 = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=[self.batch_size, 1, self.c_out],
								name="gyr_dropout3")

		self.final_dense = tf.keras.layers.Dense(self.class_num, name='final_dense')

	def call(self, inputs, training=True):
		print("inputs: ", inputs)
		acc_inputs, gyr_inputs = tf.split(inputs,
									num_or_size_splits=2,
									axis=2)
		x_acc = self.acc_layer1(acc_inputs)
		if training == True:
			x_acc = self.acc_dropout1(x_acc)
		x_acc = self.acc_layer2(x_acc)
		if training == True:
			x_acc = self.acc_dropout2(x_acc)
		x_acc = self.acc_layer3(x_acc)
		if training == True:
			x_acc = self.acc_dropout3(x_acc)

		x_gyr = self.gyr_layer1(gyr_inputs)
		if training == True:
			x_gyr = self.gyr_dropout1(x_gyr)
		x_gyr = self.gyr_layer2(x_gyr)
		if training == True:
			x_gyr = self.gyr_dropout2(x_gyr)
		x_gyr = self.gyr_layer3(x_gyr)
		if training == True:
			x_gyr = self.gyr_dropout3(x_gyr)

		x_fusion = tf.concat([x_acc, x_gyr], 2)
		x_fusion = self.fusion_layer1(x_fusion)
		if training == True:
			x_fusion = self.fusion_dropout1(x_fusion)
		x_fusion = self.fusion_layer2(x_fusion)
		if training == True:
			x_fusion = self.fusion_dropout2(x_fusion)
		x_fusion = self.fusion_layer3(x_fusion)
		if training == True:
			x_fusion = self.fusion_dropout3(x_fusion)

		x_fusion = tf.reduce_mean(x_fusion, 1)
		outputs = self.final_dense(x_fusion)

		return outputs

