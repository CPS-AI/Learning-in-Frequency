import numpy as np
import math

import tensorflow as tf

from stf_filter_layer import STFFilterLayer
from stf_conv_layer import STFConvLayer

class STFNet(tf.keras.Model):
	def __init__(self, 
				layer_type,
				class_num,
				sensor_number,
				sensor_channel,
				batch_size,
				fft_list=[16, 32, 64, 128], 
				kernel_size=3, 
				c_out=64,
				act_domain='time',
				dropout_rate=0.2,
				name='STFNet'):
		super(STFNet, self).__init__(name=name)

		self.layer_type = layer_type
		self.sensor_number = sensor_number
		self.fft_list = fft_list
		self.kernel_size = kernel_size
		self.act_domain = 'time'
		self.dropout_rate = dropout_rate

		self.sensor_stf_layers = []
		self.sensor_dropout_layers = []

		for s in range(sensor_number):
			self.sensor_stf_layers.append(self.__create_stf_layer(
								sensor_channel, 
								c_out, 
								"sensor{}_stf1".format(s)))
			self.sensor_dropout_layers.append(
				self.__create_dropout_layer([batch_size, 1, c_out], 
								"sensor{}_dropout1".format(s)))
			
			self.sensor_stf_layers.append(self.__create_stf_layer(
								c_out,
								c_out, 
								"sensor{}_stf2".format(s)))
			self.sensor_dropout_layers.append(
				self.__create_dropout_layer([batch_size, 1, c_out], 
								"sensor{}_dropout2".format(s)))

			self.sensor_stf_layers.append(self.__create_stf_layer(
								c_out, 
								c_out//sensor_number, 
								"sensor{}_stf3".format(s)))
			self.sensor_dropout_layers.append(
				self.__create_dropout_layer([batch_size, 1, c_out//sensor_number], 
								"sensor{}_dropout3".format(s)))

		self.fusion_layer1 = self.__create_stf_layer(
								c_out//sensor_number * sensor_number,
								c_out,
								"fusion_stf1")
		self.fusion_dropout1 = self.__create_dropout_layer([batch_size, 1, c_out],
								"fusion_dropout1")
			
		self.fusion_layer2 = self.__create_stf_layer(
								c_out,
								c_out,
								"fusion_stf2")
		self.fusion_dropout2 = self.__create_dropout_layer([batch_size, 1, c_out],
								"fusion_dropout2")

		self.fusion_layer3 = self.__create_stf_layer(
								c_out,
								c_out,
								"fusion_stf3")
		self.fusion_dropout3 = self.__create_dropout_layer([batch_size, 1, c_out],
								"fusion_dropout3")

		self.final_dense = tf.keras.layers.Dense(class_num, 
								name='final_dense')

	def __create_stf_layer(self, c_in, c_out, name):
		if self.layer_type == 'filter':
			layer = STFFilterLayer(self.fft_list, 
								self.kernel_size,
								c_in, 
								c_out,
								self.act_domain, 
								name
							)
		else: 
			layer = STFConvLayer(self.fft_list, 
								self.kernel_size,
								c_in, 
								c_out,
								self.act_domain, 
								name
							)

		return layer

	def __create_dropout_layer(self, noise_shape, name):
		layer = tf.keras.layers.Dropout(self.dropout_rate,
								noise_shape=noise_shape,
								name=name)
		return layer

	def call(self, inputs, training=True):
		sensor_inputs = tf.split(inputs,
							num_or_size_splits=self.sensor_number,
							axis=2)

		for n_sensor in range(self.sensor_number):
			for n_layer in range(3):
				sensor_inputs[n_sensor] = self.sensor_stf_layers[n_layer](sensor_inputs[n_sensor])
				if training:
					sensor_inputs[n_sensor] = self.sensor_dropout_layers[n_layer](sensor_inputs[n_sensor])

		fusion_inputs = tf.concat(sensor_inputs, 2)

		fusion_inputs = self.fusion_layer1(fusion_inputs)
		if training:
			fusion_inputs = self.fusion_dropout1(fusion_inputs)

		fusion_inputs = self.fusion_layer2(fusion_inputs)
		if training:
			fusion_inputs = self.fusion_dropout2(fusion_inputs)

		fusion_inputs = self.fusion_layer3(fusion_inputs)
		if training:
			fusion_inputs = self.fusion_dropout3(fusion_inputs)
		
		fusion_inputs = tf.reduce_mean(fusion_inputs, 1)

		outputs = self.final_dense(fusion_inputs)
		
		return outputs