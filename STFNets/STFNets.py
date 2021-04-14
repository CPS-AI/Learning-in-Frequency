import tensorflow as tf 
import numpy as np 
from sklearn.metrics import f1_score
import os
import sys

import math
import plot

from tfrecord_stft_util import input_pipeline_har

os.environ["CUDA_VISIBLE_DEVICES"]="0"

layers = tf.contrib.layers 

BATCH_SIZE = 64

GEN_FFT_N = [16, 32, 64, 128]
GEN_FFT_STEP = [FFT_N_ELEM for FFT_N_ELEM in GEN_FFT_N]

FILTER_LEN = [3, 3, 3, 3]
DILATION_LEN = [1, 2, 4, 8]

GEN_FFT_N2 = [12, 24, 48, 96]
SERIES_SIZE2 = 384
GEN_FFT_STEP2 = [FFT_N_ELEM for FFT_N_ELEM in GEN_FFT_N2]

GEN_C_OUT = 64 #72
KEEP_PROB = 0.5


select = 'motion' # {'hhar', 'wifi'}
if len(sys.argv) > 1:
	select = sys.argv[1]
# if select != 'wifi' and select != 'hhar':
# 	print('select wifi or hhar')
# 	sys.exit("select wifi or hhar")
print('select', select)

if select == 'wifi':
	SERIES_SIZE = 512
	SENSOR_AXIS = 30
	SENSOR_NUM = 2
	OUT_DIM = 6
if select == 'hhar':
	SERIES_SIZE = 512
	SENSOR_AXIS = 3
	SENSOR_NUM = 2
	OUT_DIM = 6
if select == 'motion':
	SERIES_SIZE = 512
	SENSOR_AXIS = 3
	SENSOR_NUM = 2
	OUT_DIM = 10


FILTER_EXP_SEL = 'linear_interp' #{linear_interp', 'time_zeropadding'}

FILTER_INIT = 'real' #{'real', 'complex'}

GLOBAL_KERNEL_SIZE = 32

CONV_KERNEL_INIT = 'freq' #{'time', 'freq'}

MERGE_INIT = 'zero'

ADAM_LR = 1e-4
ADAM_B1 = 0.9
ADAM_B2 = 0.99

if select == 'wifi':
	ACT_DOMIAN = 'freq'
	FILTER_FLAG = False
	FREQ_CONV_FLAG = True
if select == 'hhar':
	ACT_DOMIAN = 'time'
	FILTER_FLAG = True
	FREQ_CONV_FLAG = False
	ADAM_B1 = 0.5
	ADAM_B2 = 0.9
if select == 'motion':
	ACT_DOMIAN = 'time'
	FILTER_FLAG = True
	FREQ_CONV_FLAG = False
	ADAM_B1 = 0.9
	ADAM_B2 = 0.99

DROP_FLAG = True
INPUT_COMPLEX_NORM_FLAG = True
CLIP_FLAG = False


metaDict = {'hhar':[13544, 1765],
			'wifi':[11100, 900],
			'motion':[1800, 200]}
TRAIN_SIZE = metaDict[select.split('_')[-1]][0]
EVAL_DATA_SIZE = metaDict[select.split('_')[-1]][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))

TOTAL_ITER_NUM = 10000000
CLIP_VAL = 0.3

if CLIP_FLAG:
	print('CLIP_VAL', CLIP_VAL)

def complex_glorot_uniform(c_in, c_out_total, fft_list, fft_n, use_bias=True, name='complex_mat'):
	with tf.variable_scope(name):
		c_out = int(c_out_total)//len(fft_list)

		if FILTER_INIT == 'real':
			kernel = tf.get_variable('kernel', shape = [1, 1, c_in*c_out, fft_n],
							initializer=tf.contrib.layers.xavier_initializer())
			kernel_complex_org = tf.fft(tf.complex(kernel, 0.*kernel))
			kernel_complex_org = tf.transpose(kernel_complex_org, [0, 1, 3, 2])
			kernel_complex_org = kernel_complex_org[:,:,:fft_n//2+1,:]
		elif FILTER_INIT == 'complex':
			kernel_r = tf.get_variable('kernel_real', shape = [1, 1, fft_n/2+1, c_in*c_out],
								initializer=tf.contrib.layers.xavier_initializer())
			kernel_i = tf.get_variable('kernel_imag', shape = [1, 1, fft_n/2+1, c_in*c_out],
								initializer=tf.contrib.layers.xavier_initializer())
			kernel_complex_org = tf.complex(kernel_r, kernel_i)

		kernel_complex_dict = {}
		for fft_elem in fft_list:
			if fft_elem < fft_n:
				kernel_complex_r = tf.image.resize_bilinear(tf.real(kernel_complex_org), 
						[1, int(fft_elem/2)+1], align_corners=True)
				kernel_complex_i = tf.image.resize_bilinear(tf.imag(kernel_complex_org), 
						[1, int(fft_elem/2)+1], align_corners=True)
				kernel_complex_dict[fft_elem] = tf.reshape(tf.complex(kernel_complex_r, kernel_complex_i),
						[1, 1, int(fft_elem/2)+1, c_in, c_out])
			elif fft_elem == fft_n:
				kernel_complex_dict[fft_elem] = tf.reshape(kernel_complex_org, 
											[1, 1, int(fft_elem/2)+1, c_in, c_out])
			else:
				if FILTER_EXP_SEL == 'time_zeropadding':
					zero_pad = tf.zeros([1, 1, c_in*c_out, fft_elem-fft_n])
					kernel_zPad = tf.concat([kernel, zero_pad], 3)
					kernel_zPad_complex = tf.fft(tf.complex(kernel_zPad, 0.*kernel_zPad))
					kernel_zPad_complex = tf.transpose(kernel_zPad_complex, [0, 1, 3, 2])
					kernel_zPad_complex = kernel_zPad_complex[:,:,:int(fft_elem)/2+1,:]
					kernel_complex_dict[fft_elem] = tf.reshape(kernel_zPad_complex, 
												[1, 1, int(fft_elem/2)+1, c_in, c_out])
				elif FILTER_EXP_SEL == 'linear_interp':
					kernel_complex_r = tf.image.resize_bilinear(tf.real(kernel_complex_org), 
							[1, int(fft_elem/2)+1], align_corners=True)
					kernel_complex_i = tf.image.resize_bilinear(tf.imag(kernel_complex_org), 
							[1, int(fft_elem/2)+1], align_corners=True)
					kernel_complex_dict[fft_elem] = tf.reshape(tf.complex(kernel_complex_r, kernel_complex_i),
							[1, 1, int(fft_elem/2)+1, c_in, c_out])

		if use_bias:
			bias_complex_r = tf.get_variable('bias_real', shape=[c_out*len(fft_list)], 
										initializer=tf.zeros_initializer())
			bias_complex_i = tf.get_variable('bias_imag', shape=[c_out*len(fft_list)], 
										initializer=tf.zeros_initializer())
			bias_complex = tf.complex(bias_complex_r, bias_complex_i, name='bias')
			return kernel_complex_dict, bias_complex
		else:
			return kernel_complex_dict

def spectral_filter_gen(c_in, c_out_total, basic_len, len_list, use_bias, name='spectral_filter'):
	with tf.variable_scope(name):
		c_out = int(c_out_total)/len(len_list)
		if CONV_KERNEL_INIT == 'freq':
			kernel_r = tf.get_variable('kernel_real', shape = [1, basic_len, c_in, c_out],
										initializer=tf.contrib.layers.xavier_initializer())
			kernel_i = tf.get_variable('kernel_imag', shape = [1, basic_len, c_in, c_out],
										initializer=tf.contrib.layers.xavier_initializer())
		elif CONV_KERNEL_INIT == 'time':
			kernel = tf.get_variable('kernel', shape = [1, basic_len, c_out, 2*(c_in+1)],
										initializer=tf.contrib.layers.xavier_initializer())
			kernel_c = tf.fft(tf.complex(kernel, 0.*kernel))
			kernel_c = kernel_c[:, :, :, 1:(c_in+1)]
			kernel_c = tf.transpose(kernel_c, [0, 1, 3, 2])
			kernel_r = tf.real(kernel_c)
			kernel_i = tf.imag(kernel_c)
		kernel_dict = {}
		for filter_len in len_list:
			if filter_len == basic_len:
				kernel_dict[filter_len] = [kernel_r, kernel_i]
			else:
				kernel_exp_r = tf.image.resize_bilinear(kernel_r, 
								[filter_len, c_in], align_corners=True)
				kernel_exp_i = tf.image.resize_bilinear(kernel_i, 
								[filter_len, c_in], align_corners=True)
				kernel_dict[filter_len] = [kernel_exp_r, kernel_exp_i]
		if use_bias:
			bias_complex_r = tf.get_variable('bias_real', shape=[c_out], 
										initializer=tf.zeros_initializer())
			bias_complex_i = tf.get_variable('bias_imag', shape=[c_out], 
										initializer=tf.zeros_initializer())
			bias_complex = tf.complex(bias_complex_r, bias_complex_i, name='bias')
			return kernel_dict, bias_complex
		else:
			return kernel_dict

def complex_layerNorm(in_r, in_i, reuse=False, name='complex_layerNorm'):
	with tf.variable_scope(name, reuse=reuse):
		assert in_r.get_shape().as_list()[-1] == in_i.get_shape().as_list()[-1]
		assert len(in_r.get_shape().as_list()) == 4

		epsilon = 1e-4

		c_size = in_r.get_shape().as_list()[-1]
		r_mean = tf.reduce_mean(in_r, [1,2,3], keep_dims = True)
		i_mean = tf.reduce_mean(in_i, [1,2,3], keep_dims = True)
		r_center = in_r - r_mean
		i_center = in_i - i_mean
		conv_rr = tf.reduce_mean(r_center*r_center, [1,2,3], keep_dims = True) + epsilon
		conv_ii = tf.reduce_mean(i_center*i_center, [1,2,3], keep_dims = True) + epsilon
		conv_ri = tf.reduce_mean(r_center*i_center, [1,2,3], keep_dims = True) + epsilon

		tau = conv_rr + conv_ii
		delta = conv_rr*conv_ii - conv_ri*conv_ri
		s = tf.sqrt(delta)
		t = tf.sqrt(tau + 2*s)
		inverse_st = 1.0 / (s * t)
		Wrr = (conv_ii + s) * inverse_st
		Wii = (conv_rr + s) * inverse_st
		Wri = -conv_ri * inverse_st

		r_norm = Wrr * r_center + Wri * i_center
		i_norm = Wri * r_center + Wii * i_center

		beta_r = tf.get_variable('beta_real', shape=[1, 1, 1, c_size], 
										initializer=tf.zeros_initializer())
		beta_i = tf.get_variable('beta_imag', shape=[1, 1, 1, c_size], 
										initializer=tf.zeros_initializer())		
		gamma_rr = tf.get_variable('gamma_rr', shape=[1, 1, 1, c_size], 
						initializer=tf.constant_initializer(0.70710678118))	
		gamma_ii = tf.get_variable('gamma_ii', shape=[1, 1, 1, c_size], 
						initializer=tf.constant_initializer(0.70710678118))
		gamma_ri = tf.get_variable('gamma_ri', shape=[1, 1, 1, c_size], 
										initializer=tf.zeros_initializer())	

		out_r = gamma_rr*r_norm + gamma_ri*i_norm + beta_r
		out_i = gamma_ri*r_norm + gamma_ii*i_norm + beta_i
		return out_r, out_i

def zero_interp(in_patch, ratio, seg_num, in_fft_n, out_fft_n, f_dim):
	in_patch = tf.expand_dims(in_patch, 3)
	in_patch_zero = tf.tile(tf.zeros_like(in_patch),
						[1, 1, 1, ratio-1, 1])
	in_patch = tf.reshape(tf.concat([in_patch, in_patch_zero], 3), 
				[BATCH_SIZE, seg_num, in_fft_n*ratio, f_dim])
	return in_patch[:,:,:out_fft_n,:]

def complex_merge(merge_ratio, name='time_merge'):
	with tf.variable_scope(name):
		if MERGE_INIT == 'complex':
			### init complex with real and image part
			kernel_complex_r = tf.get_variable('kernel_real', shape=[1, 1, 1, 1, merge_ratio, merge_ratio],
											initializer=tf.zeros_initializer())
			kernel_complex_i = tf.get_variable('kernel_imag', shape=[1, 1, 1, 1, merge_ratio, merge_ratio],
											initializer=tf.zeros_initializer())
			kernel_complex = tf.complex(kernel_complex_r, kernel_complex_i, name='kernel')
		else:
			#### init with real number and fft to freq domian
			if MERGE_INIT == 'xavier':
				kernel = tf.get_variable('kernel', shape=[1, 1, 1, 1, merge_ratio, 2*(merge_ratio+1)],
									initializer=tf.contrib.layers.xavier_initializer())
			elif MERGE_INIT == 'zero':
				kernel = tf.get_variable('kernel', shape=[1, 1, 1, 1, merge_ratio, 2*(merge_ratio+1)],
									initializer=tf.zeros_initializer())
			kernel_complex = tf.fft(tf.complex(kernel, 0.*kernel))
			kernel_complex = kernel_complex[:, :, :, :, :, 1:(merge_ratio+1)]
			kernel_complex = tf.transpose(kernel_complex, [0, 1, 2, 3, 5, 4])

		bias_complex_r = tf.get_variable('bias_real', shape=[merge_ratio], 
										initializer=tf.zeros_initializer())
		bias_complex_i = tf.get_variable('bias_imag', shape=[merge_ratio], 
									initializer=tf.zeros_initializer())
		bias_complex = tf.complex(bias_complex_r, bias_complex_i, name='bias')

		return kernel_complex, bias_complex

def atten_merge(patch, kernel, bias):
	## patch with shape (BATCH_SIZE, seg_num, ffn/2+1, c_in, ratio)
	## kernel with shape (1, 1, 1, 1, ratio, ratio)
	## bias with shpe (ratio)
	patch_atten = tf.reduce_sum(tf.expand_dims(patch, 5)*kernel, 4)
	patch_atten = tf.abs(tf.nn.bias_add(patch_atten, bias))
	patch_atten = tf.nn.softmax(patch_atten)
	patch_atten = tf.complex(patch_atten, 0*patch_atten)
	return tf.reduce_sum(patch*patch_atten, 4)


def STFLayer(inputs, fft_list, f_step_list, kenel_len_list, dilation_len_list, c_in, c_out, reuse, out_fft_list=[0], ser_size=SERIES_SIZE, pooling=False, name='STFLayer'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		if pooling:
			assert len(fft_list) == len(out_fft_list)
			fft_n_list = out_fft_list
		else:
			fft_n_list = fft_list

		KERNEL_FFT = GLOBAL_KERNEL_SIZE
		BASIC_LEN = kenel_len_list[0]
		FFT_L_SIZE = len(fft_n_list)
		if FILTER_FLAG:
			## element in patch_kernel_dict with shape (1, 1, fft_n//2+1, c_in, int(c_out/FFT_L_SIZE))
			patch_kernel_dict, patch_bias = complex_glorot_uniform(c_in, c_out, fft_n_list, 
										KERNEL_FFT, use_bias=True, name='patch_filter')
		if FREQ_CONV_FLAG:
			conv_kernel_dict = spectral_filter_gen(c_in, c_out, BASIC_LEN, 
										kenel_len_list, use_bias=False, name='spectral_filter')

		## inputs with shape (batch, c_in, time_len)
		if inputs.get_shape()[-1] == c_in:
			inputs = tf.transpose(inputs, [0, 2, 1])

		patch_fft_list = []
		patch_mask_list = []
		for idx in range(len(fft_n_list)):
			patch_fft_list.append(0.)
			patch_mask_list.append([])

		for fft_idx, fft_n in enumerate(fft_n_list):
			## patch_fft with shape (batch, c_in, seg_num, fft_n//2+1)
			if pooling:
				in_f_step = fft_list[fft_idx]
				f_step = in_f_step
				patch_fft =  tf.contrib.signal.stft(inputs, 
								window_fn=None,
								frame_length=in_f_step, frame_step=f_step, fft_length=in_f_step)
				patch_fft = patch_fft[:,:,:,:int(fft_n/2)+1]
			else:
				f_step = fft_n
				patch_fft =  tf.contrib.signal.stft(inputs, 
								window_fn=None,
								frame_length=fft_n, frame_step=f_step, fft_length=fft_n)
			## patch_fft with shape (batch, seg_num, fft_n//2+1, c_in)
			patch_fft = tf.transpose(patch_fft, [0, 2, 3, 1])

			for fft_idx2, tar_fft_n in enumerate(fft_n_list):
				if tar_fft_n < fft_n:
					continue
				elif tar_fft_n == fft_n:
					patch_mask = tf.ones_like(patch_fft)
					for exist_mask in patch_mask_list[fft_idx2]:
						patch_mask = patch_mask - exist_mask
					patch_fft_list[fft_idx2] = patch_fft_list[fft_idx2] + patch_mask*patch_fft
				else:
					time_ratio = tar_fft_n//fft_n
					patch_fft_mod = tf.reshape(patch_fft, 
						[BATCH_SIZE, ser_size//tar_fft_n, time_ratio, int(fft_n/2)+1, c_in])
					
					patch_fft_mod = tf.transpose(patch_fft_mod, [0, 1, 3, 4, 2])

					merge_kernel, merge_bias = complex_merge(time_ratio, 
									name='complex_time_merge_{0}_{1}'.format(fft_n, tar_fft_n))
					
					patch_fft_mod = atten_merge(patch_fft_mod, merge_kernel, merge_bias)*float(time_ratio)
					
					patch_mask = tf.ones_like(patch_fft_mod)
					patch_mask = zero_interp(patch_mask, time_ratio, ser_size//tar_fft_n, 
									int(fft_n/2)+1, int(tar_fft_n/2)+1, c_in)
					for exist_mask in patch_mask_list[fft_idx2]:
						patch_mask = patch_mask - exist_mask
					patch_mask_list[fft_idx2].append(patch_mask)

					patch_fft_mod = zero_interp(patch_fft_mod, time_ratio, ser_size//tar_fft_n, 
									int(fft_n/2)+1, int(tar_fft_n/2)+1, c_in)

					patch_fft_list[fft_idx2] = patch_fft_list[fft_idx2] + patch_mask*patch_fft_mod

		patch_time_list = []
		for fft_idx, fft_n in enumerate(fft_n_list):
			# f_step = f_step_list[fft_idx]
			k_len = kenel_len_list[fft_idx]
			d_len = dilation_len_list[fft_idx]
			paddings = [(k_len*d_len-d_len)/2, (k_len*d_len-d_len)/2]

			patch_fft = patch_fft_list[fft_idx]

			patch_fft_r = tf.real(patch_fft)
			patch_fft_i = tf.imag(patch_fft)

			if INPUT_COMPLEX_NORM_FLAG:
				patch_fft_r, patch_fft_i = complex_layerNorm(patch_fft_r, patch_fft_i, 
										name='complex_layerNorm_{0}'.format(fft_idx))

			if FREQ_CONV_FLAG:
				## spectral padding
				real_pad_l = tf.reverse(patch_fft_r[:,:,1:1+paddings[0],:], [2])
				real_pad_r = tf.reverse(patch_fft_r[:,:,-1-paddings[1]:-1,:], [2])
				patch_fft_r = tf.concat([real_pad_l, patch_fft_r, real_pad_r], 2)

				imag_pad_l = tf.reverse(patch_fft_i[:,:,1:1+paddings[0],:], [2])
				imag_pad_r = tf.reverse(patch_fft_i[:,:,-1-paddings[1]:-1,:], [2])
				patch_fft_i = tf.concat([-imag_pad_l, patch_fft_i, -imag_pad_r], 2)

				conv_kernel_r, conv_kernel_i = conv_kernel_dict[k_len]

				if d_len > 1:
					conv_kernel_r = tf.expand_dims(conv_kernel_r, 2)
					conv_kernel_i = tf.expand_dims(conv_kernel_i, 2)
					zero_f = tf.tile(tf.zeros_like(conv_kernel_r), [1, 1, d_len-1, 1, 1])
					conv_kernel_r = tf.reshape(tf.concat([conv_kernel_r, zero_f], 2), 
											[1, k_len*d_len, c_in, c_out/len(fft_n_list)])
					conv_kernel_i = tf.reshape(tf.concat([conv_kernel_i, zero_f], 2),
											[1, k_len*d_len, c_in, c_out/len(fft_n_list)])
					conv_kernel_r = conv_kernel_r[:,:(k_len*d_len-d_len+1),:,:]
					conv_kernel_i = conv_kernel_i[:,:(k_len*d_len-d_len+1),:,:]

				patch_conv_rr = tf.nn.conv2d(patch_fft_r, conv_kernel_r, strides=[1,1,1,1], 
							padding='VALID', data_format='NHWC')
				patch_conv_ri = tf.nn.conv2d(patch_fft_r, conv_kernel_i, strides=[1,1,1,1], 
							padding='VALID', data_format='NHWC')
				patch_conv_ir = tf.nn.conv2d(patch_fft_i, conv_kernel_r, strides=[1,1,1,1], 
							padding='VALID', data_format='NHWC')
				patch_conv_ii = tf.nn.conv2d(patch_fft_i, conv_kernel_i, strides=[1,1,1,1], 
							padding='VALID', data_format='NHWC')

				patch_out_r = patch_conv_rr - patch_conv_ii
				patch_out_i = patch_conv_ri + patch_conv_ir

			if FILTER_FLAG:
				patch_kernel = patch_kernel_dict[fft_n]
				patch_fft = tf.complex(patch_fft_r, patch_fft_i)
				patch_fft = tf.tile(tf.expand_dims(patch_fft, 4), [1, 1, 1, 1, c_out//FFT_L_SIZE])
				patch_fft_out = patch_fft*patch_kernel
				patch_fft_out = tf.reduce_sum(patch_fft_out, 3)
				patch_out_r = tf.real(patch_fft_out)
				patch_out_i = tf.imag(patch_fft_out)

			if ACT_DOMIAN == 'freq':
				patch_out_r = tf.nn.leaky_relu(patch_out_r)
				patch_out_i = tf.nn.leaky_relu(patch_out_i)

			patch_out = tf.complex(patch_out_r, patch_out_i)

			## patch_fft_fin with shape (batch, c_out/FFT_L_SIZE, seg_num, fft_n//2+1)
			patch_fft_fin = tf.transpose(patch_out, [0, 3, 1, 2])
			patch_time = tf.contrib.signal.inverse_stft(patch_fft_fin, 
					frame_length=fft_n, frame_step=fft_n, fft_length=fft_n,
					window_fn=None)
			patch_time = tf.transpose(patch_time, [0, 2, 1])
			patch_time_list.append(patch_time)

		patch_time_final = tf.concat(patch_time_list, 2)

		if FILTER_FLAG:
			patch_time_final = tf.nn.bias_add(patch_time_final, tf.real(patch_bias))

		if ACT_DOMIAN == 'time':
			patch_time_final = tf.nn.leaky_relu(patch_time_final)
		return patch_time_final

def STFNet(inputs, train, reuse=False, name='STFNet'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		## input with shape (BATCH_SIZE, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM)
		inputs = tf.reshape(inputs, [BATCH_SIZE, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM])
		acc_in, gyro_in = tf.split(inputs, num_or_size_splits=2, axis=2)

		acc_layer1 = STFLayer(acc_in, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN,
							SENSOR_AXIS, GEN_C_OUT, reuse, name='acc_layer1')
		if DROP_FLAG:
			acc_layer1 = layers.dropout(acc_layer1, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='acc_dropout1')

		acc_layer2 = STFLayer(acc_layer1, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							GEN_C_OUT, GEN_C_OUT, reuse, name='acc_layer2')
		if DROP_FLAG:
			acc_layer2 = layers.dropout(acc_layer2, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='acc_dropout2')

		acc_layer3 = STFLayer(acc_layer2, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							GEN_C_OUT, GEN_C_OUT//2, reuse, name='acc_layer3')
		if DROP_FLAG:
			acc_layer3 = layers.dropout(acc_layer3, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, int((GEN_C_OUT/2)/len(GEN_FFT_N))*len(GEN_FFT_N)], 
				scope='acc_dropout3')

		gyro_layer1 = STFLayer(gyro_in, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							SENSOR_AXIS, GEN_C_OUT, reuse, name='gyro_layer1')
		if DROP_FLAG:
			gyro_layer1 = layers.dropout(gyro_layer1, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='gyro_dropout1')

		gyro_layer2 = STFLayer(gyro_layer1, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							GEN_C_OUT, GEN_C_OUT, reuse, name='gyro_layer2')
		if DROP_FLAG:
			gyro_layer2 = layers.dropout(gyro_layer2, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='gyro_dropout2')

		gyro_layer3 = STFLayer(gyro_layer2, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							GEN_C_OUT, GEN_C_OUT//2, reuse, name='gyro_layer3')
		if DROP_FLAG:
			gyro_layer3 = layers.dropout(gyro_layer3, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, int((GEN_C_OUT/2)/len(GEN_FFT_N))*len(GEN_FFT_N)], 
				scope='gyro_dropout3')

		sensor_in = tf.concat([acc_layer3, gyro_layer3], 2)
		sensor_layer1 = STFLayer(sensor_in, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
					int((GEN_C_OUT/2)/len(GEN_FFT_N))*len(GEN_FFT_N)*2, GEN_C_OUT, reuse, 
					out_fft_list=GEN_FFT_N2, ser_size=SERIES_SIZE2, pooling=True, name='sensor_layer1')
		if DROP_FLAG:
			sensor_layer1 = layers.dropout(sensor_layer1, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='sensor_dropout1')

		sensor_layer2 = STFLayer(sensor_layer1, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							GEN_C_OUT, GEN_C_OUT, reuse, ser_size=SERIES_SIZE2, name='sensor_layer2')
		if DROP_FLAG:
			sensor_layer2 = layers.dropout(sensor_layer2, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='sensor_dropout2')

		sensor_layer3 = STFLayer(sensor_layer2, GEN_FFT_N, GEN_FFT_STEP, FILTER_LEN, DILATION_LEN, 
							GEN_C_OUT, GEN_C_OUT, reuse, ser_size=SERIES_SIZE2, name='sensor_layer3')
		if DROP_FLAG:
			sensor_layer3 = layers.dropout(sensor_layer3, KEEP_PROB, is_training=train, 
				noise_shape=[BATCH_SIZE, 1, GEN_C_OUT], scope='sensor_dropout3')

		sensor_out = tf.reduce_mean(sensor_layer3, 1)

		logits = layers.fully_connected(sensor_out, OUT_DIM, activation_fn=None, scope='output')

		return logits

global_step = tf.Variable(0, trainable=False)

batch_feature, batch_label = input_pipeline_har(os.path.join(select, 'train.tfrecord'), BATCH_SIZE, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM, OUT_DIM)
batch_eval_feature, batch_eval_label = input_pipeline_har(os.path.join(select, 'eval.tfrecord'), BATCH_SIZE, SERIES_SIZE, SENSOR_AXIS*SENSOR_NUM, OUT_DIM, shuffle_sample=False)

logits = STFNet(batch_feature, True, name='STFNet')

predict = tf.argmax(logits, axis=1)

batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
loss = tf.reduce_mean(batchLoss)

logits_eval = STFNet(batch_eval_feature, False, reuse=True, name='STFNet')
predict_eval = tf.argmax(logits_eval, axis=1)
loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))

t_vars = tf.trainable_variables()

regularizers = 0.
for var in t_vars:
	print(var.name)
	if 'angle' in var.name:
		continue
	regularizers += tf.nn.l2_loss(var)
loss += 5e-4 * regularizers


if CLIP_FLAG:
	discOpt = tf.train.AdamOptimizer(
			learning_rate = ADAM_LR,
			beta1 = ADAM_B1,
			beta2 = ADAM_B2
		)
	gvs = discOpt.compute_gradients(loss, var_list=t_vars)
	capped_gvs = [(tf.clip_by_value(grad, -CLIP_VAL, CLIP_VAL), var) for grad, var in gvs]
	discOptimizer = discOpt.apply_gradients(capped_gvs)
else:
	discOptimizer = tf.train.AdamOptimizer(
			learning_rate = ADAM_LR,
			beta1 = ADAM_B1,
			beta2 = ADAM_B2
		).minimize(loss, var_list=t_vars)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	max_accuracy = -1
	max_f1 = -1

	for iteration in range(TOTAL_ITER_NUM):
		_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])
		_label = np.argmax(_trainY, axis=1)
		_accuracy = np.mean(_label == _predict)
		# plot.plot('train cross entropy', lossV)
		# plot.plot('train accuracy', _accuracy)

		print("iteration: {}, train cross_entropy={}, train accuracy={}".format(
			iteration, lossV, _accuracy))

		if iteration % 50 == 49:
			dev_accuracy = []
			dev_cross_entropy = []
			total_label = []
			total_predt = []
			for eval_idx in range(EVAL_ITER_NUM):
				eval_loss_v, _trainY, _predict = sess.run([loss, batch_eval_label, predict_eval])
				_label = np.argmax(_trainY, axis=1)
				_accuracy = np.mean(_label == _predict)
				total_label += _label.tolist()
				total_predt += _predict.tolist()
				dev_accuracy.append(_accuracy)
				dev_cross_entropy.append(eval_loss_v)
			
			# plot.plot('dev accuracy', np.mean(dev_accuracy))
			# plot.plot('dev cross entropy', np.mean(dev_cross_entropy))
			# plot.plot('dev macro f1', f1_score(total_label, total_predt, average='macro'))
			current_accuracy = np.mean(dev_accuracy)
			current_cross_entropy = np.mean(dev_cross_entropy)
			current_f1 = f1_score(total_label, total_predt, average='macro')
			if current_accuracy > max_accuracy:
				max_accuracy = np.mean(dev_accuracy)
			if current_f1 > max_f1:
				max_f1 = current_f1
			print("iteration: {}, test accuracy={}, test cross_entropy={}, test f1={}, max_accuracy={}, max_f1={}".format(
				iteration, 
				current_accuracy, 
				current_cross_entropy, 
				current_f1,
				max_accuracy, 
				max_f1))
			# print('dev accuracy: ', np.mean(dev_accuracy))
			# print('dev cross entropy: ', np.mean(dev_cross_entropy))
			# print('dev macro f1: ', f1_score(total_label, total_predt, average='macro'))

		# if (iteration < 5) or (iteration % 50 == 49):
		# 	plot.flush()

		# plot.tick()

