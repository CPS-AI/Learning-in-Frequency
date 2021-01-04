import tensorflow as tf
import ops

class STFConvLayer(tf.keras.layers.Layer):

	def __init__(self,
				fft_list,
				kernel_size, 
				c_in,
				c_out, 
				act_domain,
				kernel_domain='freq',
				name='stf_conv_layer'
				):
		'''
		Args:
			input:
			fft_list:
		'''
		super(STFConvLayer, self).__init__(name=name)	
		self.fft_list = fft_list
		self.kernel_size = kernel_size
		self.c_in = c_in
		self.c_out = c_out
		self.act_domain = act_domain
		self.kernel_domain = kernel_domain

		self.glorot_initializer = tf.keras.initializers.glorot_uniform()
		self.zeros_initializer = tf.zeros_initializer()

	def call(self, inputs):
		conv_kernel_dict = self.__initialize_stf_kernels(self.c_in, 
														self.c_out, 
														self.kernel_size, 
														use_bias=False)

		# [batch, feature, time]
		inputs = tf.transpose(inputs, [0, 2, 1])
		
		patch_fft_list, patch_mask_list = ops.multi_resolution_stft(
			inputs, self.fft_list
		)

		# Do convolution
		patch_time_list = []
		for fft_idx, fft_n in enumerate(self.fft_list):
			# f_step = f_step_list[fft_idx]
			k_len = self.kernel_size
			d_len = int(fft_n / self.fft_list[0])
			paddings = [int((k_len*d_len-d_len)/2), int((k_len*d_len-d_len)/2)]

			patch_fft = patch_fft_list[fft_idx]

			patch_fft_r = tf.math.real(patch_fft)
			patch_fft_i = tf.math.imag(patch_fft)

			# Convolution
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
										[1, k_len*d_len, self.c_in, self.c_out//len(self.fft_list)])
				conv_kernel_i = tf.reshape(tf.concat([conv_kernel_i, zero_f], 2),
										[1, k_len*d_len, self.c_in, self.c_out//len(self.fft_list)])
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

			if self.act_domain == 'freq':
				patch_out_r = tf.nn.leaky_relu(patch_out_r)
				patch_out_i = tf.nn.leaky_relu(patch_out_i)

			patch_out = tf.complex(patch_out_r, patch_out_i)
			## [batch, c_out/FFT_L_SIZE, seg_num, fft_n//2+1]
			patch_fft_fin = tf.transpose(patch_out, [0, 3, 1, 2])

			patch_time = tf.signal.inverse_stft(patch_fft_fin, 
					frame_length=fft_n, frame_step=fft_n, fft_length=fft_n,
					window_fn=None)
			patch_time = tf.transpose(patch_time, [0, 2, 1])
			patch_time_list.append(patch_time)

		patch_time_final = tf.concat(patch_time_list, 2)
		patch_time_final = tf.reshape(patch_time_final, 
										[-1, inputs.shape[-1], self.c_out])
		
		if self.act_domain == 'time':
			patch_time_final = tf.nn.leaky_relu(patch_time_final)

		return patch_time_final

	def __initialize_stf_kernels(self, 
							c_in, 
							c_out_total, 
							basic_len, 
							use_bias=True):
		c_out = int(c_out_total)//len(self.fft_list)
		if self.kernel_domain == 'freq':
			kernel_r = self.glorot_initializer(shape=[1, basic_len, c_in, c_out])
			kernel_i = self.glorot_initializer(shape=[1, basic_len, c_in, c_out])			
		else:
			kernel = self.glorot_initializer(shape=[1, basic_len, c_out, 2*(c_in+1)])
			kernel_c = tf.signal.fft(tf.complex(kernel, 0.*kernel))
			kernel_c = kernel_c[:, :, :, 1:(c_in+1)]
			kernel_c = tf.transpose(kernel_c, [0, 1, 3, 2])
			kernel_r = tf.math.real(kernel_c)
			kernel_i = tf.math.imag(kernel_c)

		kernel_dict = {}
		kernel_dict[basic_len] = [kernel_r, kernel_i]

		if use_bias: 
			bias_complex_r = self.glorot_initializer(shape=[c_out]) 
			bias_complex_i = self.glorot_initializer(shape=[c_out])
			bias_complex = tf.complex(bias_complex_r, bias_complex_i)
			return kernel_dict, bias_complex
		else:
			return kernel_dict