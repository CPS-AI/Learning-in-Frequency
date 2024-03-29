import tensorflow as tf
import ops


class STFFilterLayer(tf.keras.layers.Layer):
	def __init__(self, fft_list, kernel_size, c_in, c_out, act_domain, 
		kernel_domain='complex',
		name='stf_filter_layer'):
		"""
		Args:
			input:
			fft_list:
		"""
		super(STFFilterLayer, self).__init__(name=name)
		self.fft_list = fft_list
		self.kernel_size = kernel_size
		self.c_in = c_in
		self.c_out = c_out
		self.act_domain = act_domain
		self.kernel_domain = kernel_domain

		self.glorot_initializer = tf.keras.initializers.glorot_uniform()
		self.zeros_initializer = tf.zeros_initializer()

	def call(self, inputs):
		patch_kernel_dict, patch_bias = self.__initialize_stf_filters(
			self.c_in, self.c_out
		)

		# [batch, feature, time]
		inputs = tf.transpose(inputs, [0, 2, 1])

		patch_fft_list, patch_mask_list = ops.multi_resolution_stft(
			inputs, self.fft_list
		)

		patch_time_list = []
		for fft_idx, fft_n in enumerate(self.fft_list):
			k_len = self.kernel_size
			d_len = fft_n // self.fft_list[0]
			paddings = [
				(k_len * d_len - d_len) // 2,
				(k_len * d_len - d_len) // 2,
			]

			patch_fft = patch_fft_list[fft_idx]

			patch_fft_r = tf.math.real(patch_fft)
			patch_fft_i = tf.math.imag(patch_fft)

			patch_kernel = patch_kernel_dict[fft_n]
			patch_fft = tf.complex(patch_fft_r, patch_fft_i)
			patch_fft = tf.tile(
				tf.expand_dims(patch_fft, 4),
				[1, 1, 1, 1, self.c_out // len(self.fft_list)],
			)
			patch_fft_out = patch_fft * patch_kernel
			patch_fft_out = tf.reduce_sum(patch_fft_out, 3)
			patch_out_r = tf.math.real(patch_fft_out)
			patch_out_i = tf.math.imag(patch_fft_out)

			if self.act_domain == "freq":
				patch_out_r = tf.nn.leaky_relu(patch_out_r)
				patch_out_i = tf.nn.leaky_relu(patch_out_i)

			patch_out = tf.complex(patch_out_r, patch_out_i)
			## [batch, c_out/FFT_L_SIZE, seg_num, fft_n//2+1]
			patch_fft_fin = tf.transpose(patch_out, [0, 3, 1, 2])

			patch_time = tf.signal.inverse_stft(
				patch_fft_fin,
				frame_length=fft_n,
				frame_step=fft_n,
				fft_length=fft_n,
				window_fn=None,
			)
			patch_time = tf.transpose(patch_time, [0, 2, 1])
			patch_time_list.append(patch_time)

		patch_time_final = tf.concat(patch_time_list, 2)
		patch_time_final = tf.reshape(
			patch_time_final, [-1, inputs.shape[-1], self.c_out]
		)

		patch_time_final = tf.nn.bias_add(patch_time_final, tf.math.real(patch_bias))

		if self.act_domain == "time":
			patch_time_final = tf.nn.leaky_relu(patch_time_final)
		return patch_time_final

	def __initialize_stf_filters(
		self,
		c_in,
		c_out_total,
		filter_type="complex",
		use_bias=True,
	):
		"""
		Initialize STF-Filters
		Args:
			c_in: Input channel size
			c_out_total: Output channel size
			fft_list: Multi-dimensional stft segment frame lengths
			basic_len: Default kernel length
			filter_type: Filter type. 'complex' creates seperate kernels for the 
				real and imagine parts. 'real' only creates a kernel for the real 
				part and masks the imagine part 
			use_bias: A boolean value. Whether use bias or not.
		"""
		basic_len = self.fft_list[1]
		c_out = c_out_total // len(self.fft_list)

		if self.kernel_domain == 'real':
			kernel = self.glorot_initializer(
				shape=(1, 1, c_in * c_out, basic_len)
			)
			kernel_complex_org = tf.signal.fft(tf.complex(kernel, 0.*kernel))
			kernel_complex_org = tf.transpose(kernel_complex_org, [0, 1, 3, 2])
			kernel_complex_org = kernel_complex_org[:,:,:basic_len//2+1,:]
		else:
			kernel_r = self.glorot_initializer(
				shape=(1, 1, basic_len // 2 + 1, c_in * c_out)
			)
			kernel_i = self.glorot_initializer(
				shape=(1, 1, basic_len // 2 + 1, c_in * c_out)
			)
			kernel_complex_org = tf.complex(kernel_r, kernel_i)
		
		kernel_complex_dict = {}
		for fft_elem in self.fft_list:
			# If fft_elem < basic_len, shrink the initialized kernel
			if fft_elem != basic_len:
				kernel_complex_r = tf.image.resize(
					tf.math.real(kernel_complex_org), [1, fft_elem // 2 + 1]
				)
				kernel_complex_i = tf.image.resize(
					tf.math.imag(kernel_complex_org), [1, fft_elem // 2 + 1]
				)
				kernel_complex_dict[fft_elem] = tf.reshape(
					tf.complex(kernel_complex_r, kernel_complex_i),
					[1, 1, fft_elem // 2 + 1, c_in, c_out],
				)
			else:
				kernel_complex_dict[fft_elem] = tf.reshape(
					kernel_complex_org, [1, 1, fft_elem // 2 + 1, c_in, c_out]
				)

		if use_bias:
			bias_complex_r = self.glorot_initializer(shape=[c_out * len(self.fft_list)])
			bias_complex_i = self.glorot_initializer(shape=[c_out * len(self.fft_list)])
			bias_complex = tf.complex(bias_complex_r, bias_complex_i)
			return kernel_complex_dict, bias_complex
		else:
			return kernel_complex_dict
