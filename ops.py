import tensorflow as tf

def atten_merge(patch, kernel, bias):
	'''
	patch: with shape (BATCH_SIZE, seg_num, ffn/2+1, c_in, ratio)
	kernel: with shape (1, 1, 1, 1, ratio, ratio)
	bias: with shape (ratio)
	'''
	patch_atten = tf.reduce_sum(tf.expand_dims(patch, 5)*kernel, 4)
	patch_atten = tf.abs(tf.nn.bias_add(patch_atten, bias))
	patch_atten = tf.nn.softmax(patch_atten)
	patch_atten = tf.complex(patch_atten, 0*patch_atten)

	return tf.reduce_sum(patch*patch_atten, 4)

def multi_dimension_stft(inputs, fft_list):
	'''
		inputs: with shape: [batch, time_series, feature]
	'''
	patch_fft_list = []
	patch_mask_list = []

	for idx in range(len(fft_list)):
		patch_fft_list.append(0.)
		patch_mask_list.append([])

	for fft_idx, fft_n in enumerate(fft_list):
		patch_fft =  tf.signal.stft(inputs, 
					window_fn=None,
					frame_length=fft_n, 
					frame_step=fft_n, 
			 		fft_length=fft_n)
		patch_fft = tf.transpose(patch_fft, [0, 2, 3, 1])

		hologram_interleave(patch_fft, 
							patch_fft_list, 
							patch_mask_list, 
							fft_n,
							fft_list,
							inputs.shape[1])
		
def hologram_interleave(patch_fft, 
						patch_fft_list, 
						patch_mask_list, 
						fft_n,
						fft_list,
						series_size):
	'''
		patch_fft: with shape [batch, seg_num, fft_n//2+1, c_in]
	'''
	c_in = patch_fft.shape[-1]

	for fft_idx, tar_fft_n in enumerate(fft_list):
		if tar_fft_n < fft_n:
			continue
		elif tar_fft_n == fft_n:
			patch_mask = tf.ones_like(patch_fft)
			for exist_mask in patch_mask_list[fft_idx]:
				patch_mask = patch_mask - exist_mask
			patch_fft_list[fft_idx] = patch_fft_list[fft_idx] + patch_mask*patch_fft
		else:
			time_ratio = tar_fft_n//fft_n
			patch_fft_mod = tf.reshape(patch_fft, 
				[-1, series_size//tar_fft_n, time_ratio, fft_n//2+1, c_in])
			
			patch_fft_mod = tf.transpose(patch_fft_mod, [0, 1, 3, 4, 2])

			merge_kernel, merge_bias = complex_merge(time_ratio)
			
			patch_fft_mod = atten_merge(patch_fft_mod, merge_kernel, merge_bias)*float(time_ratio)
			
			patch_mask = tf.ones_like(patch_fft_mod)
			patch_mask = zero_interp(patch_mask, time_ratio, series_size//tar_fft_n, 
							fft_n//2+1, tar_fft_n//2+1, c_in)
			for exist_mask in patch_mask_list[fft_idx]:
				patch_mask = patch_mask - exist_mask
			patch_mask_list[fft_idx].append(patch_mask)

			patch_fft_mod = zero_interp(patch_fft_mod, time_ratio, series_size//tar_fft_n, 
							fft_n//2+1, tar_fft_n//2+1, c_in)

			patch_fft_list[fft_idx] = patch_fft_list[fft_idx] + patch_mask*patch_fft_mod

def complex_merge(merge_ratio):
	# kernel = tf.Variable(lambda: tf.zeros_initializer()(
	# 	shape=[1, 1, 1, 1, merge_ratio, 2*(merge_ratio+1)]))
	kernel = tf.zeros_initializer()(
		shape=(1, 1, 1, 1, merge_ratio, 2*(merge_ratio+1)))
	kernel_complex = tf.signal.fft(tf.complex(kernel, 0.*kernel))
	kernel_complex = kernel_complex[:, :, :, :, :, 1:(merge_ratio+1)]
	kernel_complex = tf.transpose(kernel_complex, [0, 1, 2, 3, 5, 4])

	bias_complex_r = tf.zeros_initializer()(
		shape=(merge_ratio))
	bias_complex_i = tf.zeros_initializer()(
		shape=(merge_ratio))
	bias_complex = tf.complex(bias_complex_r, bias_complex_i)

	return kernel_complex, bias_complex

def zero_interp(in_patch, ratio, seg_num, in_fft_n, out_fft_n, f_dim):
	in_patch = tf.expand_dims(in_patch, 3)
	in_patch_zero = tf.tile(tf.zeros_like(in_patch),
						[1, 1, 1, ratio-1, 1])
	in_patch = tf.reshape(tf.concat([in_patch, in_patch_zero], 3), 
				[-1, seg_num, in_fft_n*ratio, f_dim])
	return in_patch[:,:,:out_fft_n,:]