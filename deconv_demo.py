import tensorflow as tf
import ops # Import operation level interfaces

class STFTransConvLayer(tf.keras.layers.Layer):
	...
	# When this Tensorflow layer is called
	def call(self, inputs):
		# Initialize STFNet convolution kernels
		kernel_list = ops.initialize_stf_conv_kernels(
			self.c_in, # Input channel dimension
			self.c_out, # Output channel dimension
			self.kernel_size) # Convolution kernel size
		# Do multi-resolution STFT
		patch_fft_list = ops.multi_resolution_stft(inputs, 
      self.fft_list) # Segment sizes for  
                     # multi-dimensional STFT 
		patch_out_list = []
		for n, fft_n in enumerate(self.fft_list):
      d_ratio = fft_n // self.fft_list[0]
			# Do spectral padding
			patch_fft = ops.spectral_padding(
        patch_fft_list[n],fft_n,d_ratio)
			# Dilate kernels
			conv_kernel = ops.dilate_kernel(kernel_list[n], 
        d_ratio)
			# Do transpose convoultion  		
      patch_fft_up = self.__transposed_convolution(
        patch_fft,
        conv_kernel,
        self.upsample_times) # Target upsample times
			patch_up = self.__inverse_stft(patch_fft_up)
			patch_out_list.append(patch_up)
		return tf.concat(patch_out_list, 2)