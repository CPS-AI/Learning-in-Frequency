import tensorflow as tf

def atten_merge(patch, kernel, bias):
	## patch with shape (BATCH_SIZE, seg_num, ffn/2+1, c_in, ratio)
	## kernel with shape (1, 1, 1, 1, ratio, ratio)
	## bias with shape (ratio)
	patch_atten = tf.reduce_sum(tf.expand_dims(patch, 5)*kernel, 4)
	patch_atten = tf.abs(tf.nn.bias_add(patch_atten, bias))
	patch_atten = tf.nn.softmax(patch_atten)
	patch_atten = tf.complex(patch_atten, 0*patch_atten)

	return tf.reduce_sum(patch*patch_atten, 4)
