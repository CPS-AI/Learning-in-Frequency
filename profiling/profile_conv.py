import numpy as np
import math
import tensorflow as tf
import time
import sys
sys.path.insert(1, '../')

from stf_conv_layer import STFConvLayer

fft_list = [16, 32, 64, 128]
kernel_size = 3
c_in = 64
c_out = 64
act_domain = 'freq'
name = 'stfnet-conv'

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

layer = STFConvLayer(
    fft_list, kernel_size, c_in, c_out, act_domain, name
)

train_data = np.ones((1, 512, c_in))

test_data = np.ones((1, 512, c_out))

time_callback = TimeHistory()

model = tf.keras.Sequential()
model.add(layer)
model.compile(loss=tf.keras.losses.MSE, 
	optimizer=tf.keras.optimizers.SGD())

model.fit(train_data, test_data, 
	batch_size=1, 
	epochs=5000, 
	callbacks=[time_callback], 
	verbose=0)

# graph = tf.compat.v1.get_default_graph()
# opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    

# opt = (tf.compat.v1.profiler.ProfileOptionBuilder(
# 	tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
# 	.with_node_names(show_name_regexes=['.*acc_layer1.*',
# 								   '.*gyro_layer1.*',
# 								   '.*sensor_layer1.*'
# 								   ])
# 		.order_by('flops')
# 		.build())

# flops = tf.compat.v1.profiler.profile(graph, cmd='op', options=opts)
# if flops is not None:
# 	print('flops.total_float_ops = ', flops.total_float_ops)

times = time_callback.times
print("mean time cost = ", np.mean(times[2000:]))
# model.summary()
# in=16, out=16, k=3, 0.004964182138442993
# in=32, out=32, k=3,  0.004518288135528565
# in=32, out=32, k=5,  0.004569529294967651
# in=32, out=32, k=8,  0.004407807350158692

# in=64, out=64, k=3,  0.005505629777908325
# in=128, out=128, k=3,  0.00973701786994934
