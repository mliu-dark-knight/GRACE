import math
import tflearn
import tensorflow as tf

def weight(name, shape, init='he'):
	assert init == 'he' and len(shape) == 2
	var = tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=math.sqrt(2.0 / shape[0])))
	tf.add_to_collection('l2', tf.nn.l2_loss(var))
	return var

def bias(name, dim, initial_value=1e-2):
	return tf.get_variable(name, dim, initializer=tf.constant_initializer(initial_value))

def fully_connected(input, num_neurons, name, activation='elu'):
	func = {'tanh': tflearn.activations.tanh, 'linear': tflearn.activations.linear, 'relu': tflearn.activations.relu,
	        'elu': tflearn.activations.elu, 'lrelu': tflearn.activations.leaky_relu, 'selu': tflearn.activations.selu}
	W = weight(name + '_W', [input.get_shape().as_list()[1], num_neurons], init='he')
	l = tf.matmul(input, W) + bias(name + '_b', num_neurons)
	return func[activation](l)
