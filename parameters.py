import tensorflow as tf
import numpy as np

def initialize_parameters(n_ch, d):
	parameters = {}
	W1 = tf.get_variable('W1', [5, 5, n_ch, 128], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable('W2', [5, 5, 128, 256], initializer = tf.contrib.layers.xavier_initializer())
	W3 = tf.get_variable('W3', [5, 5, 256, 512], initializer = tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable('W4', [4, 4, 512, 1024], initializer = tf.contrib.layers.xavier_initializer())

	W5 = tf.get_variable('W5', [4, 4, d, 1024], initializer = tf.contrib.layers.xavier_initializer())
	W6 = tf.get_variable('W6', [5, 5, 1024, 512], initializer = tf.contrib.layers.xavier_initializer())
	W7 = tf.get_variable('W7', [5, 5, 512, 256], initializer = tf.contrib.layers.xavier_initializer())
	W8 = tf.get_variable('W8', [5, 5, 256, 128], initializer = tf.contrib.layers.xavier_initializer())
	W9 = tf.get_variable('W9', [5, 5, 128, n_ch], initializer = tf.contrib.layers.xavier_initializer())

	W10 = tf.get_variable('W10', [5, 5, 512, 384], initializer = tf.contrib.layers.xavier_initializer())
	W11 = tf.get_variable('W11', [5, 5, 384, 320], initializer = tf.contrib.layers.xavier_initializer())
	W12 = tf.get_variable('W12', [5, 5, 320, 288], initializer = tf.contrib.layers.xavier_initializer())
	W13 = tf.get_variable('W13', [5, 5, 288, 256], initializer = tf.contrib.layers.xavier_initializer())
	W14 = tf.get_variable('W14', [5, 5, 256, 128], initializer = tf.contrib.layers.xavier_initializer())

	W15 = tf.get_variable('W15', [14*14*128, 4096], initializer = tf.contrib.layers.xavier_initializer())
	W16 = tf.get_variable('W16', [4096, d], initializer = tf.contrib.layers.xavier_initializer())

	b1 = tf.get_variable('b1', [128], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', [256], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable('b3', [512], initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable('b4', [1024], initializer = tf.contrib.layers.xavier_initializer())

	b5 = tf.get_variable('b5', [1024], initializer = tf.contrib.layers.xavier_initializer())
	b6 = tf.get_variable('b6', [512], initializer = tf.contrib.layers.xavier_initializer())
	b7 = tf.get_variable('b7', [256], initializer = tf.contrib.layers.xavier_initializer())
	b8 = tf.get_variable('b8', [128], initializer = tf.contrib.layers.xavier_initializer())
	b9 = tf.get_variable('b9', [n_ch], initializer = tf.contrib.layers.xavier_initializer())

	b15 = tf.get_variable('b15', [4096], initializer = tf.contrib.layers.xavier_initializer())
	b16 =  tf.get_variable('b16', [d], initializer = tf.contrib.layers.xavier_initializer())

	parameters = {'W1' : W1, 'b1' : b1,
				  'W2' : W2, 'b2' : b2,
				  'W3' : W3, 'b3' : b3,
				  'W4' : W4, 'b4' : b4,
				  'W5' : W5, 'b5' : b5,
				  'W6' : W6, 'b6' : b6,
				  'W7' : W7, 'b7' : b7,
				  'W8' : W8, 'b8' : b8,
				  'W9' : W9, 'b9' : b9,
				  'W10' : W10,
				  'W11' : W11,
				  'W12' : W12,
				  'W13' : W13,
				  'W14' : W14,
				  'W15' : W15, 'b15' : b15,
				  'W16' : W16, 'b16' : b16}

	return parameters

def save_parameters(parameters, fromvae):
	"""
	docstring here
		:param parameters: the dictionary of parameters to be saved
		:param epoch: nuber of epochs the parametes are trained on
		:param fromvae: is the function called by vae or mdn
	"""
	path = "params\\"
	if fromvae:
		path += "vae\\"
	else:
		path += "mdn\\"
	for key, value in parameters.items():
		np.save(path + str(key), value)