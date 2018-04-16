import tensorflow as tf
import numpy as numpy
import pandas as pd

nh = 64  # height of input image
nw = 64  # width of input image
nc = 2  # initial  umber of channels
d = 64

X = tf.placeholder("float32", shape = [None, 64, 64, 2], name = 'Input image data')

def params(d):
	parameters = {}
	W1 = tf.get_variable('W1', [5, 5, 2, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W2 = tf.get_variable('W2', [5, 5, 128, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W3 = tf.get_variable('W3', [5, 5, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W4 = tf.get_variable('W4', [4, 4, 512, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W5 = tf.get_variable('W5', [4, 4, d, 1024], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W6 = tf.get_variable('W6', [5, 5, 1024, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W7 = tf.get_variable('W7', [5, 5, 512, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W8 = tf.get_variable('W8', [5, 5, 256, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W9 = tf.get_variable('W9', [5, 5, 128, 2], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W10 = tf.get_variable('W10', [5, 5, 512, 384], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W11 = tf.get_variable('W11', [5, 5, 384, 320], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W12 = tf.get_variable('W12', [5, 5, 320, 288], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W13 = tf.get_variable('W13', [5, 5, 288, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	W14 = tf.get_variable('W14', [5, 5, 256, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	
	parameters = {'W1' : W1,
					'W2' : W2,,
					'W3' : W3,
					'W4' : W4,
					'W5' : W5,
					'W6' : W6,
					'W7' : W7,
					'W8' : W8,
					'W9' : W9,
					'W10' : W10,
					'W11' : W11,
					'W12' : W12,
					'W13' : W13,
					'W14' : W14}


	return parameters




def encoder(X, d, keep_prob, parameters):
	Z = tf.nn.conv2d(X, parameters['W1'], [1, 2, 2, 1], padding = 'SAME') #kernel = 5*5, strides = 2, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W2'], [1, 2, 2, 1], padding = 'SAME') #kernel = 5*5, strides = 2, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W3'], [1, 2, 2, 1], padding = 'SAME') #kernel = 5*5, strides = 2, o/p channels = 512
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W4'], [1, 2, 2, 1], padding = 'SAME') #kernel = 4*4, strides = 2, o/p channels = 1024
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.contrib.layers.flatten(A) #flattening before fully connected
	A = tf.contrib.layers.fully_connected(A, d) #fully connected to d nodes passed to function
	A = tf.nn.dropout(A, keep_prob) #dropout with keep_prob passed to function

	return A

def decoder(d, parameters): 
	# to o/p 64*64*2
	# i/p = 1*1*d
	inp = tf.reshape(d, [d.shape[0], 1, 1, d.shape[1]])
	A = tf.image.resize_images(inp, [4, 4])  # Bilinear upscaling of the images

	Z = tf.nn.conv2d(A, parameters['W5'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 4*4, strides = 1, o/p channels = 1024
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [8, 8])

	Z = tf.nn.conv2d(A, parameters['W6'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, strides = 1, o/p channels = 512
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [16, 16])

	Z = tf.nn.conv2d(A, parameters['W7'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, strides = 1, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [32, 32])

	Z = tf.nn.conv2d(A, parameters['W8'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, strides = 1, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [64, 64])
	Z = tf.nn.conv2d(A, parameters['W9'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, strides = 1, o/p channels = 2
	A = tf.tanh(Z)
	A = tf.contrib.layers.batch_norm(A)	

	return A


G = tf.placeholder(float32, shape = [None, 28, 28, 512], name= 'Grey level features')

def MDN(G, d, keep_prob, parameters):
	Z = tf.nn.conv2d(G, parameters['W10'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 384
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W11'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 320
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W12'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 288
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W13'], [1, 2, 2, 1], padding = 'SAME')  # kernel = 5*5, stride = 2, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W14'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.contrib.layers.flatten(A)  # flattening before fully connected
	A_means = tf.contrib.layers.fully_connected(A, 4096)  # fully connected to 4096 nodes passed to function for means
	A_means = tf.nn.dropout(A_means, keep_prob)  # dropout with keep_prob passed to function
	A_softmax = tf.contrib.layers.fully_connected(A, 8*d + 8)  # fully connected for softmax - ed activations
	A_softmax = tf.nn.dropout(A_softmax, keep_prob)  # dropout with keep_prob passed to function
	

	return A_means, A_softmax

