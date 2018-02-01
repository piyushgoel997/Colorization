import numpy as np
import tensorflow as tf


d = 64  # encoding size

nH0 = 64  # height of input to encoder
nW0 = 64  # width of input to encoder
nC0 = 2  # number og channels of input to encoder

X = tf.placeholder("float32", [None, nH0, nW0, nC0])

def encoder(X, parameters, d, keep_prob):
	
	Z1 = tf.nn.Conv2d(X, parameters["W1"], strides = [1, 2, 2, 1], padding = "VALID") + parameters["b1"]  # kernel size = 5 X 5, 128 op channels
	A1 = tf.nn.relu(Z1)
	A1 = tf.contrib.layers.batch_norm(A1) # see vae layer_factory batch_norm fn
	
	Z2 = tf.nn.Conv2d(A1, parameters["W2"], strides = [1, 2, 2, 1], padding = "VALID") + parameters["b2"]  # kernel size = 5 X 5, 256 op channels
	A2 = tf.nn.relu(Z2)
	A2 = tf.contrib.layers.batch_norm(A2)
	
	Z3 = tf.nn.Conv2d(A2, paramerters["W3"], strides = [1, 2, 2, 1], padding = "VALID") + parameters["b3"]  # kernel size = 5 X 5, 512 op channels
	A3 = tf.nn.relu(Z3)
	A3 = tf.contrib.layers.batch_norm(A3)
	
	Z4 = tf.nn.Conv2d(A2, paramerters["W4"], strides = [1, 2, 2, 1], padding = "VALID") + parameters["b4"]  # kernel size = 4 X 4, 1024 op channels
	A4 = tf.nn.relu(Z4)
	A4 = tf.contrib.layers.batch_norm(A4)
	
	A4 = tf.contrib.layers.flatten(A4)
	z = tf.contrib.layers.fully_connected(A4, d)
	z = tf.nn.droput(z, keep_prob)  # dropout regularization

	return z  # encoding


def decoder(z, parameters):

	# consider z with image of size 1 X 1 with 'd' channels
	inp = tf.reshape(z, [z.shape[0], 1, 1, z.shape[1]])
	A5 = tf.image.resize_image(inp, [4, 4])  # Bilinear upscaling of the images

	Z6 = tf.nn.Conv2d(A5, parameters["W6"], strides = [1, 1, 1, 1], padding = "VALID") + parameters["b6"]  # kernel size = 4 X 4, 1024 op channels
	A6 = tf.nn.relu(Z6)
	A6 = tf.contrib.layers.batch_norm(A6)
	
	A7 = tf.image.resize_image(A6, [8, 8])
	
	Z8 = tf.nn.Conv2d(A7, parameters["W8"], strides = [1, 1, 1, 1], padding = "VALID") + parameters["b8"]  # kernel size = 5 X 5, 512 op channels
	A8 = tf.nn.relu(Z8)
	A8 = tf.contrib.layers.batch_norm(A8)
	
	A9 = tf.image.resize_image(A8, [16, 16])
	
	Z10 = tf.nn.Conv2d(A9, parameters["W10"], strides = [1, 1, 1, 1], padding = "VALID") + parameters["b10"]  # kernel size = 5 X 5, 256 op channels
	A10 = tf.nn.relu(Z10)
	A10 = tf.contrib.layers.batch_norm(A10)
	
	A11 = tf.image.resize_image(A10, [32, 32])
	
	Z12 = tf.nn.Conv2d(A11, parameters["W12"], strides = [1, 1, 1, 1], padding = "VALID") + parameters["b12"]  # kernel size = 5 X 5, 128 op channels
	A12 = tf.nn.relu(Z12)
	A12 = tf.contrib.layers.batch_norm(A12)
	
	A13 = tf.image.resize_image(A12, [64, 64])
	
	Z14 = tf.nn.Conv2d(A13, parameters["W14"], strides = [1, 1, 1, 1], padding = "valid") + parameters["b14"]  # kernel size = 5 X 5, 2 op channels
	A14 = tf.nn.tanh(Z14)

	return A14