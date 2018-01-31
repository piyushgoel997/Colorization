import numpy as np
import tensorflow as tf


d = 64  # encoding size

nH0 = 64
nW0 = 64
nC0 = 2

X = tf.placeholder("float32", [None, nH0, nW0, nC0])

def encoder(X, parameters, d):
	Z1 = tf.nn.Conv2d(X, parameters["W1"], strides = [1, 2, 2, 1], padding = "VALID") # kernel size = 5 X 5, 128 op channels
	A1 = tf.nn.relu(Z1)
	A1 = tf.contrib.layers.batch_norm(A1) # see vae layer_factory batch_norm fn
	Z2 = tf.nn.Conv2d(A1, parameters["W2"], strides = [1, 2, 2, 1], padding = "VALID") # kernel size = 5 X 5, 256 op channels
	A2 = tf.nn.relu(Z2)
	A2 = tf.contrib.layers.batch_norm(A2)
	Z3 = tf.nn.Conv2d(A2, paramerters["W3"], strides = [1, 2, 2, 1], padding = "VALID") # kernel size = 5 X 5, 512 op channels
	A3 = tf.nn.relu(Z3)
	A3 = tf.contrib.layers.batch_norm(A3)
	Z4 = tf.nn.Conv2d(A2, paramerters["W4"], strides = [1, 2, 2, 1], padding = "VALID") # kernel size = 4 X 4, 1024 op channels
	A4 = tf.nn.relu(Z4)
	A4 = tf.contrib.layers.batch_norm(A4)
	
	A4 = tf.contrib.layers.flatten(A4)
	z = tf.contrib.layers.fully_connected(A4, d)

	return z  # encoding


def decoder(z, parameters):
	