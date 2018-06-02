import tensorflow as tf
from load_data import mini_batches
from Colorization import save_parameters

def encoder(X, d, keep_prob, parameters):
	Z = tf.nn.conv2d(X, parameters['W1'], [1, 2, 2, 1], padding = 'SAME') + parameters['b1'] #kernel = 5*5, strides = 2, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W2'], [1, 2, 2, 1], padding = 'SAME') + parameters['b2'] #kernel = 5*5, strides = 2, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W3'], [1, 2, 2, 1], padding = 'SAME') + parameters['b3'] #kernel = 5*5, strides = 2, o/p channels = 512
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W4'], [1, 2, 2, 1], padding = 'SAME') + parameters['b4'] #kernel = 4*4, strides = 2, o/p channels = 1024
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

	Z = tf.nn.conv2d(A, parameters['W5'], [1, 1, 1, 1], padding = 'SAME') + parameters['b5']  # kernel = 4*4, strides = 1, o/p channels = 1024
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [8, 8])

	Z = tf.nn.conv2d(A, parameters['W6'], [1, 1, 1, 1], padding = 'SAME') + parameters['b6']  # kernel = 5*5, strides = 1, o/p channels = 512
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [16, 16])

	Z = tf.nn.conv2d(A, parameters['W7'], [1, 1, 1, 1], padding = 'SAME') + parameters['b7']  # kernel = 5*5, strides = 1, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [32, 32])

	Z = tf.nn.conv2d(A, parameters['W8'], [1, 1, 1, 1], padding = 'SAME') + parameters['b8']  # kernel = 5*5, strides = 1, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [64, 64])
	Z = tf.nn.conv2d(A, parameters['W9'], [1, 1, 1, 1], padding = 'SAME') + parameters['b9']  # kernel = 5*5, strides = 1, o/p channels = 2
	A = tf.tanh(Z)
	A = tf.contrib.layers.batch_norm(A)

	return A

def train_vae(X_train, parameters, d, num_epochs = 100, learning_rate = 0.009, keep_prob = 1.0, minibatch_size = 16):
	X = tf.placeholder(tf.float32, [minibatch_size] + list(X_train.shape[1:]))
	Z = decoder(encoder(X, d, keep_prob, parameters), parameters)
	cost = compute_cost(X, Z)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	with  tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatches = mini_batches(X_train, minibatch_size, shuffle=True)
			for minibatch in minibatches:
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch})
				print(temp_cost)
			save_parameters(parameters, True)
    # train_accuracy = accuracy.eval({X: X_train, Y: X_train})
    # print("Train Accuracy:", train_accuracy)
    # return train_accuracy, parameters
	return parameters

def compute_cost(inp, op):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = op, labels = inp))
	return cost

# import numpy as np
# a = np.array([1.0,2.0,3.0,4.0])
# b = np.array([1000.0,2.0,3.0,4.0])
# print(tf.Session().run(compute_cost(a, b)))
