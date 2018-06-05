import tensorflow as tf
from load_data import mini_batches
import time

def encoder(X, d, keep_prob, parameters):
	Z = tf.nn.conv2d(X, parameters['W1'], [1, 2, 2, 1], padding = 'SAME') #+ parameters['b1'] #kernel = 5*5, strides = 2, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W2'], [1, 2, 2, 1], padding = 'SAME') #+ parameters['b2'] #kernel = 5*5, strides = 2, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W3'], [1, 2, 2, 1], padding = 'SAME') #+ parameters['b3'] #kernel = 5*5, strides = 2, o/p channels = 512
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W4'], [1, 2, 2, 1], padding = 'SAME') #+ parameters['b4'] #kernel = 4*4, strides = 2, o/p channels = 1024
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

	Z = tf.nn.conv2d(A, parameters['W5'], [1, 1, 1, 1], padding = 'SAME') #+ parameters['b5']  # kernel = 4*4, strides = 1, o/p channels = 1024
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [8, 8])

	Z = tf.nn.conv2d(A, parameters['W6'], [1, 1, 1, 1], padding = 'SAME') #+ parameters['b6']  # kernel = 5*5, strides = 1, o/p channels = 512
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [16, 16])

	Z = tf.nn.conv2d(A, parameters['W7'], [1, 1, 1, 1], padding = 'SAME') #+ parameters['b7']  # kernel = 5*5, strides = 1, o/p channels = 256
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [32, 32])

	Z = tf.nn.conv2d(A, parameters['W8'], [1, 1, 1, 1], padding = 'SAME') #+ parameters['b8']  # kernel = 5*5, strides = 1, o/p channels = 128
	A = tf.nn.relu(Z)
	A = tf.contrib.layers.batch_norm(A)

	A = tf.image.resize_images(A, [64, 64])
	Z = tf.nn.conv2d(A, parameters['W9'], [1, 1, 1, 1], padding = 'SAME') #+ parameters['b9']  # kernel = 5*5, strides = 1, o/p channels = 2
	A = tf.tanh(Z)
	A = tf.contrib.layers.batch_norm(A)

	return A

def train_vae(X_train, parameters, d, num_epochs, minibatch_size, temp_path, learning_rate, keep_prob):
	X = tf.placeholder(tf.float32, [minibatch_size] + list(X_train.shape[1:]))
	Z = decoder(encoder(X, d, keep_prob, parameters), parameters)
	cost = compute_cost(X, Z)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with  tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			ep_st = time.time()
			minibatches = mini_batches(X_train, minibatch_size, shuffle=True)
			i = 1
			for minibatch in minibatches:
				mb_st = time.time()
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch})
				mb_time = time.time() - mb_st
				print("epoch no = " + str(epoch) + ", minibatch = " + str(i) + "/" + str(len(minibatches)) + ", loss = " + str(temp_cost) + " in " + str(mb_time) + "secs")
				i += 1	
			ep_time = time.time() - ep_st
			print("epoch " + str(epoch) + " completed in " + str(ep_time) + "secs")
			save_path = saver.save(sess, temp_path + "vae_model.ckpt") 
			# saver.restore(sess, "model.ckpt") -> restore the saved model
			print("Model saved in path: " + str(save_path))
    # train_accuracy = accuracy.eval({X: X_train, Y: X_train})
    # print("Train Accuracy:", train_accuracy)
    # return train_accuracy, parameters
	return parameters

def compute_cost(inp, op):
	cost = tf.nn.l2_loss(inp-op)
	return cost

def get_encodings(C, d, keep_prob, parameters):
	X = tf.placeholder(tf.float32, list(C.shape))
	enc = encoder(X, d, keep_prob, parameters)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		encodings = sess.run(enc, feed_dict={X:C})
	return encodings

# import numpy as np
# a = np.array([1.0,2.0,3.0,4.0])
# b = np.array([1000.0,2.0,3.0,4.0])
# print(tf.Session().run(compute_cost(a, b)))
