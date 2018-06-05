import tensorflow as tf
import time
from load_data import mini_batches
import numpy as np

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

	A = tf.nn.dropout(A, keep_prob)
	A = tf.contrib.layers.flatten(A)

	Z = tf.matmul(A, parameters['W15']) #+ parameters['b15']
	A = tf.tanh(Z)

	A = tf.nn.dropout(A, keep_prob)
	Z = tf.matmul(A, parameters['W16']) #+ parameters['b16']

	return Z
	
	# A = tf.contrib.layers.flatten(A)  # flattening before fully connected
	# A_means = tf.contrib.layers.fully_connected(A, 4096)  # fully connected to 4096 nodes passed to function for means
	# A_means = tf.nn.dropout(A_means, keep_prob)  # dropout with keep_prob passed to function
	# A_softmax = tf.contrib.layers.fully_connected(A, 8*d + 8)  # fully connected for softmax - ed activations
	# A_softmax = tf.nn.dropout(A_softmax, keep_prob)  # dropout with keep_prob passed to function
	#
	#
	# return A_means, A_softmax

def train_mdn(G, z, epochs, d, keep_prob, parameters, learning_rate, minibatch_size, temp_path):
	inp = tf.placeholder(tf.float32, [minibatch_size] + list(G.shape)[1:])
	expected_op = tf.placeholder(tf.float32, [minibatch_size, d])
	loss = compute_loss(MDN(inp, d, keep_prob, parameters), expected_op)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(epochs):
			ep_st = time.time()
			G, z = shuffle_in_unison(G, z)
			g_m = mini_batches(G, minibatch_size, shuffle = False)
			z_m = mini_batches(z, minibatch_size, shuffle = False)
			for i in range(len(g_m)):
				mg = g_m[i]
				mz = z_m[i]
				mb_st = time.time()
				_, temp_cost = sess.run([optimizer, loss], feed_dict={inp:mg, expected_op:mz})
				mb_time = time.time() - mb_st
				print("epoch no = " + str(epoch) + ", minibatch = " + str(i) + "/" + str(len(g_m)) + ", loss = " + str(temp_cost) + " in " + str(mb_time) + "secs")
			ep_time = time.time() - ep_st
			print("epoch " + str(epoch) + " completed in " + str(ep_time) + "secs")
			save_path = saver.save(sess, temp_path + "mdn_model.ckpt") 
			print("Model saved in path: " + str(save_path))
	return parameters

def shuffle_in_unison(a, b):
	p = np.array(range(a.shape[0]))
	np.random.shuffle(p)
	return a[p], b[p]

# def get_mixture_coeff(out_fc, d, nmix, minibatch_size):
# 	out_mu = out_fc[..., :d*nmix]
# 	out_pi = tf.nn.softmax(out_fc[..., d*nmix:])
# 	out_sigma = tf.constant(.1, shape=[minibatch_size, nmix])
# 	return out_pi, out_mu, out_sigma

# def compute_gmm_loss(z, mdn_op, d, nmix, minibatch_size):
	
# 	z_flat = tf.tile(z, [nmix, 1])

# 	pi, mu, sigma = get_mixture_coeff(mdn_op, d, nmix, minibatch_size)

# 	mu_flat = tf.reshape(mu, [nmix*minibatch_size, d])
# 	sigma_flat = tf.reshape(sigma, [nmix*minibatch_size])

# 	#N(t|x, mu, sigma): minibatch_size x nmix
# 	op_norm_dist = tf.reshape(tf.div((.5*tf.reduce_sum(tf.square(z_flat-mu_flat), reduction_indices=1)), sigma_flat), [minibatch_size, nmix])
# 	op_norm_dist_min = tf.reduce_min(op_norm_dist, reduction_indices=1)
# 	op_norm_dist_minind = tf.to_int32(tf.argmin(op_norm_dist, 1))
# 	op_pi_minind_flattened = tf.range(0, minibatch_size)*nmix + op_norm_dist_minind
# 	op_pi_min = tf.gather(tf.reshape(pi, [-1]), op_pi_minind_flattened)

# 	return tf.reduce_mean(-tf.log(op_pi_min+1e-30) + op_norm_dist_min, reduction_indices=0)

def compute_loss(inp, op):
	cost = tf.nn.l2_loss(inp-op)
	return cost