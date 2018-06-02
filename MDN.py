import tensorflow as tf

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

	Z = tf.matmul(A, parameters['W15']) + parameters['b15']
	A = tf.tanh(Z)

	A = tf.nn.dropout(A, keep_prob)
	Z = tf.matmul(A, parameters['W16']) + parameters['b16']

	return Z
	
	# A = tf.contrib.layers.flatten(A)  # flattening before fully connected
	# A_means = tf.contrib.layers.fully_connected(A, 4096)  # fully connected to 4096 nodes passed to function for means
	# A_means = tf.nn.dropout(A_means, keep_prob)  # dropout with keep_prob passed to function
	# A_softmax = tf.contrib.layers.fully_connected(A, 8*d + 8)  # fully connected for softmax - ed activations
	# A_softmax = tf.nn.dropout(A_softmax, keep_prob)  # dropout with keep_prob passed to function
	#
	#
	# return A_means, A_softmax
