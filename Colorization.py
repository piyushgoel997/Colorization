import tensorflow as tf
import numpy as numpy
import pandas as pd

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

def compute_cost(Z, Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
	return cost

def create_placeholders(m, n_H0, n_W0, n_C0):
	X = tf.placeholder(tf.float32, [m, n_H0, n_W0, n_C0])
	Y = tf.placeholder(tf.float32, [m,n_H0, n_W0, n_C0])
	return X, Y

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	np.random.seed(seed)
    m = X.get_shape().as_list()[0]
    mini_batches = []
    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def model_encoder(X_train,parameters, num_epochs = 100,learning_rate = 0.009, d = 64, keep_prob = 1.0, minibatch_size = 64):
    tf.set_random_seed(1)
    seed = 3
    m, n_H0, n_W0, n_C0 = X_train.get_shape().as_list()
    epoch = 0
    X, Y = create_placeholders(m, n_H0, n_W0, n_C0)
    Z1 = encoder(X, d, keep_prob, parameters)
    Z2 = decoder(Z1, parameters)
    cost = compute_cost(Z2, X_train)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with  tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, X_train, minibatch_size, seed)
            for minibatch in minibatches:
				# Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
    train_accuracy = accuracy.eval({X: X_train, Y: X_train})
    print("Train Accuracy:", train_accuracy)
    return train_accuracy, parameters

nh = 64  # height of input image
nw = 64  # width of input image
nc = 2  # initial  umber of channels
d = 64

X = tf.placeholder("float32", shape = [None, 64, 64, 2], name = 'Input image data')


G = tf.placeholder('float32', shape = [None, 28, 28, 512], name= 'Grey level features')
X = tf.random_normal([320, 64, 64,2], mean = 0.04, stddev = 0.97, dtype = tf.float32, seed =0, name = 'X')
parameters = params(d)
_, _, parameters = model_encoder(X, parameters)
