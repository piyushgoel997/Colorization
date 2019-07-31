import tensorflow as tf
import numpy as np
import pandas as pd
import math
#from tensorflow.python.framework import ops

nh = 64  # height of input image
nw = 64  # width of input image
nc = 2  # initial  umber of channels
d = 64  #already defined
l_mah = 0.1  #already defined
l_grad = 0.001  #already defined
k = 20  #prinicipal components
in_n_ch = 2 #already defined
nmix = 8 #GMM components
lr = 1e-5 #learning_rate
max_epoch = 5 #epochs
keep_prob = 1.0

#MDN Constants
batch_size = 320 #defined for MDN
mdn_height = 28 #MDN height
mdn_width = 28 #MDN width
mdn_nch = 512 #MDN channels
is_train = True #MDN train
nout = (d + 1) * nmix
constval = 0.001
hidden_size = 64

def params(d):
    parameters = {}
    W1 = tf.get_variable('W1', [5, 5, in_n_ch, 128],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', [5, 5, 128, 256],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable('W3', [5, 5, 256, 512],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable('W4', [4, 4, 512, 1024],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W5 = tf.get_variable('W5', [4, 4, d, 1024],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W6 = tf.get_variable('W6', [5, 5, 1024, 512],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W7 = tf.get_variable('W7', [5, 5, 512, 256],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W8 = tf.get_variable('W8', [5, 5, 256, 128],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W9 = tf.get_variable('W9', [5, 5, 128, 2],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W10 = tf.get_variable('W10', [5, 5, 512, 384],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W11 = tf.get_variable('W11', [5, 5, 384, 320],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W12 = tf.get_variable('W12', [5, 5, 320, 288],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W13 = tf.get_variable('W13', [5, 5, 288, 256],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W14 = tf.get_variable('W14', [5, 5, 256, 128],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W_fc1 = tf.get_variable('W_fc1', [14*14*128, 4096],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W_fc2 = tf.get_variable('W_fc2', [4096, nout],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    b1 = tf.get_variable("b1", [128],dtype = tf.float32, initializer = tf.zeros_initializer())
    b2 = tf.get_variable("b2", [256],dtype = tf.float32, initializer = tf.zeros_initializer())
    b3 = tf.get_variable("b3", [512],dtype = tf.float32, initializer = tf.zeros_initializer())
    b4 = tf.get_variable("b4", [1024],dtype = tf.float32, initializer = tf.zeros_initializer())
    b5 = tf.get_variable("b5", [1024],dtype = tf.float32, initializer = tf.zeros_initializer())
    b6 = tf.get_variable("b6", [512],dtype = tf.float32, initializer = tf.zeros_initializer())
    b7 = tf.get_variable("b7", [256],dtype = tf.float32, initializer = tf.zeros_initializer())
    b8 = tf.get_variable("b8", [128],dtype = tf.float32, initializer = tf.zeros_initializer())
    b9 = tf.get_variable("b9", [in_n_ch],dtype = tf.float32, initializer = tf.zeros_initializer())
    b10 = tf.get_variable("b10", [384],dtype = tf.float32, initializer = tf.zeros_initializer())
    b11 = tf.get_variable("b11", [320],dtype = tf.float32, initializer = tf.zeros_initializer())
    b12 = tf.get_variable("b12", [288],dtype = tf.float32, initializer = tf.zeros_initializer())
    b13 = tf.get_variable("b13", [256],dtype = tf.float32, initializer = tf.zeros_initializer())
    b14 = tf.get_variable("b14", [128],dtype = tf.float32, initializer = tf.zeros_initializer())
    b_fc1 = tf.get_variable('b_fc1', [4096],dtype = tf.float32, initializer = tf.constant_initializer(constval))
    b_fc2 = tf.get_variable('b_fc2', [nout],dtype = tf.float32, initializer = tf.constant_initializer(constval))
    
    parameters = {  'W1' : W1,
                    'W2' : W2,
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
                    'W14' : W14,
                    'W_fc1' : W_fc1,
                    'W_fc2' : W_fc2,
                    'b_fc1' : b_fc1,
                    'b_fc2' : b_fc2,
                    'b1' : b1,
                    'b2' : b2,
                    'b3' : b3,
                    'b4' : b4,
                    'b5' : b4,
                    'b6' : b6,
                    'b7' : b7,
                    'b8' : b8,
                    'b9' : b9,
                    'b10' : b10,
                    'b11' : b11,
                    'b12' : b12,
                    'b13' : b13,
                    'b14' : b14}
    
    return parameters


    def MDN(G, d, keep_prob, parameters):
    Z = tf.nn.conv2d(G, parameters['W10'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 384
    A = tf.nn.relu(Z)
    A = tf.contrib.layers.batch_norm(A, decay = 0.99, scale = True, epsilon = 1e-4, updates_collections = None)

    Z = tf.nn.conv2d(A, parameters['W11'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 320
    A = tf.nn.relu(Z)
    A = tf.contrib.layers.batch_norm(A, decay = 0.99, scale = True, epsilon = 1e-4, updates_collections = None)

    Z = tf.nn.conv2d(A, parameters['W12'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 288
    A = tf.nn.relu(Z)
    A = tf.contrib.layers.batch_norm(A, decay = 0.99, scale = True, epsilon = 1e-4, updates_collections = None)

    Z = tf.nn.conv2d(A, parameters['W13'], [1, 2, 2, 1], padding = 'SAME')  # kernel = 5*5, stride = 2, o/p channels = 256
    A = tf.nn.relu(Z)
    A = tf.contrib.layers.batch_norm(A, decay = 0.99, scale = True, epsilon = 1e-4, updates_collections = None)
    
    Z = tf.nn.conv2d(A, parameters['W14'], [1, 1, 1, 1], padding = 'SAME')  # kernel = 5*5, stride = 1, o/p channels = 128
    A = tf.nn.relu(Z)
    A = tf.contrib.layers.batch_norm(A, decay = 0.99, scale = True, epsilon = 1e-4, updates_collections = None)
    
    
    dropout = tf.nn.dropout(A, keep_prob)
    flatten1 = tf.reshape(dropout, [-1, 14*14*128])
    fc1 = tf.tanh(tf.matmul(flatten1, parameters['W_fc1'] + parameters['b_fc1']))
    
    dropout1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.matmul(dropout1, parameters['W_fc2']) + parameters['b_fc2']
    
    return fc2


def encoder(X, d, keep_prob, parameters):
	Z = tf.nn.conv2d(X, parameters['W1'], [1, 2, 2, 1], padding = 'SAME') #kernel = 5*5, strides = 2, o/p channels = 128
	A = tf.nn.relu(Z + parameters['b1'])
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W2'], [1, 2, 2, 1], padding = 'SAME') #kernel = 5*5, strides = 2, o/p channels = 256
	A = tf.nn.relu(Z+ parameters['b2'])
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W3'], [1, 2, 2, 1], padding = 'SAME') #kernel = 5*5, strides = 2, o/p channels = 512
	A = tf.nn.relu(Z+ parameters['b3'])
	A = tf.contrib.layers.batch_norm(A)

	Z = tf.nn.conv2d(A, parameters['W4'], [1, 2, 2, 1], padding = 'SAME') #kernel = 4*4, strides = 2, o/p channels = 1024
	A = tf.nn.relu(Z+ parameters['b4'])
	A = tf.contrib.layers.batch_norm(A)

	A = tf.contrib.layers.flatten(A) #flattening before fully connected
	A = tf.contrib.layers.fully_connected(A, d) #fully connected to d nodes passed to function
	A = tf.nn.dropout(A, keep_prob) #dropout with keep_prob passed to function

	return A

	def random_mini_batches(X, mini_batch_size):

    np.random.seed(0)        
    m = X.shape[0]  
    
    mini_batches = []
    
    tf.random_shuffle(X)
    
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    #print(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = X[k*mini_batch_size:(k+1)*mini_batch_size, :, :, :]
        
        
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)
    
    
    if m % mini_batch_size != 0:
        mini_batch_X = X[mini_batch_size*math.floor(m / mini_batch_size):m, :, :, :]
        mini_batch = (mini_batch_X)
    
    mini_batches.append(mini_batch)
    
    return mini_batches

    parameters = params(d)

    def get_coeff(out, minibatch_size):
    mu = out[:,:hidden_size * nmix]
    pi = tf.nn.softmax(out[:, hidden_size * nmix:])
    sigma = tf.constant(0.1, shape = [minibatch_size, nmix])
    
    return mu, pi, sigma

    def loss_mdn_train(z, g,minibatch_size , summ = False):
    
    tensor_flat = tf.tile(z, [nmix, 1])
    
    mu, pi, sigma = get_coeff(g, minibatch_size)
    mu.get_shape().as_list()
    pi.get_shape().as_list()
    sigma.get_shape().as_list()
    
    mu_new = tf.reshape(mu, [nmix*minibatch_size, hidden_size])
    sigma_new = tf.reshape(sigma, [nmix*minibatch_size])
    
    op_norm_dist = tf.reshape(tf.div((0.5*tf.reduce_sum(tf.square(tensor_flat-mu_new), reduction_indices=1)), sigma_new), [minibatch_size, nmix])
    op_norm_dist_min = tf.reduce_min(op_norm_dist, reduction_indices=1)
    op_norm_dist_minind = tf.to_int32(tf.argmin(op_norm_dist, 1))
    op_pi_minind_flattened = tf.range(0, minibatch_size)*nmix + op_norm_dist_minind
    op_pi_min = tf.gather(tf.reshape(pi, [-1]), op_pi_minind_flattened)
    if (summ == True):
        gmm_loss = tf.reduce_mean(-tf.log(op_pi_min + 1e-30) + op_norm_dist_min, reduction_indices = 0)
    else:
        gmm_loss = tf.reduce_mean(op_norm_dist_min, reduction_indices = 0)
        
    if(summ == True):
        tf.summary.scalar('gmm_loss', gmm_loss)
        tf.summary.scalar('op_norm_dist_min', tf.reduce_min(op_norm_dist))
        tf.summary.scalar('op_norm_dist_max', tf.reduce_max(op_norm_dist))
        tf.summary.scalar('op_pi_min', tf.reduce_mean(op_pi_min))
    
    return gmm_loss, pi, mu, sigma

def model_mdn(G, z, epochs, d, keep_prob, parameters, learning_rate, minibatch_size):
    
    ip = tf.placeholder(tf.float32, [minibatch_size, mdn_height, mdn_width, mdn_nch])
    op = tf.placeholder(tf.float32, [minibatch_size, hidden_size])
    
    output = MDN(ip, d, keep_prob, parameters)
    
    loss, _, _, _ = loss_mdn_train(op, output,minibatch_size, summ = True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            minibatches = random_mini_batches(G, minibatch_size)
            i = 1
            for m in minibatches:
                _, temp_cost = sess.run([optimizer, loss], feed_dict={ip:m, op:z})
                print("epoch no = " + str(epoch) + ", minibatch = " + str(i) + "cost = " + str(temp_cost))
                
    
    return parameters

    p = model_mdn(G, d, max_epoch, d, keep_prob, parameters, 0.009, 16)
    
