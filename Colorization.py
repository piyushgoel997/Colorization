import tensorflow as tf
import numpy as np
from load_data import load_images
import math
from VAE import train_vae, get_encodings
from MDN import train_mdn
print("importing done")

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

	W10 = tf.get_variable('W10', [5, 5, n_ch, 32], initializer = tf.contrib.layers.xavier_initializer())
	W11 = tf.get_variable('W11', [5, 5, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
	W12 = tf.get_variable('W12', [5, 5, 64, 128], initializer = tf.contrib.layers.xavier_initializer())
	W13 = tf.get_variable('W13', [5, 5, 128, 256], initializer = tf.contrib.layers.xavier_initializer())
	W14 = tf.get_variable('W14', [5, 5, 256, 128], initializer = tf.contrib.layers.xavier_initializer())

	W15 = tf.get_variable('W15', [32*32*128, 256], initializer = tf.contrib.layers.xavier_initializer())
	W16 = tf.get_variable('W16', [256, d], initializer = tf.contrib.layers.xavier_initializer())

	# b1 = tf.get_variable('b1', [128], initializer = tf.contrib.layers.xavier_initializer())
	# b2 = tf.get_variable('b2', [256], initializer = tf.contrib.layers.xavier_initializer())
	# b3 = tf.get_variable('b3', [512], initializer = tf.contrib.layers.xavier_initializer())
	# b4 = tf.get_variable('b4', [1024], initializer = tf.contrib.layers.xavier_initializer())

	# b5 = tf.get_variable('b5', [1024], initializer = tf.contrib.layers.xavier_initializer())
	# b6 = tf.get_variable('b6', [512], initializer = tf.contrib.layers.xavier_initializer())
	# b7 = tf.get_variable('b7', [256], initializer = tf.contrib.layers.xavier_initializer())
	# b8 = tf.get_variable('b8', [128], initializer = tf.contrib.layers.xavier_initializer())
	# b9 = tf.get_variable('b9', [n_ch], initializer = tf.contrib.layers.xavier_initializer())

	# b15 = tf.get_variable('b15', [4096], initializer = tf.contrib.layers.xavier_initializer())
	# b16 =  tf.get_variable('b16', [d], initializer = tf.contrib.layers.xavier_initializer())

	parameters = {'W1' : W1, #'b1' : b1,
				  'W2' : W2, #'b2' : b2,
				  'W3' : W3, #'b3' : b3,
				  'W4' : W4, #'b4' : b4,
				  'W5' : W5, #'b5' : b5,
				  'W6' : W6, #'b6' : b6,
				  'W7' : W7, #'b7' : b7,
				  'W8' : W8, #'b8' : b8,
				  'W9' : W9, #'b9' : b9,
				  'W10' : W10,
				  'W11' : W11,
				  'W12' : W12,
				  'W13' : W13,
				  'W14' : W14,
				  'W15' : W15, #'b15' : b15,
				  'W16' : W16} #'b16' : b16}

	return parameters


# nh = 64  # height of input image
# nw = 64  # width of input image
# n_ch = 3  # initial  number of channels
d = 64  # size of the encoding
temp_path = "C:\My Folder\Programming\Deep Learning\Colorization\\temp\\"
# X = tf.placeholder("float32", shape = [None, 64, 64, 2], name = 'Input image data')


### Train VAE
X, G = load_images("./lfw-deepfunneled")
print("data loaded")
parameters = initialize_parameters(3, d)
print("parameters initialized")
parameters = train_vae(X, parameters, d, 100, 64, temp_path, 0.009, 1.0)
### save encodings
z = get_encodings(X, d, 1.0, parameters)
np.save(temp_path + "encodings", z)

### train MDN
z = np.load(temp_path + "encodings")
train_mdn(G, z, 100, d, 1.0, parameters, 0.009, 16, temp_path)

# G = tf.placeholder('float32', shape = [None, 28, 28, 512], name= 'Grey level features')
# X = tf.random_normal([320, 64, 64,2], mean = 0.04, stddev = 0.97, dtype = tf.float32, seed =0, name = 'X')

##############################################
# X = np.random.randn(320,64,64,3)
# z = get_encodings(X, d, 1.0, parameters)
# print(type(z))
# print(z.shape)
# print(z)
# np.save("C:\My Folder\Programming\Deep Learning\Colorization\\temp\encodings", z)
##############################################

###############  MDN TESTING  ################

# G = np.random.randn(320,64,64,3)
# z = np.random.randn(320, d)
# print("training started")
# train_mdn(G, z, 5, d, 1.0, parameters, 0.001, 16, nmix)


##############################################


# from VAE import encoder, decoder
# e = encoder(X, 64, 1.0,parameters)
# op = decoder(e,parameters)
# print(X.shape)
# print(e.shape)
# print(op.shape)


# parameters = train_vae(X, parameters, d)
