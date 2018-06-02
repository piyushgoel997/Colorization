import tensorflow as tf
import numpy as np
from load_data import load_images
import math
from VAE import train_vae
from parameters import initialize_parameters


# nh = 64  # height of input image
# nw = 64  # width of input image
# n_ch = 3  # initial  number of channels
d = 64  # size of the encoding

# X = tf.placeholder("float32", shape = [None, 64, 64, 2], name = 'Input image data')


# G = tf.placeholder('float32', shape = [None, 28, 28, 512], name= 'Grey level features')
# X = tf.random_normal([320, 64, 64,2], mean = 0.04, stddev = 0.97, dtype = tf.float32, seed =0, name = 'X')
X, G = load_images("./lfw-deepfunneled")
print("data loaded")
parameters = initialize_parameters(3, d)
parameters = train_vae(X, parameters, d)

# from VAE import encoder, decoder
# e = encoder(X, 64, 1.0,parameters)
# op = decoder(e,parameters)
# print(X.shape)
# print(e.shape)
# print(op.shape)


# parameters = train_vae(X, parameters, d)
