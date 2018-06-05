import os
import cv2
import numpy as np
import math

def mini_batches(X, mini_batch_size=16, shuffle=False):
    if shuffle:
        np.random.shuffle(X)
    m = X.shape[0]
    mini_batches = []
    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batches.append(X[k * mini_batch_size:(k + 1) * mini_batch_size])

    # if m % mini_batch_size != 0:
    #     mini_batches.append(X[num_complete_minibatches*mini_batch_size:])
        
    return mini_batches


def load_images(root_path, image_shape = (64, 64, 3)):
    paths = []
    for root, _, files in os.walk(root_path):
        for f in files:
            paths.append(str(root)+"/"+str(f))

    X = np.zeros((len(paths), *image_shape), dtype='float32')
    G = np.zeros((len(paths), *image_shape[:2]), dtype='float32')
    i = 0
    for p in paths:
        img = cv2.resize(cv2.imread(p), image_shape[:2])
        X[i] = img
        G[i] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        i += 1
        # if i % 100 == 0:
        #     print(i)
    
    # print(X.shape)
    # print(G.shape)
    # mini_batches(X)
    return X, G
    


##### Testing code
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("path")
# args = parser.parse_args()
# path = args.path
# load_images(path)


