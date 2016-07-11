# -*- coding: utf-8 -*-
#http://stackoverflow.com/questions/31972990/increasing-speed-of-a-pure-numpy-scipy-convolutional-neural-network-implementati
import numpy as np
import scipy.signal
#from time import time

def max_pool(x):
    """Return maximum in groups of 2x2 for a N,h,w image"""
    N,h,w = x.shape
    return np.amax([x[:,(i>>1)&1::2,i&1::2] for i in range(4)],axis=0)

def conv_layer(params,x):
    """Applies a convolutional layer (W,b) followed by 2*2 pool followed by RelU on x"""
    W,biases = params
    print "weight shape: ", W.shape
    print 'Weight: ' , W
    print "biases shape: ", biases.shape
    print 'Biases: ' , biases
    num_in = W.shape[1]
    print 'w.shape(1): ' , num_in
    A = []
    for f,bias in enumerate(biases):
        conv_out = np.sum([scipy.signal.convolve2d(x[i],W[f][i],mode='valid') for i in range(num_in)], axis=0)
        A.append(conv_out + bias)
    x = np.array(A)
    print "convFeat shape: ", x.shape
    print 'convFeat: ' , x
    x = max_pool(x)
    print "poolFeat shape: ",x.shape
    print 'poolFeat: ' , x
    return np.maximum(x,0)

W = np.random.randn(3,3,3,3).astype(np.float32)
b = np.random.randn(3).astype(np.float32)
I = np.random.randn(3,6,6).astype(np.float32)


O = conv_layer((W,b),I)