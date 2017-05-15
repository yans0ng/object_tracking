from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Layer, Flatten
from keras import backend as K
from keras.layers import Input, Dense
from keras import layers, optimizers
from keras.layers import Add, add, Merge, Concatenate, multiply
from keras.layers.core import Lambda, Permute, Reshape
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import sys
import tensorflow as tf
import numpy as np
from tensorflow import Session
from keras.layers.wrappers import TimeDistributed
from ipykernel import kernelapp as app
import pydot
import graphviz
from keras import regularizers
from matplotlib import pyplot as plt
#%matplotlib inline
import cv2
import os
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import os, re, glob, cv2
import cPickle, gzip

# Configuration
F = 32
H = 416
W = 448
C = 3
batch_size_mnist = 16
batch_size_coil = 4
epochs = 2
num_hw_layers = 5
model_save_path = '{epoch:02d}-{val_loss:.2f}.hdf5'
log_save_path = 'log'
training_times = 1
split_ratio = 0.2
input_dim = (3,None,None)
def createBasicBlock(input_dim, filters = 32, kernel_size = 3, strides = 1, act_func = 'relu'):
    model = Sequential()
    model.add(BatchNormalization(axis = 1, input_shape = input_dim))
    model.add(Conv2D(filters = filters, kernel_size = 3, strides = 1, 
                     input_shape = input_dim, padding = 'same', 
                     data_format = 'channels_first'
                    )
             )
    model.add(Activation(act_func))
    return model

def highway_helper(x):
    ## Should only be called by createHighwayBlock
    from keras.layers import multiply
    input_tensor, conv_tensor, trans_tensor = x
    return multiply([conv_tensor, trans_tensor]) + multiply([input_tensor, 1-trans_tensor])

def matchCost(input):
    left = input[0]
    right = input[1]
    #left_result = Flatten()(left)
    #right_result = Flatten()(right)
    #result = Activation('tanh')(K.dot(left_result, right_result))
    output = K.sqrt(K.mean(K.square(left - right), axis = [1,2,3]))
    output = K.expand_dims(output, axis = 1)
    return output

def createHighwayBlock(input1, input2, input_dim, filters = 32, kernel_size = 3, stride = 1):
    bb = createBasicBlock(input_dim, filters, kernel_size, stride)
    trans = createBasicBlock(input_dim, filters, kernel_size, stride, act_func = 'sigmoid')    
    bb1 = bb(input1)
    bb2 = bb(input2)
    trans1 = trans(input1)
    trans2 = trans(input2)
    hb = Lambda(highway_helper)
    hb1 = hb([input1, bb1, trans1])
    hb2 = hb([input2, bb2, trans2])    
    return hb1, hb2

def createSiameseNetwork(input1, input2, input_dim, filters = 32, kernel_size = 3, stride = 1, num_hw_layers = 3):
    basic = Conv2D(filters = filters, kernel_size = kernel_size, 
                   strides = stride, input_shape = input_dim,
                   data_format = 'channels_first'
                  )
    hb1 = basic(input1)
    hb2 = basic(input2)
    for i in range(num_hw_layers):
        input_shape = hb1.get_shape().as_list()
        hb1, hb2 = createHighwayBlock(hb1, hb2, input_shape[1:], filters, kernel_size, stride)
    return Lambda(matchCost)([hb1, hb2])

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def logit_loss(y_true, y_pred):
	return K.mean(K.log(1+K.exp(-y_true*y_pred)))
def load_data(path):
	data = np.load(path)
	left = data['left']
	right = data['right']
	label = data['label']
	return left, right, label

def genCallBacks(model_save_path, log_save_path):
        callback_tb = TensorBoard(log_dir=log_save_path, histogram_freq=0, write_graph=True, write_images=True)
        callback_mc = ModelCheckpoint(model_save_path, verbose = 1, save_best_only = True, save_weights_only = True, period = 1)
        callback_es = EarlyStopping(min_delta = 0, patience = 2, verbose = 1)
        return [callback_tb, callback_mc, callback_es]


if  __name__ == '__main__':
	a = Input((3, None, None))
	b = Input((3, None, None))
	y_pred = createSiameseNetwork(a, b,input_dim, num_hw_layers = num_hw_layers)
	model = Model([a,b], y_pred)
	if int(sys.argv[1]) == 1:
		model.load_weights(sys.argv[2])
	model.compile(optimizer='rmsprop', loss=contrastive_loss, metrics = ['binary_accuracy'])
	data_list = ['coil_data.npz', 'mnist_data.npz']
	callbacks = genCallBacks(model_save_path, log_save_path)
	for i in range(training_times):
		if i%2 == 0:
			batch_size = batch_size_coil
		else:
			batch_size = batch_size_mnist
		for data in data_list:
			left, right, label = load_data(data)
			label[label == -1] = 0
			model.fit([left, right], label, batch_size = batch_size, epochs = epochs, validation_split = split_ratio, callbacks = callbacks)
		print "training with ", data, ' is complete'
	#model.save('siamese_model.h5')
	print "training complete"
