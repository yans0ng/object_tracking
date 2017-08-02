from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import layers, optimizers
from keras.layers import Input, Activation, add, multiply
from keras.layers.core import Lambda

from keras import regularizers
import numpy as np
from ipykernel import kernelapp as app

# Configuration
num_hw_layers = 16 
input_dim = (3,64, 64)
input1 = Input((3, 64, 64))
input2 = Input((3,64, 64))
filters = 48 
stride = 1
kernel_size = 3

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

## Should only be called by createHighwayBlock
def highway_helper(x):
    
    input_tensor, conv_tensor, trans_tensor = x
    return multiply([conv_tensor, trans_tensor]) + multiply([input_tensor, 1-trans_tensor])


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

def createSiameseNetwork(output): 
    if output == 'e':
	outFunc = euclideanDistance
    elif output == 'c':
	outFunc = cosineSimilarity
    else:
	raise Exception ('output function {0} is not supported, only support e for euclideanDistance and c for cosineSimilarity'.format(output))
    pad1 = ZeroPadding2D(input_shape = input_dim, data_format = 'channels_first')
    pad2 = ZeroPadding2D(data_format = 'channels_first')
    basic = Conv2D(filters = filters, kernel_size = 5, 
                   strides = 2, input_shape = input_dim,
                   data_format = 'channels_first', padding = 'valid'
                  )
    basic2 = Conv2D(filters = filters, kernel_size = 5,
                   strides = 2, input_shape = input_dim,
                   data_format = 'channels_first', padding = 'valid'
                  )
    hb1 = pad2(basic2(basic(pad1(input1))))
    hb2 = pad2(basic2(basic(pad1(input2))))
    for i in range(num_hw_layers):
        input_shape = hb1.get_shape().as_list()
        hb1, hb2 = createHighwayBlock(hb1, hb2, input_shape[1:], filters, kernel_size, stride)
    deconv1 = Conv2DTranspose(filters = filters/2, kernel_size = kernel_size, strides = stride, data_format = 'channels_first', padding = 'valid')
    deconv2 = Conv2DTranspose(filters = 1, kernel_size = kernel_size, strides = stride, data_format = 'channels_first', padding = 'valid')
    hb1 = deconv1(hb1)
    #print "output dim = ", deconv2.compute_output_shape(hb1.get_shape())
    hb1 = deconv2(hb1)
    hb2 = deconv1(hb2)
    hb2 = deconv2(hb2)
    y = Lambda(outFunc)([hb1, hb2])
    model = Model([input1, input2], y)
    return model

def euclideanDistance(input):
    left = input[0]
    right = input[1]
    output = K.sqrt(K.sum(K.square(left - right), axis = [2,3]))
    return output
    
def cosineSimilarity(input):
	left = input[0]
	right = input[1]
	left_len = K.sqrt(K.sum(K.sum(K.square(left), axis = 2), axis = 2))
        right_len = K.sqrt(K.sum(K.sum(K.square(right), axis = 2), axis = 2))
	inner_prod = K.sum(K.sum(multiply([left, right]),axis = 2),axis = 2)
	return inner_prod / (left_len*right_len)

