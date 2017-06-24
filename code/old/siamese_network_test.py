from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import layers, optimizers
from keras.layers import Input, Activation, add, multiply
from keras.layers.core import Lambda, Permute, Reshape
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
import numpy as np
from ipykernel import kernelapp as app

# Configuration
stride = 1
kernel_size = 3
w = 64
h = 64
input1 = Input((3, w, h))
input_dim =  ((3, None, None)) 

num_hw_layers = int(np.log(w/4)/np.log(2))
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


def createHighwayBlock(input1, input2, filters = 32, kernel_size = 3, stride = 1):
    bb = createBasicBlock((32, None, None), filters, kernel_size, stride)
    trans = createBasicBlock((32, None, None), filters, kernel_size, stride, act_func = 'sigmoid')    
    hh = Lambda(highway_helper)
    bb1 = bb(input1)
    trans1 = trans(input1)
    hb1 = hh([input1, bb1, trans1])

    bb2 = bb(input2)
    trans2 = trans(input2)
    hb2 = hh([input2, bb2, trans2])
    return hb1, hb2

def euclideanDistance(input):
    # target dimension = (batch, 1, height, width)
    # candidates dimension = (batch, num_patches, height, width)
    target = input[0]
    candidates = input[1]
    target = K.squeeze(target, 2)
    target = K.squeeze(target, 2)

    candidates = Permute((2,1,3))(candidates)
    #print "hello"
    output = K.dot(target, candidates)
    output = K.squeeze(output, 1)
    return output
    
def cosineSimilarity(input):
    left = input[0]
    right = input[1]
    left_len = K.sqrt(K.sum(K.sum(K.square(left), axis = 2), axis = 2))
    right_len = K.sqrt(K.sum(K.sum(K.square(right), axis = 2), axis = 2))
    inner_prod = K.sum(K.sum(multiply([left, right]),axis = 2),axis = 2)
    return inner_prod / (left_len*right_len)

def createSiameseNetwork(output, filters = 32, mode = 'train'): 
    if mode == 'train':
        input2 = Input((3,64, 64))
    else:
        input2 = Input(input_dim)
    if output == 'e':
    outFunc = euclideanDistance
    elif output == 'c':
    outFunc = cosineSimilarity
    else:
    raise Exception ('output function {0} is not supported, only support e for euclideanDistance and c for cosineSimilarity'.format(output))
    conv = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', data_format = 'channels_first')
    hb1 = conv(input1)
    hb2 = conv(input2)
    for i in range(num_hw_layers):
        hb1, hb2 = createHighwayBlock(hb1, hb2, filters, kernel_size, stride)
        conv = Conv2D(filters = filters, kernel_size = 3, strides = 2, padding = 'same', data_format = 'channels_first')
        hb1 = conv(hb1)
        hb2 = conv(hb2)
    conv = Conv2D(filters = filters, kernel_size = 4, strides = 1, padding = 'valid', data_format = 'channels_first')
    hb1 = conv(hb1)
    hb2 = conv(hb2)
    y = Lambda(outFunc)([hb1, hb2])
    model = Model([input1, input2], y)
    return model
