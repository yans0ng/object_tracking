from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Activation, add, multiply, dot
from keras.layers.core import Lambda, Reshape
from ipykernel import kernelapp as app
import tensorflow as tf
# Configuration
stride = 1
kernel_size = 3
num_hw_layers = 4
def createBasicBlock(input_dim, filters = 32, kernel_size = 3, strides = 1, act_func = 'tanh', num_conv = 1):
	model = Sequential()
    	for i in range(num_conv):
		model.add(Conv2D(filters = filters, kernel_size = 3, strides = 1, 
        	             	input_shape = input_dim, padding = 'valid', 
                	     	data_format = 'channels_first'
                   		 )
             		)
		model.add(BatchNormalization(axis = 1))
    		model.add(Activation(act_func))
	return model

def highway_helper(x):
    input_tensor, output_tensor, trans_tensor = x
    return multiply([output_tensor, trans_tensor]) + multiply([input_tensor, 1 - trans_tensor])

def createHighwayBlock(inputs, filters = 32, kernel_size = 3, stride = 1, act_func = 'relu', use_resnet = True):
    input1, input2 = inputs
    input_dim = K.int_shape(input1)[1:]
    bb = createBasicBlock(input_dim, filters, kernel_size, stride, act_func =  act_func)
    bb1 = bb(input1)
    bb2 = bb(input2)
    res = createBasicBlock(input_dim, filters, kernel_size, stride, act_func = act_func)
    res1 = res(input1)
    res2 = res(input2)
    if use_resnet:
    	hb1 = add([res1, bb1])
	hb2 = add([res2, bb2])
    else:
    	trans = createBasicBlock(input_dim, filters, kernel_size, stride, act_func = 'sigmoid')    
    	trans1 = trans(input1)
    	trans2 = trans(input2)
    	hh = Lambda(highway_helper)
    	hb1 = hh([res1, bb1, trans1])
    	hb2 = hh([res2, bb2, trans2])
    return [hb1, hb2]

'''
def euclideanDistance2(input):
    target = input[0]
    candidates = input[1]
    b,f,h,w = K.int_shape(candidates)
    output = K.sqrt(K.sum(K.square(target - candidates), axis = [1,2,3]))
    output = Reshape((1,))(output)
    return output
'''

def euclideanDistance(inputs):
	candidates, target = inputs
    	b,c,h,w = tf.unstack(tf.shape(target))
    	target2 = tf.transpose(target,(0,2,3,1))
    	ones = tf.ones((b,h,w,c), dtype = tf.float32)	
    	ct_dot,a = K.map_fn(
    		lambda inputs:
    	    		[K.conv2d(
    	    		        K.expand_dims(inputs[0], 0),
    	    	        	K.expand_dims(inputs[1], 3),
    	    	        	strides=(1,1),
    	    	        	padding = 'valid',
       	   	        	data_format = 'channels_first',
       	   	  		)
       	 		,1],
       	 	elems = [candidates, target2]
    	)
    	ct_dot = ct_dot[:,0,0]
    	square_candidates = K.square(candidates)
    	square_sum_can,a = K.map_fn(
        	lambda inputs:
        		[K.conv2d(
        	        	K.expand_dims(inputs[0], 0),
        	        	K.expand_dims(inputs[1], 3),
        	        	strides = (1,1),
        	        	padding = 'valid',
        	        	data_format = 'channels_first' ),
        	 	1],
       		 elems = [square_candidates, ones]
    	)
    	square_sum_can = (square_sum_can[:,0,0,:,:])
    	square_sum_target = K.sum(K.sum(K.sum(K.square(target2), axis = 1), axis = 1), axis = 1)
    	square_sum_target = K.expand_dims(square_sum_target, 1)
    	square_sum_target = K.expand_dims(square_sum_target, 1)
   	square_sum_target = tf.tile(square_sum_target, multiples = [1,tf.shape(square_sum_can)[1],tf.shape(square_sum_can)[2]])
    	output = K.sqrt(square_sum_target + square_sum_can - 2*ct_dot)
    	output = tf.reshape(output, (-1,tf.shape(output)[1]*tf.shape(output)[2]))
    	return output

def cosineSimilarity(inputs, mode = 'train'):
    target, candidates = inputs
    b,c,h,w = tf.unstack(tf.shape(target))
    target2 = tf.transpose(target,(0,2,3,1))
    ones = tf.ones((b,h,w,c), dtype = tf.float32)

    ct_dot,a = K.map_fn(
        lambda inputs:
        [K.conv2d(
                K.expand_dims(inputs[0], 0),
                K.expand_dims(inputs[1], 3),
                strides=(1,1),
                padding = 'valid',
                data_format = 'channels_first',
                )
        ,1],
        elems = [candidates, target2]
    )
    ct_dot = ct_dot[:,0,0]
    square_candidates = K.square(candidates)
    square_sum_can,a = K.map_fn(
        lambda inputs:
        [K.conv2d(
                K.expand_dims(inputs[0], 0),
                K.expand_dims(inputs[1], 3),
                strides = (1,1),
                padding = 'valid',
                data_format = 'channels_first' ),
         1],
       elems = [square_candidates, ones]
    )
    square_sum_can = (square_sum_can[:,0,0,:,:])
    square_target = K.sum(K.sum(K.sum(K.square(target2), axis = 1), axis = 1), axis = 1)
    square_target = K.expand_dims(square_target, 1)
    square_target = K.expand_dims(square_target, 1)
    square_target = tf.tile(square_target, multiples = [1,tf.shape(square_sum_can)[1],tf.shape(square_sum_can)[2]])
    denom = K.sqrt(tf.multiply(square_sum_can, square_target))
    output = tf.multiply(ct_dot, 1/denom)
    if mode == 'train':
    	output = tf.reshape(output, (-1,tf.shape(output)[1]*tf.shape(output)[2]))
    return output 
'''
def cosineSimilarity2(input):
	left = input[0]
	right = input[1]
	left_flatten = Reshape((-1,))(left)
	right_flatten = Reshape((-1, ))(right)
	left_l2 = K.l2_normalize(left_flatten, axis = 1)
	right_l2 = K.l2_normalize(right_flatten, axis = 1)
	output = K.batch_dot(left_l2, right_l2, axes = [1,1])
	return output

def cosineSimilarity3(input):
	left, right = input
	left = Reshape((-1,))(left)
        right = Reshape((-1, ))(right)
	return dot([left, right], axes = [1,1], normalize = True)
'''

def getOutputFunction(output):
	if output == 'e':
        	outFunc = euclideanDistance
    	elif output == 'c':
        	outFunc = cosineSimilarity
    	else:
        	raise Exception ('output function {0} is not supported, only support e for euclideanDistance and c for cosineSimilarity'.format(output))
	return outFunc

def createSiameseNetwork(output, filters = 32, basic_size = 128, use_resnet = True, mode = 'train'): 
    print "creating siamese network"
    input1 = Input((3, basic_size, basic_size))
    input2 = Input((3, None, None))
    outFunc = getOutputFunction(output)
    #conv = Conv2D(filters = filters, kernel_size = 3, strides = 1, padding = 'same', data_format = 'channels_first')
    #hb1 = conv(input1)
    #hb2 = conv(input2)
    hb = [input1, input2]
    #hb = [createHighwayBlock(hb, filters, kernel_size, stride, use_resnet = use_resnet) for i in range(num_hw_layers)]
    for i in range(num_hw_layers):
	if i == num_hw_layers - 1:
		act_func = 'linear'
	else:
		act_func = 'relu'
	hb = createHighwayBlock(hb, filters = filters, kernel_size = kernel_size, stride = stride, act_func = act_func, use_resnet = use_resnet)
    y=Lambda(outFunc, arguments = {'mode' : mode})(hb)
    model = Model([input1, input2], y)
    return model
