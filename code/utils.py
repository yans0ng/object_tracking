from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from siamese_network import *
from keras import optimizers
import numpy as np
model = None
basic_size = 127
use_resnet = False
def getOptimizers(lr = 0.001):
	return optimizers.RMSprop(lr = lr)

def contrastive_loss(y_true, y_pred):
    margin = 100
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))

def logit_loss(y_true, y_pred):
	return K.log(1+K.exp(-y_true*y_pred))

def square_loss(y_true, y_pred):
	return K.mean(K.square(y_true - y_pred), axis = -1)

def customMetric(y_true, y_pred):
	return K.mean(K.equal((y_true+1)/2, K.round((y_pred+1)/2)), axis=-1)

def getLossAndMetrics(loss_func):
	if loss_func == 'c':
		return contrastive_loss, 'mean_squared_error'
	elif loss_func == 's':
		return square_loss, customMetric
	elif loss_func == 'l':
		return logit_loss, customMetric
	else:
		raise Exception('loss function {0} is not supported. We currently only support c for contrastive loss, l for logistic loss and s for square loss'.format(loss_func))

def getEDOptLocation(input):
	return np.unravel_index(np.argmin(input), input.shape)

def getCSOptLocation(input):
	return np.unravel_index(np.argmax(input), input.shape)	

criteria = {
       'e' : getEDOptLocation,
       'c' : getCSOptLocation
}

def genCallBacks(weight_save_path, log_save_path):
        callback_tb = TensorBoard(log_dir=log_save_path, histogram_freq=0, write_graph=True, write_images=True)
        callback_mc = ModelCheckpoint(weight_save_path, verbose = 1, save_best_only = True, save_weights_only = True, period = 1)
        #callback_es = EarlyStopping(min_delta = 0, patience = 1, verbose = 1)
        return [callback_tb, callback_mc]

def readData(path, train = True):
	data = np.load(path)	
	left = data['left']
	right = data['right']
	if train == True:
		label = data['label']
		return left, right, label
	return left, right

def train(train_data, output_func = 'e',
		 loss = 'c', epochs = 5, 
		 batch_size = 1, lr = 0.001, 
		 val_data = None, weight_path = None, 
		 val_ratio = 0.2, weight_save_path = '{epoch:.2d}-{val_loss:.2f}',
		 log_save_path = 'log'
		 ):
	if train_data == None:
		raise Exception('No training data')
	model = createSiameseNetwork(output_func, basic_size = basic_size, use_resnet = use_resnet)
	callbacks = genCallBacks(weight_save_path, log_save_path)
	optimizer = getOptimizers(lr)
	loss_func, metric = getLossAndMetrics(loss)
	left, right, label = readData(train_data, train = True)
	if weight_path:
		print "load weight from {}".format(weight_path)
		model.load_weights(weight_path)
	
	model.compile(optimizer=optimizer, loss=loss_func, metrics = [metric])
	
	if val_data == None:
		model.fit([left, right], label, epochs = epochs, batch_size = batch_size, validation_split = val_ratio, callbacks = callbacks)
	else:
		print 'Use provided validation data instead of splitting train data'
		left_val, right_val, label_val = readData(val_data, train = True)
		model.fit([left, right], label, epochs = epochs, batch_size = batch_size, validation_data = ([left_val, right_val], label_val), callbacks = callbacks)

def predict(output_func, weight_path, data_path, batch_size):
	if data_path == None:
		raise Exception('No training data')
	left, right = readData(data_path)
	model = createSiameseNetwork(output_func)
	model.load_weights(weight_path)
	return model.predict([left, right], batch_size = batch_size)

def calcOptScores(target_patch, candidate_patches, weight_path, output_func):
	global model
	global critera
	criterium = None
	if model == None:
		criterium = criteria[output_func]
		print "First time importing the module, Initializing model..."
		model = createSiameseNetwork(output_func, mode = 'test')
		print "Loading weights from ", weight_path
		model.load_weights(weight_path)
		print "Initialization Complete"
	scores = model.predict([target_patch, candidate_patches])
	loc = criterium(scores)
	return loc
	
