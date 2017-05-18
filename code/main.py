import argparse
import numpy as np
from siamese_network import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-data', required = True, help = 'training data or data to predicted')
parser.add_argument('-mode', required = True, help = '0 for prediction, otherwise training with data', type = int)
parser.add_argument('-output', help ='Output function, e for euclidean distance, c for cosine similarity', default = 'e')
parser.add_argument('-wpath', '--weight_path', help = 'pretrained weight', default = None)
parser.add_argument('-split_ratio', '--validation_split_ratio', help = 'ratio of validation data used during training'default = 0.2, type = float)
parser.add_argument('-ep', '--epoch', help = 'number of epochs', default = 5, type = int)
parser.add_argument('-bs', '--batch_size', help = 'mini batches', default = 4, type = int)
parser.add_argument('-wspath', '--weight_save_path', help = 'Path for saving trained weight', default = '{epoch:02d}-{val_loss:.2f}.hdf5')
parser.add_argument('-pspath','--prediction_save_path', help = 'Path for saving predicted result', default = 'prediction.npz')
parser.add_argument('-lr', '--learning_rate', help = 'learning rate for RMSprop optimizer', default = 0.001, type = float)
parser.add_argument('-lspath', '--log_save_path', help = 'Path for saving log. Used for visualization with Tensorboard', default = 'log')
parser.add_argument('-loss', '--loss_function', help = 'Loss function for model, l for logistic loss, c for contrastive loss', default = 'l')
parser.add_argument('-val_data', '--validation_data', help = 'validation data path. If provided, overwrite validation split ratio', default = None)
args = parser.parse_args()
mode = args.mode
output_func = args.output
data_path = args.data
batch_size = args.batch_size
weight_path = args.weight_path

if mode == 0:
	pred_save_path = args.prediction_save_path
	pred = predict(output_func, weight_path, data_path, batch_size)
	np.savez(pred_save_path, prediction = pred)
	print "Prediction success"
else:
	weight_save_path = args.weight_save_path
	log_save_path = args.log_save_path
	val_split_ratio = args.validation_split_ratio
	epochs = args.epoch
	lr = args.learning_rate
	loss = args.loss_function
	val_path = args.validation_data
	train(train_data = data_path, output_func = output_func,
		  loss_func = loss_func, epochs = epochs,
		  lr = lr, batch_size = batch_size,
		  val_data = val_path, weight_path = weight_path,
		  val_ratio = val_split_ratio, weight_save_path = weight_save_path,
		  log_save_path = log_save_path)
	print 'Training Complete'
