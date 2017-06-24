from utils import *
from keras.models import Model
from siamese_network import *
import numpy as np
import cv2
np.random.seed(123456)
model = 'model/cosine_face_val80.hdf5'
#target = np.zeros((1,3,128,128))
#candidates = np.zeros((1,3,256,256))
#target = np.random.normal(10, 15, (1,3,128,128))
#candidates = np.random.normal(30, 5, (1,3,256,256))
#candidates[0, :, 255, 255] = -1000
data = cv2.imread('data/test.jpg')
data = np.transpose(data, (2,0,1))
data = np.expand_dims(data, axis = 0)
target = data[[0],:, 0:128, 0:128].astype(np.float64)
candidates = data[[0], :, 0:256, 0:256].astype(np.float64)
print candidates.shape
result = calcOptScores(target, candidates,model, 'c')
print result
