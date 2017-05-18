"""
In order to run this program, download dataset from http://vis-www.cs.umass.edu/lfw/
Place the "lfw" file at the same directory of this program.
"""
import glob
import os
import numpy as np
import cv2
import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt
import imutils
#%matplotlib inline

personFiles = glob.glob('lfw/*')
print len(personFiles)

positive_cases = []
num_of_pos_pairs = 0
num_of_pos_images = 0
negative_cases = []
for personFile in personFiles:
    picturePaths = glob.glob(personFile+'/*.jpg')
    if len(picturePaths) == 1:
        negative_cases.append(picturePaths[0])
    elif len(picturePaths) == 2:
        positive_cases.append(picturePaths)
        num_of_pos_images += len(picturePaths)
        num_of_pos_pairs += 1
        
    else:
        positive_cases.append(picturePaths)
        num_of_pos_images += len(picturePaths)
        num_of_pos_pairs += len(picturePaths)

print 'Number of positive pair', num_of_pos_pairs
print 'Number of positive images', num_of_pos_images
print 'Number of negative images', len(negative_cases)
print 'Total pairs ', len(negative_cases) + num_of_pos_pairs
print 'Total images ', len(negative_cases) + num_of_pos_images

left = np.ndarray([12454,3,250,250], dtype = np.uint8)
right = np.ndarray([12454,3,250,250], dtype = np.uint8)

""" generate negative cases """
for idx, negative_case in enumerate(negative_cases):
    # shift the negative index by 1
    #print idx
    left[idx,:,:,:] = right[(idx+1)%4069,:,:,:] = mpimg.imread(negative_case).astype(np.uint8).T

""" generate positive cases """
idx = 4069
for positive_case in positive_cases:
    if len(positive_case) == 2:
        left[idx,:,:,:] = mpimg.imread(positive_case[0]).astype(np.uint8).T
        right[idx,:,:,:] = mpimg.imread(positive_case[1]).astype(np.uint8).T
        idx += 1
    else:
        for j,image in enumerate(positive_case):
            left_idx = idx + j
            right_idx = idx + (j+1)%len(positive_case)
            #print 'left ', left_idx
            #print 'right', right_idx
            left[left_idx,:,:,:] = right[right_idx,:,:,:] = mpimg.imread(image).astype(np.uint8).T
        idx += len(positive_case)

# verification
for i in [0,1,4067,4068,4069,4070,12452,12453]:
    print 'i =',i
    print 'left'
    plt.imshow(left[i,:,:,:].T)
    plt.show()
    print 'right'
    plt.imshow(right[i,:,:,:].T)
    plt.show()

# make label data
label = np.ones([12454,1],dtype=int)
label[0:4069,0] = -1

np.savez('face.npz',right = right, left = left, label = label)

# load data
#with np.load('face.npz') as data:
#    left = data['left']
