import numpy as np
import cv2, sys

path = sys.argv[1]
size = int(sys.argv[2])
data = np.load(path)
left = data['left']
right = data['right']
left_new = np.zeros((left.shape[0], 3, size, size))
right_new = np.zeros((right.shape[0], 3, size, size))
for i in range(left.shape[0]):
	left_new[i] = cv2.resize(left[i].transpose(1,2,0), (size, size)).transpose(2,0,1)
	right_new[i] = cv2.resize(right[i].transpose(1,2,0), (size, size)).transpose(2,0,1)
save_path = sys.argv[3]
np.savez(save_path, left = left_new, right = right_new, label = data['label'])
print "resize is finish"
