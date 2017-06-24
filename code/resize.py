import numpy as np
import cv2, sys

path = sys.argv[1]
size = int(sys.argv[2])
v_ratio = float(sys.argv[3])
save_path = sys.argv[4]
data = np.load(path)
left = data['left']
right = data['right']
label = data['label']
nleft = np.zeros((left.shape[0], 3, size, size))
nright = np.zeros((right.shape[0], 3, size, size))
for i in range(left.shape[0]):
	nleft[i] = cv2.resize(left[i].transpose(1,2,0).astype(np.uint8), (size, size)).transpose(2,0,1)
	nright[i] = cv2.resize(right[i].transpose(1,2,0).astype(np.uint8), (size, size)).transpose(2,0,1)
print "nleft shape: ", nleft.shape
print "nright shape: ", nright.shape
if not v_ratio == 0:
	idx = np.arange(left.shape[0])
	np.random.shuffle(idx)
	num_val = int(v_ratio * len(idx))
	vidx = idx[0:num_val]
	tidx = idx[num_val:]
	tleft = nleft[tidx]
	vleft = nleft[vidx]
	tright = nright[tidx]
	vright = nright[vidx]
	tlabel = label[tidx]
	vlabel = label[vidx]
	tsave_path = save_path + "_train.npz"
	vsave_path = save_path + "_val.npz"
	np.savez(tsave_path, left = tleft, right = tright, label = tlabel)
	np.savez(vsave_path , left = vleft, right= vright, label = vlabel)
else:
	save_path += ".npz"
	np.savez(save_path, left = left_new, right = right_new, label = data['label'])

print "resize is finish"
