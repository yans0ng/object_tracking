import numpy as np
import cv2
import glob
import argparse
import os
from utils import *
avg = 2
"""
The class Bounding Box consists of coordinate information and a
deep copy of image patch. Instance variable are public so that
the class Tracker can access conviniently.
"""
class Bounding_Box:
    def __init__(self, x0, y0, patch):
        self.x0 = x0
        self.y0 = y0
        self.width = patch.shape[0]
        self.height = patch.shape[1]
        self.patch = patch.copy().astype(np.uint8)

"""
The tracker:
    - private instance variables:
        __obj_traj : 
            a list consisting of a series of bounding boxes, which can trace 
            the motion of object.
        __candidate_boxes : 
            buffer storage for candidate BB's. After the score is received 
            and the minimum is picked, the list will be cleared.

        __filepath :
            the filepath where the tracker reads the frame images. In the file,
            all the images should be named like frame#.jpg

        __num_frames :
            Total number of frames in the filepath

        __current_frame :
            Current frame is stored in order to generate bounded image.

        __current_index :
            Acting as iterator.

        __AVG_NUM :
            Number of patches to average.

        __target :
            The averaged target image. This image will be feeding the
            learning model.
    
    
    - Public methods:

        averaged_target()
            Compute the __averaged_target based on previous K frames.

        get_next_patches(): 
            generate 10 candidate patches randomly and move the iterator forward.

        catch_score( list scores): 
            Find the BB with least contrast score and store the BB in __obj_traj.

        output()
            Bind the latest bounding box and current frame together to generate
            an image with BB.

        has_next(): 
            Report whether the tracker has next frame.
"""
class Tracker:
    __obj_traj = []             # record the trajectory of obj
    __candidate_boxes = []      # buffer storage of candidate BB's

    def __init__(self,filepath,x0,y0,width,height,avg_num=2,sample_num=50,var=5.):
        self.__filepath = filepath
        self.__num_frames = len(glob.glob(filepath+'/*.jpg'))
        self.__current_frame = cv2.imread(filepath+'/0.jpg')
        self.__current_index = 0
        self.__SAMPLE_NUM = sample_num
        self.__VARIANCE = var
        print 'Dimension of video frames: ',self.__current_frame.shape
        

        first_patch = self.__current_frame[x0:x0+width,y0:y0+height,:]
        print 'Patch dimension', first_patch.shape
        first_BB = Bounding_Box(x0, y0, first_patch)
        self.__obj_traj.append(first_BB)
        self.__AVG_NUM = avg_num
        self.__target = first_patch.copy()

    """Average object patch from the most recent K BB's"""
    def averaged_target(self):
        # average previous patches
        average = np.ndarray((self.get_width(),self.get_height(),3),dtype=float)
        average.fill(0.)
        for BB in self.__obj_traj:
            average += BB.patch.astype(float) / float(self.__AVG_NUM)
        
        self.__avg_patch = average.astype(np.uint8)
        return self.__avg_patch.T

    """Sample patches from new frame based on the position of  previous BB"""
    def get_next_patches(self):
        self.__current_index += 1
        self.__current_frame = cv2.imread(self.__filepath+'/'+str(self.__current_index)+'.jpg')

        current_BB = self.__obj_traj[-1]
        print '(frame ',self.__current_index, ') Generating random patches...'

        candidate_patches = []
        del self.__candidate_boxes[:]
        qty = 0
        while (qty < self.__SAMPLE_NUM):
            x_start = int(np.random.normal(current_BB.x0, self.__VARIANCE))
            y_start = int(np.random.normal(current_BB.y0, 5.))
            x_end = x_start + current_BB.width
            y_end = y_start + current_BB.height
            
            # omit out of bound
            if( x_end < 0 or x_start > self.__current_frame.shape[0]):
                continue
            if( y_end + current_BB.height < 0 or y_start > self.__current_frame.shape[1]):
                continue

            if( x_start < 0):
                x_start = 0

            if( y_start < 0):
                y_start = 0

            if( x_end > self.__current_frame.shape[0] ):
                x_end = self.__current_frame.shape[0]

            if( y_end > self.__current_frame.shape[1] ):
                y_end = self.__current_frame.shape[0]

            if( x_start >= x_end or y_start >= y_end):
                continue

            # extract subimages as candidate patch
            new_patch = self.__current_frame[x_start:x_end, y_start:y_end,:]
            if new_patch.shape[0] > current_BB.width:
                print 'BB size changed!'
            if new_patch.shape[1] > current_BB.height:
                print 'BB size changed!'
            
            # zero padding
            x_diff = current_BB.width - new_patch.shape[0]
            y_diff = current_BB.height - new_patch.shape[1]
            if new_patch.shape[0] != current_BB.width or new_patch.shape[1] != current_BB.height:
                new_patch = cv2.copyMakeBorder(new_patch,0, x_diff, 0, y_diff, cv2.BORDER_CONSTANT,(0,0,0))
            
            # feed the learning model transposed patch
            candidate_patches.append(new_patch.T)
            
            new_BB = Bounding_Box(x_start, y_start, new_patch)
            self.__candidate_boxes.append(new_BB)
            qty += 1

        return candidate_patches

    def catch_score(self, scores):
        i = np.argmin(scores)
        self.__obj_traj.append(self.__candidate_boxes[i])

        # keep only patches of previous N steps
        if len(self.__obj_traj) > self.__AVG_NUM:
            self.__obj_traj.pop(0)

        # clear temporary storage
        del self.__candidate_boxes[:]
        print '(frame ', self.__current_index, ') finished'


    """Output is the current frame plus bounding box"""
    def output(self, stroke = 3):
        output = self.__current_frame.copy()
        current_BB = self.__obj_traj[-1]
        # Note that the coordinate in cv2 is (y, x)
        pt1 = (current_BB.y0, current_BB.x0)
        pt2 = (current_BB.y0 + current_BB.height, current_BB.x0+current_BB.width)
        cv2.rectangle(output, pt1, pt2, (255,0,0), 2)

        return output

    """Check if next frame is avalible"""
    def has_next(self):
        return self.__current_index+1 < self.__num_frames
    """
    The size of tracker depends on the latest bounding box.
    This is designed for future work where BB's may be
    changing size.
    """
    def get_width(self):
        return self.__obj_traj[-1].width

    def get_height(self):
        return self.__obj_traj[-1].height

"""View image for debugging purpose"""
def debug_show(img):
    cv2.imshow('first frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--source_file", required=True, help="Video source")
    ap.add_argument("-x", "--init_x", required=True, help="Initial Object Location(x)",type=int)
    ap.add_argument("-y", "--init_y", required=True, help="Initial Object Location(y)",type=int)
    ap.add_argument("-s", "--size", required=False, default=64, help="Object Size", type=int)
    ap.add_argument("-n", "--sample_num", required=False, default=50, help="Number of Samples",type=int)
    ap.add_argument("-v", "--variance", required=False, default=5.,help="Sampling Variance",type=float)
    ap.add_argument("-o", "--output_file", required=False,default='tracker_output', help="Output file")
    ap.add_argument("-w", "--weight_path", required = True, help = "pretrained weight path")
    ap.add_argument("-c", "--output_func", required = False, default = 'e', help = "output function, e for euclidean, c for cosine similarity")
    args = vars(ap.parse_args())
    # instantiate tracker
    t = Tracker(args["source_file"],args["init_x"],args["init_y"],args["size"],args["size"], avg_num = avg, sample_num = args["sample_num"],var = args["variance"])

    count = 0
    while( t.has_next()):
        count += 1
        candidate_patches = t.get_next_patches()
        scores = calculateScores(t.averaged_target(), candidate_patches, args['weight_path'], args['output_func'])
        # test with arbitary score list
        #scores = [1, 2, 3, 4, 5]

        t.catch_score(scores)

        # if you want to store target as well, create a file called "target"
        # and uncomment the following two lines.
        #target = t.averaged_target()
        #cv2.imwrite("target/"+str(count)+".jpg",target)

        bounded_frame = t.output()
        if(not os.path.isdir(args["output_file"])):
            os.mkdir(args["output_file"])
        cv2.imwrite(args["output_file"]+"/"+str(count)+".jpg",bounded_frame)
        #cv2.imwrite("images/"+str(count)+".jpg",bounded_frame)
        #cv2.imshow('frame',bounded_frame)

    print 'Video ended...'
    print 'Total number of frames = ', count
