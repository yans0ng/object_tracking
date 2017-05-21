"""
find_first_frame.py

Yan-Song Chen
May 11, 2017

This program allows the user to pick up the position of  bounding box of the 
first frame. The first frame can then be parse to tracker to drive the face 
tracking program.

Arguments:
    -i: required, the image path
    -s: optional, the patch size

Functions:
    c: clear the bounding boxes
    q: end program andclose the image window
"""
import argparse
import cv2
from video2image import *
import os

global fp_offset1, fp_offset2, fp_image

# the signature is required by cv2.setMouseCallback()
def click(event, x, y, flags, param):
    global fp_offset1, fp_offset2

    if event == cv2.EVENT_LBUTTONUP:
        start = (x-fp_offset1, y-fp_offset1)
        end = (x+fp_offset2,y+fp_offset2)
        # coordinates in cv2 is reversed, so print out reversely
        print 'bounding box', start[::-1], ' -> ', end[::-1]
        # draw a rectangle around the region of interest
        cv2.rectangle(fp_image, start, end, (255,0,0),2 )
        cv2.imshow("image", fp_image)

#
def semi_auto_recognize(image_path, patch_size):
    global fp_offset1, fp_offset2, fp_image
    fp_image = cv2.imread(image_path)   # load image
    original_image = fp_image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)

    fp_offset1 = fp_offset2 = patch_size/2    # compute corner coordinate
    if patch_size%2 == 1:               # handle odd number
        fp_offset2 += 1

    while True:
        cv2.imshow("image",fp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            fp_image = original_image.copy()

        elif key == ord("q"):
            break
    cv2.destroyAllWindows()

def pairs(s):
    try:
        x,y = map(int,s.split(","))
        return (x,y)
    except:
        raise argparse.ArgumentTypeError("Size must by (x, y)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video_path", required=False, help="Video path")
    ap.add_argument("-f", "--frame_path", required=False, help="Frame storage path", default="video_frames")
    ap.add_argument("-fs","--frame_size", required=False,help="Frame size", default=(640,341), type=pairs, nargs=1)
    ap.add_argument("-s", "--size", required=False, help="Size of bounding box", type=int, default=64)
    args = vars(ap.parse_args())

    # if frame path does not exist, make a directory
    if(not os.path.isdir(args["frame_path"])):
        os.mkdir(args["frame_path"])

    if (args["video_path"] != None):
        video2image(args["video_path"],args["frame_path"],args["frame_size"])
    
    semi_auto_recognize(args["frame_path"]+'/frame0.jpg', args["size"])
