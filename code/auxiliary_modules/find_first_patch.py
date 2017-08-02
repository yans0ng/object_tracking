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
    r: clear the bounding boxes
    c: close the image window
"""
import argparse
import cv2

# the signature is required by cv2.setMouseCallback()
def click(event, x, y, flags, param):
    global offset1, offset2

    if event == cv2.EVENT_LBUTTONUP:
        start = (x-offset1, y-offset1)
        end = (x+offset2,y+offset2)
        print 'bounding box', start, ' -> ', end

        # draw a rectangle around the region of interest
        #image = original_image.copy()
        cv2.rectangle(image, start, end, (255,0,0),2 )
        cv2.imshow("image", image)

# construct the argumet parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path")
ap.add_argument("-s", "--size", required=False, help="Size of bounding box", type=int, default=64)
args = vars(ap.parse_args())

# load image
image = cv2.imread(args["image"])
original_image = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

patch_size = int(args["size"])
offset1 = offset2 = patch_size/2
if patch_size%2 == 1:
    offset2 += 1

while True:
    #display image
    cv2.imshow("image",image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = original_image.copy()

    elif key == ord("c"):
        break

cv2.destroyAllWindows()
