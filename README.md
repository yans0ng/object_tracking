## Introduction
"tracker.py" is a program that tracks the object given the first location of
bounding box. This program is written for the final project of ELEN 4830 Digital
Image Processing (Spring 2017).

Author: Chen, Yan-Song, Lin, Hung-Shih
Date  : May 14, 2017

## Arguments
Required arguments:
    -p: video source file. Note that the frame images must be named as 
        "frame#.jpg", where the number must begin from 0 and should be 
        consecutive. 
        For examle: frame0.jpg  frame1.jpg ... frame99.jpg


    -x: initial object location. The x coordinate of left and upper of the 
        initial bounding box.

    -y: initial object location. The y coordinate of left and upper of the
        initial bounding box.


Optional arguments:
    -s: the size (in pixels) of bounding box. 
        Default value is 64.

    -n: number of samples of candidate patches of each frame
        Default value is 50.

    -v: the sampling variance used for generating random patches.
        Default value is 5.0.

    -o: the output directory. If the specified directory does not exist, this
        program will automatically create one. The output images will be
        "#.jpg".
        Default path is "tracker_output"

## Code Example
The following command specify the video path and the upper left cornor of bounding 
box to be (30, 30)
python tracker.py -p some_filepath -x 30 -y 30 
