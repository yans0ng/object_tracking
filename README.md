## Introduction
"tracker.py" is a program that tracks the object given the first location of
bounding box. This program is written for the final project of ELEN 4830 Digital
Image Processing (Spring 2017).

## Contributors
    * Chen, Yan-Song  Columbia University
    * Lin, Hung-Shih  Columbia University

## Contents
1. tracker.py: the main program that interects with the Siamese network model and video file.

2. siamese_network.py: construct the architecture of Siamese network.

## Running the program
1. "tracker.py" is the only source file that need to be execute. Before running
this program, make sure "siamese.py" is placed in the same directory.
2. The images of frames of video must be extracted beforehand. They should be
named as "frame#.jpg". For example, frame0.jpg, frame1.jpg, frame2.jpg..., and so on.
3. Place the frame images in a file, which is also in the directory of source files.
4. Up till now, the directory should have at least "tracker.py", "siamese.py",
"video_frames_file". Use the command below to run tracker.py
    $python tracker.py -p video_frames_file -x 30 -y 30

This command specifies the image source, and the left upper coordinate of 
bounding box to be (30, 30).
Advanced parameters can be assigned via the following arguments.

### Required arguments:
    -p: video source file. Note that the frame images must be named as 
        "frame#.jpg", where the number must begin from 0 and should be 
        consecutive. 
        For examle: frame0.jpg  frame1.jpg ... frame99.jpg


    -x: initial object location. The x coordinate of left and upper of the 
        initial bounding box.

    -y: initial object location. The y coordinate of left and upper of the
        initial bounding box.


#### Optional arguments:
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

    $python tracker.py -p some_filepath -x 30 -y 30 
