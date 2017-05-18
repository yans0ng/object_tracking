## Introduction
This project originiates from the final project of Digital Image Processing at Columbia University (ELEN 4830 Spring 2017).

## Model Details:
The algorithm is based on Siamese model with several highway nets. Model is first trained offline with
static images to learn features of different classes. In the tracking phase, assuming known initial object location, the tracking algorithm samples randomly from current frame centered on the objection location in the previous frame. Similarity for each pair is computed, and the patch that has maximum similarity is choosen as the object location at the current frame. Averaging target frames is used to reduce error-accumulation problem. 

## Contributors
* Hung-Shih Lin,  Columbia University
* Yan-Song Chen,  Columbia University

## Contents

* main.py

The program used to call functions on utils. This program is for training Siamese model.

* siamese_network.py

Constructs the architecture of Siamese network.

* tracker.py

The main program that interects with the Siamese network model and video file.

* utils.py

Contains functions used for training and prediction.

* resize.py

Resize images

## Training the model
Execute main.py with following arguments:

#### Required arguments: 
    -data: training data or data used for prediction

    -mode: 0 fo prediction. otherwise training

#### Optional arguments:
    -ep: epochs
    
    -bs: batch size
    
    -lr: learning rate
    
    -output: output function. e for euclidean distance, c for cosine similarity
    
    -wpath: pretrained weight path
    
    -wspath: path for saving trained weight
    
    -lspath: path for saving log file
    
    -pspath: path for saving prediciton
    
    -split_ratio: ratio of training data used for validation
    
    -loss: loss function. l for logistic, c for contrastive
    
    -val_data: validation data. if given, overwrite split ratio

### Code Example:
    $python main.py -mode 1 -data <path for training data> -ep 5 -bs 2 -lr 0.001 -output l -loss c -split_ratio 0.2

## Running the program
1. "tracker.py" is the only source file that need to be execute. Before running
this program, make sure "siamese.py" is placed in the same directory.
2. The images of frames of video must be extracted beforehand. They should be
named as "frame#.jpg". For example, frame0.jpg, frame1.jpg, frame2.jpg..., and so on.
3. Place the frame images in a file, which is also in the directory of source files.
4. Up till now, the directory should have at least "tracker.py", "siamese.py".Parameters can be assigned via the following arguments.

#### Required arguments:
    -p: video source file. Note that the frame images must be named as 
        "frame#.jpg", where the number must begin from 0 and should be 
        consecutive. 
        For examle: frame0.jpg  frame1.jpg ... frame99.jpg


    -x: initial object location. The x coordinate of left and upper of the 
        initial bounding box.

    -y: initial object location. The y coordinate of left and upper of the
        initial bounding box.

    -w: pretrained weight path


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

    -e: model's output function. e for euclidean distance. c for cosine similarity

#### Code Example
The following command specify the video path and the upper left cornor of bounding 
box to be (30, 30)

    $python tracker.py -p some_filepath -x 30 -y 30 
