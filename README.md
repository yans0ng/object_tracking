## Introduction
This project originated from the final project of Digital Image Processing at Columbia University (ELEN 4830 Spring 2017).

## Model Details:
The algorithm is based on Siamese model with several highway nets. Model is first trained offline with
static images to learn features of different classes. In the tracking phase, assuming known initial object location, the tracking algorithm samples randomly from current frame centered on the objection location in the previous frame. Similarity for each pair is computed, and the patch that has maximum similarity is choosen as the object location at the current frame. Averaging target frames is used to reduce error-accumulation problem. 

## Contributors
* Hung-Shih Lin,  Columbia University
* Yan-Song Chen,  Columbia University

## Contents

* main.py: the program used to call functions on utils. This program is for training Siamese model.

* siamese_network.py: constructs the architecture of Siamese network.

* tracker.py: takes in image frames and mark object bounding boxes by prediction of the Siamese network model.

* find_first_frame.py: import video frames and allow user to pick the first patch interactively

* video2image: convert video to image frames

* utils.py: contains functions used for training and prediction.

* resize.py: resize images

## Preprocessing training data
The face of human beings has sophisticated details with which we distinguish one from another. Hence, we pick the face to train out 
Siamese network model. In this project, the face training data was downloaded from http://vis-www.cs.umass.edu/lfw/ .Individuals who 
has only 1 photo were used to generate negative pairs; individuals who has 2 or more photos are used to generatepositive pairs. 
In order to generate training data:

1. Download data from http://vis-www.cs.umass.edu/lfw/
2. Decompress and place the "lfw" file to the same directory of face_data_prep.py
3. Execute face_data_prep.py

#### Code example:
    $ python face_data_prep.py

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

#### Code Example:
    $python main.py -mode 1 -data <path for training data> -ep 5 -bs 2 -lr 0.001 -output l -loss c -split_ratio 0.2

## Capturing initial object location
We focused on the tracking problem. Therefore, the program is semi-automated, which means the user have to assign the
first location of bounding box of object.

#### Required arguments:
    -v: path of video to convert

    -f: file to store image frames

1. If -v and -f are assigned, store image frames to -f, and display the first image in -f.

2. If only -v is assigned, create a file named "video_frames", and display the first image in the file.

3. If only -f is assigned, do not convert video to images, and only disply the first image in -f.

#### Optional arguments:
    -fs: size of image frames. Default = (640,340)

    -bs: size of bounding box. Default = 64

#### Picking the location of bounding box

1. Click on the region of interest, a bounding box centered at the click will be displayed. Also, the command line will print the coordinate of the bounding box.

2. Press c to clear the bounding boxes

3. Press q to quit

#### Code example:

    $ python find_first_patch.py -v video_file -f where_to_store -fs 1080,1240 -bs 100

    $ python find_first_patch.py -f image_file -bs 100

## Object tracking
After getting the initial position of the object and the pretrained model, execute tracker.py

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
