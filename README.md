# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md this file, summarizing the results

A couple of additional utility libraries are included:
* images.py containing the ImageSet class useful to parse, cleanup and handling the
 image paths and the steering labels
* normalization.py containing the code for normalization and the generator
  augmenting the dataset with random rotation, and horizontal flips.


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
./drive.py
```

(model.h5 is used by default, and images are recorded in the ./images subfolder)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based on the NVIDIA [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
and consists of 5 convolution neural network layers and 5 dense layers, as shown in the following schema.

The model includes various ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 31).

#### 2. Attempts to reduce overfitting in the model

The model contains a Dropout layer (20%) in order to reduce overfitting (model.py:52).

After every convolutional layer a MAX Pooling layer (2x2) has been introduced to reduce the spatial size and hence reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was
 not overfitting (code line 84-106). (the selected dataset is the union of the Udacity sample + a serie of recordings done with the beta-simulator as 10 Feb 2017) (model.py:71-76).

A validation set of 20% the dataset size is randomly chosen.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py:80).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used a combination of center lane driving, recovering from the left and right sides of the road from the Udacity Samples (when using the stable simulator version).

I then started using the beta version (Feb2017) it record only the center camera (the left/right camera are cleaned when imported in the image.py:67-107).

To smooth the steering records (my records came from keyboard usage), I processed the training data with a rolling average. For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to combine a set of convolutional layer with dense neural networks layers and to add an appropriate number of layers to reduce overfitting.

My first step was to use a convolution neural network model similar to the NVIDIA Self Driving Cars model, that was implemented for the exact same purpose.

The large amount of ways to avoid overfitting was quite useful during training: while lowering the loss function I noticed that the error produced on the training set was always in the order of the one produced with the validation set.
This is a good marker to know that the model is not overfitting.

I had found also that the various Pooling layers helped keeping the size of the hyper-parameters down, and this allowed me to increase the batch size enough, without occurring in memory errors.

I had various challenges to produce correct appropriate driving data with the keyboard, the steering angles are not smooth, using a mouse at full speed was hard to keep the car on track and proceeding slowly was taking ages. I also found that the version of simulator I was using (the stable as of Jan 2017) was behaving differently that the one of the reviewers.

Namely, in previous submissions, the car was driving correctly (even if wobbling often) on my machine, but not on the reviewers one. I thought it can be for GPU processing power differences (if the `send_control` arrives late the machine keep the old throttle and steering, falling off track). Updating the simulator to the new beta version (in Feb 2017) I found that the behavior was quite different than the one with the stable version!
Also the produced steering records look a lot better, so I trained again the model and now
it's easy also record images and video to showcase the proper driving behavior.

I also preprocessed the input steering to smooth them with a rolling average of two consecutive valid records, where the current frame has a weight of 75% (images.py:65).

The datasets, prior the augmentation are composed of a total of 64.358 images.
46.812 center images, 8.773 left and 8.773 right images (I discarded 71.802 left/right images on straight).

With augmentation (random rotation and hflip) the dataset has been expanded to 411.896 points.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py:27-66) consisted of the model discussed above. Keras model definition is really readable and quickly tweakable, it greatly helped experimenting various parameter changes.

Here is a visualization of the architecture:

<div style='text-align:center'>
  ![The model visualized][model]
</div>

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, for the previous submission I used the stable version
and I was using the center images, and the left/right one adjusted of 0.15 in the center direction (this helps producing more data, and is based on the intuition that if the car is slightly more on the left we should turn a little more right to move it to the center).

I then used the beta simulator on track1 to record a new dataset, with only center images.
Both the datasets have been driven using the keyboard, so I added the above-mentioned rolling-average function to smooth them.

Here an example of a center image, color normalized from the dataset

![alt text][image-center]

All the datasets have been normalized (to have a zero average, and ~1 sigma) and augmented.

The augmentation consisted of random small rotations (up to 5 degrees) like the ones below using the nice Keras ImageDataGenerator (normalization.py:13):

![Random rotation 1][image-rot1]
![Random rotation 2][image-rot2]

Then again, every single image is used twice, on e normally and once flipped horizontally inverting the steering label. Here are a couple of examples from the same image.

![Random flipped rotation 1][image-flip-rot1]
![Random flipped rotation 2][image-flip-rot2]

The augmented datasets are 8 times bigger than the original ones.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

I set the max number of epochs to 30, but I'm using a EarlyStopping callback (model.py:94) that stops when the loss is not decreasing for 3 consecutive generations. That will made my training terminate at 13th generation (taking the results of generation 9).

I used the Keras ModelCheckpoint (model.py:93) to save the model at every epoch and to keep the best result only.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Here is the video record of track 1:

<video controls="controls">
  <source type="video/mp4" src="docimg/recording.mp4"></source>
  <p>Your browser does not support the video element.</p>
</video>

[//]: # (Image References)

[model]: docimg/p3_model.png "Model Visualization"
[video1]: docimg/recording.mp4 "Recorded video of track 1"
[image-center]: docimg/output_6_1.png "Grayscaling"
[image-rot1]: docimg/output_6_3.png "Random rotation 1"
[image-rot2]: docimg/output_6_5.png "Random rotation 2"
[image-flip-rot1]: docimg/output_6_13.png "Random flipped rotation 1"
[image-flip-rot2]: docimg/output_6_15.png "Random flipped rotation 2"
