
P3 - Behavioral cloning
-----------------------

# Data collection and augmentation

Let's start getting the images.

I used the sample data for track 1, then I was looking to get new training data in in the simulator.

Driving the car properly for training was the major problem in my case, I found the sample dataset having a much smoother input and it's training behaviour was much more accurate.
This is probably due to the keyboard input that produce hard turns and make the trained model more "wobbling".

At first I used the dev version, using the mouse seemed an interesting thing to have smooth angles, but it ends up to be hard to control, and I was producing bad data.
(as a result, I started deleting some batch of images when I was going out of road, so in the cleanup below, I'm filtering out csv lines leading to inexistents files).
Also I liked in the stable version, that 3 pictures are taken, it allow to produce a lot more training data, expecially useful for "recovery" normal angles.

I'm considering the left/right images, correcting the angle of 0.20 (in the direction of the center of the road).

Using a throttle/brake adjustment based on the predicted steer I was able to get good results, and surprisingly to train a model that complete the first track using only the Udacity samples.

To parse the CSV, validate that the images exist and dealing with different recording session I created the `ImageSet` class in the `image.py`. It is used to collect all the center/left/right images and save them with the "corrected" steer.


```python
%load_ext autoreload
% autoreload 2
% matplotlib inline

```


```python
# get the images and the steer angles
# clean up the images, considering only ones leading to existing files
# the collected images are quite big, and are not part of this repository


from images import *
sample_set = get_sample_set()
training_set = get_training_set()
```

    Balancing leftright of 0.2
    Checking /home/dario/tmp/driverecords/data
    Checking /home/dario/tmp/driverecords



```python
# Import what we need
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
```
## Loaded the sample set
Let's see some random image for the training set (this is the first group of my recording).  
A positive angle means the car will turn right, negative means left.

In the normalization, when training I choose randomly one of the variations of the original image:
the original or a horizontal flip (with inverted steer).
Both of them are cutted top-bottom so to remove the car hood and part of the sky (that is not relevant)

Here, for a random image I display the variations

```python
from normalization import *

selected_set = training_set
s = training_set
rnd = random.randint(0, len(s)-1)
img_path = s.images[rnd]
orig_steer = s.steers[rnd]

plt.figure()
plt.title("steer: {steer:2.2f}".format(
    steer=orig_steer
    ))
plt.axis('off')
image = mpimg.imread(img_path)
plt.title('original: %2.2f'%orig_steer)
plt.imshow(image)

plt.figure()
for i, v in enumerate(variants):
    img, steer = v(image, orig_steer)
    plt.subplot(1,2,i+1); plt.axis('off')
    plt.title('variant: %2.2f' % steer)
    plt.imshow(img)

```


![png](output_6_0.png)



![png](output_6_1.png)


# Model

Model is defined in the model.py using Keras, here we'll just import it.
I started adapting the NVIDIA architectural found in the 
[End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
The original model itself is made of 5 convnet, followed by other 5 Dense layers.
I added a Dropout layer set at 20% to reduce overfitting, and I added a pooling layer (MaxPooling 2x2) after the first 3 convnet, to reduce the model size.
On my machine, using GPU, the Pooling helped fit the graphic card memory, using a batch size of 128.


```python
from model import get_model
model = get_model()
```

    Loaded model from disk


# Training

For the training I used the adam optimized and the mean_squared_error as loss function.

Using Keras generator to load images on batches was required given the large size of the dataset images.
The `normalization.py` contains the generator, it randomize the datasets on every epoch and for every image choose to get the original or an horizontal flip, training on a reverse steer value.

I also found it useful to slighly cut the images, to remove the car hood and the top part of the sky.  
Images are then cutted like this `img[50:-20, :]` (50px top, 20 pixels bottom)



```python
from normalization import img_set_generator_factory
model.compile(
    optimizer='adam',
    loss='mse'
)

selected_set = sample_set
history = model.fit_generator(generator=img_set_generator_factory(selected_set, batch_size=128),
                              samples_per_epoch=len(selected_set),
                              nb_epoch=2)
```

    Epoch 1/2
    24192/24108 [==============================] - 257s - loss: 0.0152   
    
    Epoch 2/2
    24192/24108 [==============================] - 251s - loss: 0.0153   


For the current model, as I was mentioning before, I trained only to the sample Udacity dataset.  
A `batch_size=128` to fit in memory and 10 generations (with a generation big as the dataset size).


# Drive

The drive part didn't require many changes, but one, that I found useful.

While trying to use as less data as possible, I found that correcting the speed based on the predicted steering was really helpful.

I changed the call to `send_control` to set a throttle of 0.2, accelerating, always but when the speed is over 15 and the steering angle has an aplitutde > 0.1 we break a little.

This approach, works quite well and the car go fast enough on the straight lanes, and slow down and oscillate a little on the curves.

# Future improvements

I'm waiting for an analog joystick to be able to get better car recording. Given the results with a limited dataset having proper additional data would be beneficial. The advantage was that the training phase last only ~1 hour on my laptop GPU.

Given the time constrain I didn't spent time on training and trying the 2nd track, however I left in the normalization the ImageDataGenerator keras generator, that is really promising to generate a large amount of derived images, useful for the 2nd track when we sometime have up and down shifts.

For the same track, having also an augmentation based varying the image luminance could be useful.
