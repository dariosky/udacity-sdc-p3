# an ImageDataGenerator, is a convenient class for augment the dataset
# in this case it's disabled, track 1 is quite stable
# we'll flib horizontally
import random

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import cv2

datagen = ImageDataGenerator(
    # rotation_range=5,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.2,
    # horizontal_flip=True,
    fill_mode='nearest'
)


def invariant(img, steer):
    return img[50:-20, :], steer  # no variation


def flipped(img, steer):
    return np.fliplr(img[50:-20, :]), -steer


variants = (
    invariant,
    flipped  # flipped horizontally
)


def img_set_generator_factory(selected_set, image_shape=(90, 320, 3), batch_size=32):
    """ This Keras generator takes a selected_set
    (that is groups of valid center-left-right paths with steers)
    and create a batch for the fit function
    """
    batch_x = np.zeros((batch_size,) + image_shape)
    batch_y = np.zeros((batch_size), dtype=np.float32)
    index = 0
    while 1:
        s = selected_set
        s.images, s.steers = shuffle(s.images, s.steers)  # shuffle on every loop

        for img_path, steer in zip(s.images, s.steers):
            img = mpimg.imread(img_path)  # img
            # Let's cut the original image to avoid the car hood and the top sky

            variant_function = random.choice(variants)
            # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            batch_x[index], batch_y[index] = variant_function(img, steer)
            index += 1
            if index == batch_size:
                # print("returning", batch_x.shape, batch_y.shape)
                yield (batch_x, batch_y)
                index = 0
