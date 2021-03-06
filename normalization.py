# an ImageDataGenerator, is a convenient class for augment the dataset
# in this case it's disabled, track 1 is quite stable
# we'll flib horizontally
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

datagen = ImageDataGenerator(
    rotation_range=5,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.2,
    # horizontal_flip=True, # we do it on variation inverting steer
    fill_mode='nearest'
)


def invariant(img, steer):
    return img[60:-20, :], steer  # no variation


def flipped(img, steer):
    cropped, steer = invariant(img, steer)
    return np.fliplr(cropped), -steer


variants = (
    invariant,
    flipped  # flipped horizontally
)


def img_set_generator_factory(selected_set, image_shape=(80, 320, 3),
                              batch_size=128):
    """ This Keras generator takes a selected_set
    (that is groups of valid center-left-right paths with steers)
    and create a batch for the fit function
    """
    batch_x = np.zeros((batch_size,) + image_shape)
    batch_y = np.zeros((batch_size), dtype=np.float32)
    s = selected_set
    index = 0
    while 1:
        s.images, s.steers = shuffle(s.images, s.steers)  # shuffle on every loop

        for img_path, steer in zip(s.images, s.steers):
            img = mpimg.imread(img_path)  # img

            # we fill the batch with all the possible variants of the image:
            #   currently the straight and the hflipped version
            #   cutting the original image to avoid the car hood and the top sky
            chosen_variant = random.choice(variants)
            x, y = chosen_variant(img, steer)
            batch_x[index], batch_y[index] = (x, y)  # add the undeformed image
            index += 1

            if index >= batch_size:
                yield batch_x, batch_y
                index = 0


def show_variations():
    from images import get_sample_set
    sample_set = get_sample_set()
    selected_set = sample_set
    rnd = random.randint(0, len(selected_set) - 1)
    cutSet = selected_set[rnd:rnd + 1]
    print(cutSet)
    for batch_x, batch_y in img_set_generator_factory(cutSet,
                                                      # the batch return 1 image with all variations
                                                      batch_size=1):

        for x, y in zip(batch_x, batch_y):
            plt.imshow(x)
            print(y)
            plt.axis('off')
            plt.show()
        break


if __name__ == '__main__':
    show_variations()
