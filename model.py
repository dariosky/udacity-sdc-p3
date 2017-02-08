#!/usr/bin/env python3

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, model_from_json

from images import *
from normalization import img_set_generator_factory


def save_model(model, filename="model"):
    # serialize model to JSON
    model_json = model.to_json()
    with open("%s.json" % filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5" % filename)
    print("Saved model to disk")


def load_model(filename="model"):
    try:
        # load json and create model
        json_file = open('%s.json' % filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("%s.h5" % filename)
        print("Loaded model from disk")
        return model
    except Exception as e:
        print(e)


def get_model(DO_TRAIN_MODEL=False, POOLING=True):
    """
    :type DO_TRAIN_MODEL: bool  when true, the model is also trained (even if it already exists)
    :type POOLING: bool when true we add the POOLING layes after the convnet
    """
    image_shape = (90, 320, 3)
    model = load_model()
    if not model:
        # ************  MODEL DEFINITION ************
        print("Building model from scratch")

        # This is based on the NVIDIA paper - adapted with pooling and dropout
        model = Sequential()

        # Normalization layers
        # Normalize values in the range [-0.5, 0.5]
        model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=image_shape))

        # 5 convolutional layers ***
        model.add(Convolution2D(24, 5, 5))
        model.add(Activation('relu'))
        if POOLING: model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(36, 5, 5))
        model.add(Activation('relu'))
        if POOLING: model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(48, 5, 5))
        model.add(Activation('relu'))
        if POOLING: model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))

        model.add(Dropout(0.2))
        model.add(Flatten())

        # 5 dense layers ***

        model.add(Dense(1164))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('relu'))

        model.add(Dense(1))  # it ends up to a single float value

        DO_TRAIN_MODEL = True

    if DO_TRAIN_MODEL:
        selected = (
            get_sample_set()
            # get_training_set() +
            # get_refinement_set_1() +
            # get_refinement_set_2()
        )

        # ****************** TRAINING ***************
        model.compile(
            optimizer='adam',
            loss='mse'
        )

        print("Training from %s" % selected)
        tot_training_samples = len(selected)
        nb_val_samples = tot_training_samples * 20 // 100
        print("Using validation set of %d" % nb_val_samples)
        # split validation set and training set
        selected.shuffle()
        selected, validation_set = selected[:-nb_val_samples], selected[-nb_val_samples:]

        num_epochs = 2
        for epoch in range(num_epochs):
            # the generator produce 8 variations for every single image
            model.fit_generator(generator=img_set_generator_factory(selected, batch_size=8),
                                validation_data=img_set_generator_factory(validation_set,
                                                                          batch_size=8),
                                nb_val_samples=len(validation_set) * 8,
                                samples_per_epoch=len(selected) * 8,
                                nb_epoch=1)
            save_model(model)  # let's save on every epoch
    return model


if __name__ == '__main__':
    # get_model load the model from disk (if any) or generate one from scratch and train it
    # if DO_TRAIN_MODEL is passed, training will be done on every call (useful for refinement)

    model = get_model(DO_TRAIN_MODEL=True)
