#!/usr/bin/env python3
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
import keras
from images import *
from normalization import img_set_generator_factory


def get_model(DO_TRAIN_MODEL=False, POOLING=True):
    """
    :type DO_TRAIN_MODEL: bool  when true, the model is also trained (even if it already exists)
    :type POOLING: bool when true we add the POOLING layes after the convnet
    """
    image_shape = (90, 320, 3)
    try:
        model = load_model('model.h5')
    except OSError:
        model = None
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
        model.add(Activation('elu'))
        if POOLING: model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(36, 5, 5))
        model.add(Activation('elu'))
        if POOLING: model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(48, 5, 5))
        model.add(Activation('elu'))
        if POOLING: model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('elu'))

        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('elu'))

        model.add(Dropout(0.2))
        model.add(Flatten())

        # 5 dense layers ***

        model.add(Dense(1164))
        model.add(Activation('elu'))
        model.add(Dense(100))
        model.add(Activation('elu'))
        model.add(Dense(50))
        model.add(Activation('elu'))
        model.add(Dense(10))
        model.add(Activation('elu'))

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
        train_set, validation_set = selected[:-nb_val_samples], selected[-nb_val_samples:]

        callbacks = [
            keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=0,
                                            save_best_only=True, save_weights_only=False,
                                            mode='auto', period=1)
        ]
        num_epochs = 2
        # the generator produce 8 variations for every single image
        model.fit_generator(generator=img_set_generator_factory(train_set, batch_size=8),
                            validation_data=img_set_generator_factory(validation_set,
                                                                      batch_size=8),
                            nb_val_samples=len(validation_set) * 8,
                            samples_per_epoch=len(train_set) * 8,
                            nb_epoch=num_epochs,
                            callbacks=callbacks
                            )
    return model


if __name__ == '__main__':
    # get_model load the model from disk (if any) or generate one from scratch and train it
    # if DO_TRAIN_MODEL is passed, training will be done on every call (useful for refinement)

    model = get_model(DO_TRAIN_MODEL=True)
