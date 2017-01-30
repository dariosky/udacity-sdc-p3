#!/usr/bin/env python3
import argparse
import base64
from io import BytesIO
import statistics
import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask
from keras.models import model_from_json

from normalization import invariant, flipped

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
app.debug = True
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle_prev = float(data["steering_angle"])
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    steering_angle = 0

    image_array_a, _ = invariant(image_array, steering_angle)
    transformed_image_a = image_array_a[None, :, :, :]
    # This model currently assumes that the features of the model are just the images.
    #  Feel free to change this.
    steering_angle_a = float(model.predict(transformed_image_a, batch_size=1))

    # image_array_b, _ = flipped(image_array, steering_angle)
    # transformed_image_b = image_array_b[None, :, :, :]
    # steering_angle_b = float(model.predict(transformed_image_b, batch_size=1))
    steering_angle = statistics.mean([
        steering_angle_a,
        # -steering_angle_b,
    ])

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if speed > 15 and abs(steering_angle) > 0.1:
        throttle = -2
    else:
        throttle = 0.2
    try:
        print("steer: {steering:4.3f} speed:{speed:3f} throttle:{throttle:3.1f}".format(
            steering=steering_angle,
            speed=speed,
            throttle=throttle
        ))
    except Exception as e:
        print(e)

    send_control({'steering_angle': str(steering_angle),
                  'throttle': str(throttle),
                  'brake': str(-throttle if throttle < 0 else 0),
                  })


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control({'steering_angle': str(0),
                  'throttle': str(0),
                  })


def send_control(data):
    sio.emit("steer", data=data, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        nargs='?',
                        default='model.json',
                        help='Path to model definition json.'
                             ' Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
