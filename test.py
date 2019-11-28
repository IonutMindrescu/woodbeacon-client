#!/usr/bin/env python3
"""
    Wood Beacon - Alpha 1.1a (sound recognition & alerting)
"""
import logging
from time import sleep
import RPi.GPIO as GPIO
from dragino import Dragino
import json

import numpy as np
import sounddevice
import librosa.display, os, gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from audio_recognition import *

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img

GPIO.setwarnings(False)

# JSON Data to be sent
data = {"action": "sound-detected", "battery": 27, "sound": "drujba", "lat": 47.640173, "lng": 26.258861}

D = Dragino("dragino.ini", logging_level=logging.DEBUG)
D.join()
while not D.registered():
    print("Waiting")
    sleep(2)

# Load ML model

model = load_model('test_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def main():
    number_of_detection = 0
    while True:
        get_sound()
        img = img_to_array(load_img('rec.png', target_size=(100, 100)))
        img = np.expand_dims(img, axis=0)
        predict = model.predict_classes(img, batch_size=10)t
        if predict[0][0] == 1:
            number_of_detection += 1
        else:
            number_of_detection = 0

        if number_of_detection == 2:
            D.send(json.dumps(data))
            print("Sound detected, beacon has been sent!")
            number_of_detection = 0

if __name__ == "__main__":
    main()
