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

data = {"action": "sound-detected", "battery": 27, "sound": "drujba", "lat": 47.640173, "lng": 26.258861}

D = Dragino("dragino.ini", logging_level=logging.DEBUG)
D.join()
while not D.registered():
    print("Waiting")
    sleep(2)
#sleep(10)

duration = 3 #seconds
sample_rate=22400 #frequency

chainsaw_model = AudioModel("chainsaw.json", "chainsaw.h5")

def main():
    number_of_detection = 0
    while True:
        get_sound()
        img = img_to_array(load_img('rec.png', target_size=(100, 100)))
        img = np.expand_dims(img, axis=0)
        predict = chainsaw_model.predict(img)[0][0]
        print(predict)
        if predict >= 0.999: #predict == 1.0:
             number_of_detection += 1
        if number_of_detection == 2:
            D.send(json.dumps(data))
            print("Beacon has been sent!")
            number_of_detection = 0

if __name__ == "__main__":
    main()
