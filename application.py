import logging
import json
import numpy as np
import RPi.GPIO as GPIO

from time import sleep
from dragino import Dragino
from keras.preprocessing.image import img_to_array, load_img

from audio_recognition import *


GPIO.setwarnings(False)
chainsaw_model = AudioModel("chainsaw.json", "chainsaw.h5")
D = Dragino("dragino.ini", logging_level=logging.DEBUG)
D.join()

data = {"action": "sound-detected", "battery": 27, "sound": "chainsaw", "lat": 47.640173, "lng": 26.258861}

def main():
    while not D.registered():
        print("Waiting")
        sleep(2)

    while True:
        get_sound()
        img = img_to_array(load_img('rec.png', target_size=(100, 100)))
        img = np.expand_dims(img, axis=0)
        predict = chainsaw_model.predict(img)[0][0]
        print(predict)
        if predict >= 0.9999 and predict <= 1.0:
            D.send(json.dumps(data))
            print("Beacon has been sent!")

if __name__ == "__main__":
    main()
