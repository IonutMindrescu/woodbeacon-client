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

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, load_img

duration = 3
sample_rate=48000

def get_sound():
    # import ipdb; ipdb.set_trace()
    audio = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    audio = np.squeeze(audio)
    sr=sample_rate
    sounddevice.wait()
    S = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis("off")
    ax.axis("tight")
    plt.margins(0)
    librosa.display.specshow(log_S, sr=sr)
    fig.savefig('rec.png', dpi=100, pad_inches=0)
    plt.close(fig)
    plt.close('all')
    del audio, S, log_S, ax, fig

def extract_spectrogram(fname, iname):
    audio, sr = librosa.load(fname, res_type='kaiser_fast')
    S = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis("off")
    ax.axis("tight")
    plt.margins(0)
    librosa.display.specshow(log_S, sr=sr)
    fig.savefig(iname, dpi=100, pad_inches=0)
    plt.close(fig)
    plt.close('all')
    del audio, S, log_S, ax, fig

class AudioModel(object):

    EMOTIONS_LIST = ['CHAINSAW']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict(self, img):
        self.preds = self.loaded_model.predict(img)
        # return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]np.argmax(
        return self.preds

model = AudioModel("chainsaw.json", "chainsaw.h5")
GPIO.setwarnings(False)
data = {"action": "sound-detected", "battery": 27, "sound": "chainsaw", "lat": 47.640173, "lng": 26.258861}
D = Dragino("dragino.ini", logging_level=logging.DEBUG)
D.join()
while not D.registered():
    print("Waiting")
    sleep(2)

def main():
    number_of_detection = 0
    while True:
        get_sound()
        img = img_to_array(load_img('rec.png', target_size=(100, 100)))
        img = np.expand_dims(img, axis=0)
        predict = model.predict(img)[0][0]
        print(predict)
        if predict >= 0.9999 and predict <= 1.0:
            print('Pidaras in padure')
            D.send(json.dumps(data))
            print("Beacon has been sent!") 
            number_of_detection = 0

if __name__ == "__main__":
    main()




