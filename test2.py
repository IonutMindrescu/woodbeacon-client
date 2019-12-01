import numpy as np
import sounddevice
import librosa.display, os, gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from keras.models import model_from_json, load_model
from keras.preprocessing.image import img_to_array, load_img

from audio_recognition import *

model = load_model('test_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def main():
    number_of_detection = 0
    while True:
        sound = get_sound()
        if not sound:
            continue
        img = img_to_array(load_img('rec.png', target_size=(100, 100)))
        img = np.expand_dims(img, axis=0)
        predict = model.predict_classes(img, batch_size=10)
        if predict[0][0] == 1:
            number_of_detection += 1
        else:
            number_of_detection = 0

        if number_of_detection == 2:
            print("Chainsaw detected")
            number_of_detection = 0

if __name__ == "__main__":
    main()
