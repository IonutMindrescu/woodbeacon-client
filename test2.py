import numpy as np
import sounddevice
import librosa.display, os, gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from keras.models import model_from_json, load_model
from keras.preprocessing.image import img_to_array, load_img

duration = 1
sample_rate = 48000

def get_sound():
    # import ipdb; ipdb.set_trace()
    # duration = 1
    # sample_rate=48000
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

model = load_model('test_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#img = img_to_array(load_img('rec.png', target_size=(100, 100)))
#img = np.expand_dims(img, axis=0)
#classes = model.predict_classes(img, batch_size=10)

# print(classes)

def main():
    number_of_detection = 0
    while True:
        get_sound()
        img = img_to_array(load_img('rec.png', target_size=(100, 100)))
        img = np.expand_dims(img, axis=0)
        predict = model.predict_classes(img, batch_size=10)
        # print(predict)
        if predict[0][0] == 1:
            number_of_detection += 1
        else:
            number_of_detection = 0

        if number_of_detection == 2:
            print("Pidaras in padure")
            number_of_detection = 0

if __name__ == "__main__":
    main()
