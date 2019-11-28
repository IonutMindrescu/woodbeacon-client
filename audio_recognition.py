import numpy as np
import sounddevice
import librosa.display, os, gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from keras.models import model_from_json, load_model
from keras.preprocessing.image import img_to_array, load_img

# Configurations Variables
duration = 1
sample_rate = 48000

def get_sound():
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
