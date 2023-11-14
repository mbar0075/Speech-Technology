import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

data_dir = '../data/'
speaker_folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]

for speaker in speaker_folders:
    features_folder = speaker.replace('data','features')
    if not os.path.exists(features_folder):
        os.mkdir(features_folder)

    file_paths = [f.path for f in os.scandir(speaker) if f.path.endswith('.wav')]
    for wav_file in file_paths:
        (rate, sig) = wav.read(wav_file)
        mfcc_feat = mfcc(sig, rate, nfft=1024)
        feature_path = wav_file.replace('data','features')
        feature_path = feature_path.replace('.wav', '.npy')
        np.save(feature_path, mfcc_feat)