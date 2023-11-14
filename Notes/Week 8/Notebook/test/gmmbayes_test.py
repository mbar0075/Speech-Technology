import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from python.GMMBayes import GMMBayes

# Load up dataset
(rate,sig) = wav.read('female.wav')
mfcc_feat_female = mfcc(sig,rate,nfft=1024)

(rate,sig) = wav.read('male.wav')
mfcc_feat_male = mfcc(sig,rate,nfft=1024)

labels_female = np.zeros((len(mfcc_feat_female),1))
labels_male = np.ones((len(mfcc_feat_male),1))

samples = np.concatenate((mfcc_feat_female,mfcc_feat_male))
labels = np.ravel(np.concatenate((labels_female,labels_male)))

# Fit the NGMM aive Bayes classifier to all original dimensions
gmm_nb = GMMBayes(128) # 128 components per class
gmm_nb.fit(samples, labels)
