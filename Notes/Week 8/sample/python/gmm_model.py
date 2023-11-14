import os
import numpy as np
from GMMBayes import GMMBayes

features_dir = '../features/'
speaker_folders = [f.path for f in os.scandir(features_dir) if f.is_dir()]

training_mfccs = []
training_labels = []
testing_files = []
testing_labels = []

# prepare data
speaker_count = 0
for speaker in speaker_folders:
    print('Preparing data from: '+speaker)
    file_paths = [f.path for f in os.scandir(speaker) if f.path.endswith('.npy')]
    file_count = 0
    for file in file_paths:
        mfccs = np.load(file)
        labels = np.zeros((len(mfccs),1))
        labels[:] = speaker_count
        if file_count<2:
            training_mfccs.extend(mfccs)
            training_labels.extend(labels)
        else:
            testing_files.append(file)
            testing_labels.append(speaker_count)
        file_count+=1
    speaker_count+=1

print(testing_files)
print(testing_labels)

# training
gmm_nb = GMMBayes(256) # 128 components per class, can be changed
gmm_nb.fit(np.asarray(training_mfccs), np.ravel(training_labels))

# testing
correct_results = 0
test_file_count = 0
for file in testing_files:
    file_label = testing_labels[test_file_count]
    mfccs = np.load(file)
    #classify each mfcc in this file
    predictions = (gmm_nb.predict(np.asarray(mfccs))).tolist()
    # get most common prediction (majority vote) over whole utterance
    most_frequent = max(set(predictions), key = predictions.count)
    if most_frequent == file_label:
        correct_results+=1
    test_file_count+=1

accuracy = correct_results/len(testing_labels)
print(accuracy)

