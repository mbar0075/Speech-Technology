# Speech Technology

<!-- <center>
<p align='center'>

<h1>
Deep Learning for Speaker Identification: Architectural Insights from AB-1
Corpus Analysis and Performance Evaluation 
</h1> -->

<!-- </p>
</center> -->
<p align="justify">

This repository contains all the code and findings from the research study `"Deep Learning for Speaker Identification: Architectural Insights from AB-1 Corpus Analysis and Performance Evaluation"`.

## Abstract
*In the fields of security systems, forensic investigations, and personalized services, the importance of speech as a fundamental human input outweighs text-based interactions. This research delves deeply into the complex field of
Speaker Identification (SID), examining its essential components and emphasising Mel Spectrogram and Mel Frequency Cepstral Coefficients (MFCC) for feature extraction. Moreover, this study evaluates six slightly distinct
model architectures using extensive analysis to evaluate their performance, with hyperparameter tuning applied to the best-performing model. This work performs a linguistic analysis to verify accent and gender accuracy, in addition to bias evaluation within the AB-1 Corpus dataset.*


## Introduction

Speaker identification (SID) involves determining a speaker’s identity from an audio sample within a known pool of speakers. Its applications span various fields like forensics, security, and customization, approached as a pattern recognition challenge. The SID pipeline relies on two core components: feature extraction and feature classification, collaborating to assign an input speech segment to one of N known enrolled speakers.

Feature extraction involves deriving specific characteristics necessary for individual identification from voice data. In the subsequent feature classification stage, features obtained from an unidentified individual are compared with those of known speakers to detect unique qualities, leading to speaker identification. An ideal speaker identification system aims for minimal intra- and inter-speaker variance, easily quantifiable attributes, noise resilience, accurate imitation detection, and independence from other qualities.

### Dataset

The dataset utilized for this project was the `Accents of the British Isles (ABI-1) Corpus`, organized into 14 folders, each representing data from a distinct accent:

- BRM (Birmingham)
- CRN (Cornwall)
- EAN (East Anglia)
- EYK (East Yorkshire)
- GLA (Glasgow)
- ILO (Inner London)
- LAN (Lancashire)
- LVP (Liverpool)
- NCL (Newcastle)
- NWA (Northern Wales)
- ROI (Republic of Ireland)
- SHL (Scottish Highlands)
- SSE (Standard Southern English)
- ULS (Ulster)

Each accent folder contained data from approximately `20 speakers`, categorized into separate male/female folders. The folder names served as unique IDs for the respective speakers. Each speaker's folder contained multiple .WAV files along with transcriptions in .TRS or .TXT formats. However, for the assignment's purposes, the transcriptions were not required. Specifically, only the .WAV files labeled as "shortpassage" in their filenames were utilized. These .WAV files were of significant duration, approximately 40-60 seconds each, featuring the speaker reading a passage. The dataset was split into training, validation, and test sets, with a `60:20:20 ratio`, respectively.

## Feature Extraction

Feature extraction plays a critical role in transforming raw audio data into meaningful features for speech analysis. Mel Spectrogram and Mel Frequency Cepstral Coefficients (MFCC) are widely used techniques in this domain. The Mel Spectrogram visualizes the frequency content of an audio source over time by translating frequencies into the Mel scale, emphasizing human auditory perception. In contrast, MFCCs extract concise spectral features from the audio signal. This study evaluates both feature extraction methods as inputs to developed system architectures, aiming to comprehensively assess their performance and suitability within the context.

<table>
  <tr>
    <td align="center">
      <img src="Assignment\plots\mel_spectrogram.png" alt="Mel Spectrogram"  width="100%" height="auto" />
      <!-- <p><b>Mel Spectrogram Feature Extraction</b></p> -->
    </td>
    <td align="center">
      <img src="Assignment\plots\mfcc.png" alt="MFCC" width="100%" height="auto" />
      <!-- <p><b>MFCC Feature Extraction</b></p> -->
    </td>
  </tr>
</table>

## Model Architectures

This study explored a total of `six` distinct model architectures, each building upon the previous one.

### Model 1 Architecture

<p align="center">

| No.   | Layer Type         | Details              |
|-------|--------------------|----------------------|
| 1     | `Conv2D`             | (3, 3), 32 filters   |
| 2     | `ReLU`               | -                    |
| 3     | `Conv2D`             | (3, 3), 64 filters   |
| 4     | `ReLU`               | -                    |
| 5     | `MaxPooling2D`       | (2, 2)               |
| 6     | `Conv2D`             | (3, 3), 64 filters   |
| 7     | `ReLU`               | -                    |
| 8     | `MaxPooling2D`       | (2, 2)               |
| 9     | `Reshape`            | -                    |
| 10    | `LSTM`               | 64 units             |
| 11    | `Flatten`             | -                    |
| 12    | `BatchNormalization` | -                    |
| 13    | `Dropout`            | 30%                  |
| 14    | `Dense`              | 285 neurons          |
| 15    | `Softmax`            | 285 outputs          |

</p>

The first architecture combines elements from various sources, incorporating early feature extraction convolution layers and a modified lightweight LSTM layer. Batch normalization and dropout layers were added to regularize the LSTM output, improving model robustness and preventing overfitting. The final dense layer activated softmax to generate probability scores for each of the `285 speakers` in the AB-1 corpus dataset.

### Subsequent Architectures

The subsequent architectures in this research build upon the preceding model, introducing modifications to enhance feature extraction capabilities, sequence comprehension, and model complexity while addressing overfitting issues.


## Evaluation

TensorFlow Keras was utilized for training due to its user-friendly interface and effective neural network creation. The Adam optimizer, known for its versatility, facilitated training. An early stopping callback with a patience score of `5` was implemented to prevent `overfitting`.

<p align='center'><img src="Assignment\plots\Table of Results.png" alt="Results"/></p>

Models 1 and 5 demonstrated superior performance in test accuracy, precision, recall, and F-score compared to others. However, model 2, employing the MFCC feature extractor, consistently yielded comparatively poor results. Models utilizing the Mel Spectrogram feature extractor outperformed MFCC-based models consistently. Performance metrics ranged between `0.8` and `0.97`, indicating substantial model performance, with test loss ranging between `0.14` and `0.65`, indicating avoidance of overfitting.

Following the discovery of model 1’s superior efficiency, a comprehensive hyperparameter tuning process involving fifteen trials fine-tuned its parameters. The best hyperparameters found were a learning rate of `0.001`, a dropout rate of `0.4`, and the application of the `tanh` activation function specifically to the second layer in the model architecture of model 1, while the remaining layers utilized `relu`.

<p align='center'><img src="Assignment\plots\best_model_mel_accuracy.png" alt="Best Model Training"/></p>

The findings highlighted a notable performance improvement achieved by optimizing the best model with these refined parameters compared to the original Model 1.

### Best Model Architecture

<p align="center">

| No.   | Layer Type         | Details              |
|-------|--------------------|----------------------|
| 1     | `Conv2D`             | (3, 3), 32 filters   |
| 2     | `Tanh`               | -                    |
| 3     | `Conv2D`             | (3, 3), 64 filters   |
| 4     | `ReLU`               | -                    |
| 5     | `MaxPooling2D`       | (2, 2)               |
| 6     | `Conv2D`             | (3, 3), 64 filters   |
| 7     | `ReLU`               | -                    |
| 8     | `MaxPooling2D`       | (2, 2)               |
| 9     | `Reshape`            | -                    |
| 10    | `LSTM`               | 64 units             |
| 11    | `Flatten`             | -                    |
| 12    | `BatchNormalization` | -                    |
| 13    | `Dropout`            | 40%                  |
| 14    | `Dense`              | 285 neurons          |
| 15    | `Softmax`            | 285 outputs          |


</p>

## Best Model Evaluation

The model showcased here delivers exceptional performance, achieving an impressive accuracy of `97.09%` on the test set, accompanied by a low loss of `0.136`. Precision, a measure of the accuracy of positive predictions, stands at `97.38%`, while recall, indicating the model's ability to find all positive instances, is at `97.09%`. The harmonic mean of precision and recall, the F1 score, is calculated at `97.10%`. These metrics collectively demonstrate the model's robustness in making accurate predictions and its ability to effectively capture relevant patterns in the data. 

<table>
  <tr>
    <td align="center">
      <img src="Assignment\plots\best_model_mel_metrics_top_20_speakers.png" alt="Best Model CM Top 20"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assignment\plots\best_model_mel_metrics_bottom_20_speakers.png" alt="Best Model CM Bottom 20" width="100%" height="auto" />
    </td>
  </tr>
</table>


## Analysis

A linguistic analysis was performed on the top-performing model to assess its gender and accent correctness while ensuring neutrality and the absence of bias towards specific accents or genders. The assessment showed a slight variance in gender accuracy, with slightly better predictions for female speakers by a margin of `0.02`. However, this indicates gender equality in the model's predictions, suggesting unbiased representation within the dataset.

Regarding accent accuracy, there was a more noticeable difference between the best and lowest accuracy, approximately `0.05`. Standard Southern English, Scottish Highlands, and East Anglia were the easiest accents to predict, while Newcastle, Northern Wales, and Cornwall were more challenging for the model.

<table>
  <tr>
    <td align="center">
      <img src="Assignment\plots\best_model_mel_dataset_evaluation_accents.png" alt="Accent accuracy"  width="100%" height="auto" />
      <!-- <p><b>Accent accuracy and bias evaluation</b></p> -->
    </td>
    <td align="center">
      <img src="Assignment\plots\best_model_mel_dataset_evaluation_genders.png" alt="Gender accuracy" width="100%" height="auto" />
      <!-- <p><b>Gender accuracy and bias evaluation</b></p> -->
    </td>
  </tr>
</table>

## Conclusion

This study investigated the effectiveness of Mel Spectrogram and MFCC as feature extraction methods for Speaker Identification (SID) and proposed robust model architectures tailored for SID. The best model exhibited significant accuracy, precision, recall, and F1-score of 0.97, highlighting the efficacy of these techniques. Gender analysis revealed minimal variation, confirming dataset balance and the absence of gender bias in predictions. However, accent accuracy showed more noticeable differences, emphasizing the necessity for further analysis and model refinement, particularly in addressing accent-related challenges in SID.

</p>