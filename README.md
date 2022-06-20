# Smart Room Simulator
As technology develops, smart home solutions are increasingly improved and more modern. Including Vietnamese
voice control solution for smart home devices. We decided to build a website to demo the voice-controlled virtual home that looks like the image below
![9f4e1d9253f493aacae5](https://user-images.githubusercontent.com/103128064/174653290-d2734885-e941-401e-817f-5189f09d256c.jpg)
This repository is our assignment for Course: Speech Processing (INT3411 20), where we attempt to use CTC [1] for Speech recognition task and deploy a web  3D application.

Table of Contents
================
* [Abstract](#abstract)
* [Dataset](#dataset)
* [Training](#training)
* [Experiment tracking & Tuning hyperparameter](#experiment-tracking-&-tuning-hyperparameter)
* [Result](#result)
* [Deployment](#deployment)
    * [Streamlit share](#streamlit-share)
    * [Localhost run](#Localhost-run)
* [Reference](#reference)

Abstract
========
Our main goal is to recognize commands and then manipulate objects in the virtual room.
 
Dataset
=======

Audios have many different ways to be represented, going from raw time series to time-frequency decompositions. By representing with Spectrogram which consist of 2D images representing sequences of Short Time Fourier Transform (STFT) with time and frequency as axes, and brightness representing the strength of a frequency component at each time frame, the input of the model will be noisy voice spectrogram and the grouth truth will be clean voice spectrogram.


**The clean voices** were approximately 10 hours of reading Vietnamese commands by us, student of Speech Processing Course at UET. 

**Recording rules

- Record in a quiet environment.
- File extension .wav
- Sample rate: 16kHz
- Bit-depth: 16-bit
- Channel: 1 (mono)
- Maximum recording time 2s.
- Say each sentence 100 times
**Naming rules:** <speaker_name>_<sentence id>_<recording>.wav
Example: Thanh recorded the verse "Turn on the music" for the 12th time: `Thanh_10_12.wav`
   
We generate data and label it ourselves and do data cleaning, from audios to spectrograms. Audios were sampled at 16kHz and we extracted windows slighly above 2 second. Noises have been blended to clean voices with a randomization of the noise level (between 20% and 80%). 

Training
========


Experiment tracking & Tuning hyperparameter
==================

Result
======


Deployment
=========



Team member
===========
Tran Van Trong Thanh: 
- Record and label
- Clean data and preprocess
- Data Augmentation
- Tracking and tuning model CTC

Tran Khanh Hung: 
- Create date and label
- Clean data and preprocess
- Tranform data
- Build baseline model CTC(https://www.kaggle.com/code/hngtrnkhnh/notebook35b94de2ec/notebook)


Reference
============
<a id="1">[1]</a> 
Sequence Modeling with CTC: https://distill.pub/2017/ctc/

<a id="2">[2]</a> 
Connectionist Temporal Classification: https://www.aiourlife.com/2020/04/connectionist-temporal-classification.html

<a id="3">[3]</a> 
An Intuitive Explanation of Connectionist Temporal Classification: https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
  
<a id="4">[4]</a> 
Speech Recognition - Deep Speech, CTC, Listen, Attend and Spell: https://jonathan-hui.medium.com/speech-recognition-deep-speech-ctc-listen-attend-and-spell-d05e940e9ed1
