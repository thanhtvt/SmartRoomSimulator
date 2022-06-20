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

**The environmental noise** were gathered from ESC-50 dataset [[3]](#3). However, we only focus on 20 classes which we believe are the most relevant to daily environmental noise. These classes are: 

|                 |   |             |   |                  |   |
|-----------------|---|-------------|---|------------------|---|
| vacuum cleaner  | <img src="source/vaccum-cleaner.jpg" height="100"/>  | engine      |  <img src="source/engine.jpg" height="100"/> | keyboard typing  | <img src="source/keyboard.jpg" height="100"/> |
| fireworks       | <img src="source/firework.jpg" height="100"/>  | mouse click | <img src="source/mouse-click.png" height="100"/>  | footsteps        | <img src="source/footsteps.jpg" height="100"/>  |
| clapping        | <img src="source/clapping.jpg" height="100"/> | clock alarm | <img src="source/clock-alarm.jpg" height="100"/>  | car horn         | <img src="source/car-horn.jpg" height="100"/>  |
| door wood knock | <img src="source/knock.jpg" height="100"/>  | wind        | <img src="source/wind.jpg" height="100"/>  | drinking sipping | <img src="source/drinking-sipping.jpg" height="100"/>  |
| washing machine | <img src="source/washing-machine.jpeg" height="100"/> | rain        | <img src="source/rain.png" height="100"/>  | rooster          | <img src="source/rooster.jpg" height="100"/>  |
| snoring         | <img src="source/snoring.jpg" width="100"/> | breathing   | <img src="source/breathing.jpg" height="100"/>  | toilet flush     | <img src="source/toilet-flush.jpg" height="100"/>  |
| clock tick      | <img src="source/clock-tick.jpg" height="100"/>  | laughing    | <img src="source/laughing.jpg" height="100"/>  |                  |   |

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
