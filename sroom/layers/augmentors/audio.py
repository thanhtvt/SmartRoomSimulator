from typing import List

from librosa.effects import pitch_shift, time_stretch
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers.preprocessing import PreprocessingLayer


class Augmentor(PreprocessingLayer):

    def __init__(self, augmentors: List[Layer], name='augmentor', **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.augmentors = augmentors

    def call(self, audio: tf.Tensor) -> tf.Tensor:
        for augmentor in self.augmentors:
            audio = augmentor(audio)
        return audio


class Gain(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 min_gain: float = 10,
                 max_gain: float = 12,
                 name='gain',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.min_gain = min_gain
        self.max_gain = max_gain

    def call(self, audio: tf.Tensor) -> tf.Tensor:
        gain = tf.random.uniform(shape=(),
                                 minval=self.min_gain,
                                 maxval=self.max_gain,
                                 dtype=tf.float32)
        augmented_audio = audio * 10 ** (gain / 20)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)


class PitchShift(PreprocessingLayer):

    def __init__(
        self,
        prob: float = 0.5,
        min_semitones: int = -4,
        max_semitones: int = 4,
        sample_rate: int = 16000,
        name='pitch_shift',
    ):
        super().__init__(trainable=False, name=name)
        self.prob = prob
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.sr = sample_rate

    def call(self, audio: tf.Tensor) -> tf.Tensor:
        num_semitones = tf.random.uniform(shape=(),
                                          minval=self.min_semitones,
                                          maxval=self.max_semitones,
                                          dtype=tf.int32)
        augmented_audio = tf.numpy_function(pitch_shift_librosa, [audio, self.sr, num_semitones], tf.float32)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)


class TimeStretch(PreprocessingLayer):
    
    def __init__(
        self,
        prob: float = 0.5,
        min_speed_rate: float = 0.8,
        max_speed_rate: float = 1.1,
        name='time_stretch',
    ):
        super().__init__(trainable=False, name=name)
        self.prob = prob
        self.min_speed_rate = min_speed_rate
        self.max_speed_rate = max_speed_rate

    def call(self, audio: tf.Tensor) -> tf.Tensor:
        speed_rate = tf.random.uniform(shape=(),
                                       minval=self.min_speed_rate,
                                       maxval=self.max_speed_rate,
                                       dtype=tf.float32)
        augmented_audio = tf.numpy_function(time_stretch_librosa, [audio, speed_rate], tf.float32)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)
        

def pitch_shift_librosa(audio, sr, n_steps):
    return pitch_shift(audio, sr, n_steps)


def time_stretch_librosa(audio, stretch_factor):
    return time_stretch(audio, stretch_factor)
