from typing import List

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer


class Augmentor(PreprocessingLayer):

    def __init__(self, augmentors: List[Layer], name='augmentor', **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.augmentors = augmentors

    def call(self, feature: tf.Tensor) -> tf.Tensor:
        for augmentor in self.augmentors:
            feature = augmentor(feature)
        return feature


class FrequencyMask(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 frequency_masking_para: int = 27,
                 frequency_mask_num: int = 2,
                 name='freq_mask',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.frequency_masking_para = frequency_masking_para
        self.frequency_mask_num = frequency_mask_num

    def call(self, feature: tf.Tensor) -> tf.Tensor:
        augmented_feature = feature
        for _ in range(self.frequency_mask_num):
            augmented_feature = tfio.audio.freq_mask(augmented_feature, param=self.frequency_masking_para)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_feature,
                       lambda: feature)


class TimeMask(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 time_masking_para: float = 10,
                 time_mask_num: int = 1,
                 name='time_mask',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.time_masking_para = time_masking_para
        self.time_mask_num = time_mask_num

    def call(self, feature: tf.Tensor) -> tf.Tensor:
        augmented_feature = feature
        for _ in range(self.time_mask_num):
            augmented_feature = tfio.audio.time_mask(augmented_feature, param=self.time_masking_para)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_feature,
                       lambda: feature)
