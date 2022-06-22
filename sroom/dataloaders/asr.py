import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers.experimental.preprocessing import PreprocessingLayer


class Dataloader(tf.data.Dataset):

    def __new__(
        cls, data: tf.data.Dataset,
        text_encoder: PreprocessingLayer,
        audio_encoder: PreprocessingLayer = None,
        audio_augmentor: PreprocessingLayer = None,
        feature_augmentor: PreprocessingLayer = None,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
        shuffle: bool = True,
        shuffle_buffer_size: int = 128,
        use_norm: bool = False,
        num_repeat: int = 1,
        batch_size: int = 32,
        drop_remainder: bool = True,
        name: str = 'dataloader',
    ):
        if num_repeat > 1:
            data = data.repeat(num_repeat)
            
        if shuffle:
            data = data.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
        data = data.map(
            lambda wavpath, label: (tf.io.read_file(wavpath), label),
            num_parallel_calls=num_parallel_calls,
        )
        data = data.map(
            lambda audio, label: (tf.audio.decode_wav(audio, desired_channels=1)[0], label),
            num_parallel_calls=num_parallel_calls,
        )
        data = data.map(
            lambda audio, label: (tf.squeeze(audio, axis=-1), label),
            num_parallel_calls=num_parallel_calls,
        )
        
        if audio_augmentor:
            data = data.map(
                lambda audio, label: (audio_augmentor(audio), label),
                num_parallel_calls=num_parallel_calls,
            )

        if use_norm:
            data = data.map(
                lambda audio, label: (normalize_audio(audio), label),
                num_parallel_calls=num_parallel_calls,
            )
        
        if audio_encoder:
            data = data.map(
                lambda audio, label: (audio_encoder(audio), label),
                num_parallel_calls=num_parallel_calls,
            )
            audio_padded_shape = [None, audio_encoder.num_mel_bins]
        else:
            audio_padded_shape = [None]
        
        if feature_augmentor:
            data = data.map(
                lambda audio, label: (feature_augmentor(audio), label),
                num_parallel_calls=num_parallel_calls,
            )
        
        data = data.padded_batch(
            batch_size,
            padded_shapes=(audio_padded_shape, []),
            padding_values=(0.0, None),
            drop_remainder=drop_remainder,
        )

        data = data.map(
            lambda audio, label: (audio, text_encoder(label)),
            num_parallel_calls=num_parallel_calls,
        )

        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        data.name = name
        return data


def normalize_audio(audio: tf.Tensor) -> tf.Tensor:
    return (audio - tf.reduce_mean(audio)) / tf.sqrt(tf.math.reduce_variance(audio) + 1e-8)
