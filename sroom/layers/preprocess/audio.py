import tensorflow as tf
import tensorflow_text as tftext


class FBank(tf.keras.layers.preprocessing.PreprocessingLayer):
    def __init__(
        self,
        use_stack: bool = True,
        window_size: int = 3,
        window_step: int = 3,
        num_mel_bins: int = 80,
        sample_rate: int = 16000,
        frame_ms: int = 25,
        stride_ms: int = 10,
        lower_edge_hertz: int = 0,
        upper_edge_hertz: int = 8000,
        name="fbank",
        **kwargs,
    ):
        super().__init__(trainable=False, dynamic=False, name=name, **kwargs)
        self.use_stack = use_stack
        self.window_size = window_size
        self.window_step = window_step
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz

    def call(self, audio: tf.Tensor) -> tf.Tensor:
        log_mel_spectrograms = compute_fbanks(
            audio,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz
        )

        return log_mel_spectrograms


def compute_fbanks(audio: tf.Tensor,
                   frame_length: int = 512,
                   frame_step: int = 160,
                   sample_rate: int = 16000,
                   num_mel_bins: int = 80,
                   lower_edge_hertz: int = 0,
                   upper_edge_hertz: int = 8000) -> tf.Tensor:
    stfts = tf.signal.stft(
        audio, frame_length=frame_length,
        frame_step=frame_step, pad_end=True
    )
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = tf.shape(stfts)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate,
        lower_edge_hertz, upper_edge_hertz)

    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return log_mel_spectrograms
