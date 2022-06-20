from typing import List

import tensorflow as tf


class DeepSpeech2(tf.keras.Model):
    """Simplify DeepSpeech 2"""
    def __init__(
        self,
        input_dim: int,
        output_units: int,
        num_filters: int = 32,
        kernel_size: List[List[int]] = [[11, 41], [11, 21]],
        strides: List[List[int]] = [[2, 2], [1, 2]],
        padding: str = 'same',
        use_bias_cnn: bool = False,
        rnn_units: int = 128,
        num_rnn_layers: int = 2,
        use_bias_rnn: bool = True,
        return_sequences: bool = True,
        reset_after: bool = True,
        dropout_rate: float = 0.5,
        name: str = 'deepspeech2',
    ):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_units = output_units

        self.reshape_before_cnn = tf.keras.layers.Reshape((-1, input_dim, 1), name='expand_dims')
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size[0],
            strides=strides[0],
            padding=padding,
            use_bias=use_bias_cnn,
            name='conv1',
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size[1],
            strides=strides[1],
            padding=padding,
            use_bias=use_bias_cnn,
            name='conv2',
        )

        self.bidirectionals = []
        self.dropouts = []
        for i in range(1, num_rnn_layers + 1):
            gru = tf.keras.layers.GRU(
                units=rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=use_bias_rnn,
                return_sequences=return_sequences,
                reset_after=reset_after,
            )
            bidirectional = tf.keras.layers.Bidirectional(gru, merge_mode='concat', name=f'bidir_{i}')
            self.bidirectionals.append(bidirectional)
            if i < num_rnn_layers:
                dropout = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i}')
                self.dropouts.append(dropout)

        self.dense1 = tf.keras.layers.Dense(units=rnn_units * 2, name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=output_units, name='dense2')
        self.softmax = tf.keras.layers.Softmax()
        self.rnn_units = rnn_units

    def summary(self):
        inputs = tf.keras.Input(shape=(None, self.input_dim), batch_size=None, dtype=tf.float32)
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # CNN Module
        outputs = self.reshape_before_cnn(inputs)
        outputs = self.conv1(outputs)
        outputs = self.batch_norm1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.batch_norm2(outputs)
        outputs = self.relu(outputs)

        # RNN Module
        outputs = tf.keras.layers.Reshape((-1, outputs.shape[-1] * outputs.shape[-2]))(outputs)
        for bidirectional, dropout in zip(self.bidirectionals, self.dropouts):
            outputs = bidirectional(outputs)
            outputs = dropout(outputs)
        outputs = self.bidirectionals[-1](outputs)

        # FC Module
        outputs = self.dense1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dense2(outputs)
        outputs = self.softmax(outputs)

        return outputs
