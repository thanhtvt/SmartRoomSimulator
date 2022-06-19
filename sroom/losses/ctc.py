import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        name='ctc_loss',
        **kwargs
    ):
        super(CTCLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        batch_len = tf.shape(y_true)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype=tf.int32)
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype=tf.int23)

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss
