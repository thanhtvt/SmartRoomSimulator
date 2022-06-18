import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank: int = 0,
        name='ctc_loss',
        **kwargs
    ):
        super(CTCLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kwargs)
        self.blank = blank

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        pass
