import os

import tensorflow as tf


class Trainer():
    def __init__(
        self,
        model: tf.keras.Model,
        loss: tf.keras.losses.Loss,
        pretrained_model: str = None,
        checkpoint_path: str = 'checkpoints',
        ckpt_save_freq: str = 'epoch',
        backup_dir: str = 'backup',
        num_epochs: int = 70,
        tb_log_dir: str = 'logs',
        tb_update_freq: str = 'epoch',
        tb_profile_batch: int = 0,
    ):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model = model
        if pretrained_model:
            self.model.load_weights(pretrained_model).expect_partial()

        self.model.compile(optimizer=self.optimizer, loss=loss)
        self.num_epochs = num_epochs
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            update_freq=tb_update_freq,
            profile_batch=tb_profile_batch,
        )
        if not checkpoint_path:
            checkpoint_path = 'checkpoints/ckpt-epoch-{epoch:02d}.ckpt'

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            save_freq=ckpt_save_freq,
        )

        os.makedirs(backup_dir, exist_ok=True)
        backup_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir)

        self.callbacks = [tb_callback, checkpoint_callback, backup_callback]

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        cmvn_dataset: tf.data.Dataset = None,
    ):
        if cmvn_dataset and self.model.cmvn:
            print("Start compute cmvn...")
            self.model.adapt(cmvn_dataset, batch_size=1)
            print("Finish compute cmvn.")

        self.model.fit(
            train_dataset,
            epochs=self.num_epochs,
            validation_data=val_dataset,
            callbacks=self.callbacks,
        )
