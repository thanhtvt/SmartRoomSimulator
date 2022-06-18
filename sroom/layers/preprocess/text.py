import os

import tensorflow as tf
import tensorflow_text as tftext
import sentencepiece as spm


class Subword(tf.keras.layers.preprocessing.PreprocessingLayer):
    def __init__(
        self,
        model_prefix: str,
        data_path: str,
        character_coverage: float = 0.995,
        model_type: str = 'bpe',
        num_threads: int = 4,
        unk_id: int = 1,
        bos_id: int = -1,
        eos_id: int = -1,
        pad_id: int = 0,
        unk_piece: str = '<unk>',
        bos_piece: str = '<s>',
        eos_piece: str = '</s>',
        pad_piece: str = '<pad>',
        vocab_size: int = 54,
        out_type: tf.Dtype = tf.int32,
        user_defined_symbols: str = '',
        name='subword',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.model_prefix = model_prefix
        model_path = model_prefix + '.model'
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.num_threads = num_threads
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_piece = unk_piece
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece
        self.vocab_size = vocab_size
        self.user_defined_symbols = user_defined_symbols
        self.use_string = (out_type == tf.string)

        if not os.path.isfile(model_path) and data_path:
            self.adapt(data_path)

        with open(model_path, 'rb') as f:
            model = f.read()

        self.tokenizer = tftext.SentencepieceTokenizer(
            model=model, out_type=out_type, nbest_size=0, alpha=1.0,
            reverse=False, add_bos=bos_id != -1, add_eos=eos_id != -1,
            return_nbest=False, name=name
        )

    def adapt(self, data_path):
        spm.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            num_threads=self.num_threads,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            unk_piece=self.unk_piece,
            bos_piece=self.bos_piece,
            eos_piece=self.eos_piece,
            pad_piece=self.pad_piece,
            user_defined_symbols=self.user_defined_symbols
        )

    def call(self, text: tf.Tensor) -> tf.Tensor:
        sparse_tokens = self.tokenizer.tokenize(text)
        blank = self.pad_piece if self.use_string else self.pad_id
        dense_tokens = sparse_tokens.to_tensor(default_value=blank)
        return dense_tokens

    def decode(self, dense_tokens: tf.Tensor) -> tf.Tensor:
        return self.tokenizer.detokenize(dense_tokens)