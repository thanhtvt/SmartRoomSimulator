import os
import argparse
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml


parser = argparse.ArgumentParser(description="Train CTC model for specific voice commands task")
parser.add_argument('config_file', type=str, help="Path to a yaml file using the extended YAML syntax")
parser.add_argument('-d', '--devices', default="-1", type=str, help="Devices for training, separated by comma")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices


def train(config_file):
    with open(config_file, 'r') as f:
        config = load_hyperpyyaml(f)
        model = config['model']
        model.summary()
        train_loader = config['train_loader']
        val_loader = config['val_loader']
        trainer = config['trainer']
        trainer.train(train_loader, val_loader)


if args.devices != '-1' and len(args.devices.split(',')) > 1:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        train(args.config_file)
else:
    train(args.config_file)
