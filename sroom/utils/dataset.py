"""Create CsvDataset"""
import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List


def load_files(paths: List[str]):
    """
    Args:
        paths: list of root paths to load files from"""
    id2label = {
        '0' : 'xoay ghế trái',
        '1' : 'xoay ghế phải',
        '2' : 'bật đèn',
        '3' : 'bật đèn lên',
        '4' : 'tắt đèn',
        '5' : 'tắt đèn đi',
        '6' : 'sáng quá',
        '7' : 'tối quá',
        '8' : 'bật nhạc',
        '9' : 'bật nhạc lên',
        '10': 'dừng nhạc',
        '11': 'chuyển nhạc',
        '12': 'bật màn hình',
        '13': 'tắt màn hình',
        '14': 'bật laptop',
        '15': 'tắt laptop',
        '16': 'bật tv',
        '17': 'tắt tv',
    }

    data_input = {
        'filepath': [],
        'transcription': [],
    }

    classes = os.listdir(paths[0])
    for path in tqdm(paths, desc='Loading files'):
        for class_ in classes:
            wavdir = os.path.join(path, class_)
            wavpaths = [os.path.join(wavdir, filepath) for filepath in os.listdir(wavdir) if filepath.endswith('.wav')]
            data_input['filepath'].extend(wavpaths)
            data_input['transcription'].extend([id2label[class_]] * len(wavpaths))

    df = pd.DataFrame.from_dict(data_input).sort_values(by='transcription', ignore_index=True)
    return df


def split_df(df, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Split dataframe into train, val, test with given distribution of labels
    """
    df_labels = {'label': [], 'df': [], 'amount': []}
    label_counts = df.transcription.value_counts()
    for label, df_label in df.groupby('transcription'):
        df_labels['label'].append(label)
        df_labels['df'].append(df_label.sample(frac=1).reset_index(drop=True))
        df_labels['amount'].append(label_counts[label])

    df_train = pd.DataFrame(columns=['filepath', 'transcription'])
    df_val = pd.DataFrame(columns=['filepath', 'transcription'])
    df_test = pd.DataFrame(columns=['filepath', 'transcription'])

    for idx in range(len(df_labels['df'])):
        num_train = int(train_ratio * df_labels['amount'][idx])
        num_val = int(val_ratio * df_labels['amount'][idx])
        df_label = df_labels['df'][idx]

        df_train = pd.concat([df_train, df_label[:num_train]])
        df_val = pd.concat([df_val, df_label[num_train:(num_train + num_val)]])
        df_test = pd.concat([df_test, df_label[(num_train + num_val):]])

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_val = df_val.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    return df_train, df_val, df_test


def create_dataset(paths: List[str], data_dir='data'):
    df = load_files(paths)
    df_train, df_val, df_test = split_df(df)
    os.makedirs(data_dir, exist_ok=True)

    df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    print(f'Dataset created at {data_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('-p', '--paths', nargs='+', help='Folder paths to load files from')
    parser.add_argument('-d', '--data_dir', default='data', help='Folder path to save dataset')
    args = parser.parse_args()

    create_dataset(args.paths, args.data_dir)