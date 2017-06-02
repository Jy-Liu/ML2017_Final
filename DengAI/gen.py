import sys, os
import argparse
import numpy as np
from utils import DataReader


def ensure_dir(file_path):
  directory = file_path
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


class DataProcessor:
    @classmethod
    def split_cities(cls, data, label=False, block_size=4):
        if label:
            maps = {'sj': data[data[:, 0] == 'sj'][block_size-1:, 1:],
                    'iq': data[data[:, 0] == 'iq'][block_size-1:, 1:]}
        else:
            selected_col = np.r_[1, 2, 4:data.shape[1]]
            maps = {'sj': data[data[:, 0] == 'sj'][:, selected_col],
                    'iq': data[data[:, 0] == 'iq'][:, selected_col]}
        return maps

    @classmethod
    def block_features(cls, maps, block_size=4):
        sj, iq = maps['sj'], maps['iq']
        new_sj, new_iq = [], []
        for i in range(sj.shape[0] - block_size + 1):
            buf = sj[i:i+block_size]
            new_sj.append(buf)
        maps['sj'] = np.array(new_sj)
        for i in range(iq.shape[0] - block_size + 1):
            buf = iq[i:i+block_size]
            new_iq.append(buf)
        maps['iq'] = np.array(new_iq)
        return maps

    @classmethod
    def block_testdata(cls, maps, testdata, block_size=4):
        sj, iq = maps['sj'], maps['iq']
        new_X, new_Y = [], []
        X = np.concatenate((sj[-(block_size-1):, :], testdata['sj']), axis=0)
        for i in range(X.shape[0] - block_size + 1):
            buf = X[i:i+block_size]
            new_X.append(buf)
        Y = np.concatenate((iq[-(block_size-1):, :], testdata['iq']), axis=0)
        for i in range(Y.shape[0] - block_size + 1):
            buf = Y[i:i+block_size]
            new_Y.append(buf)
        ret = {'sj': np.array(new_X), 'iq': np.array(new_Y)}
        return ret


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_f', help='training features data path')
  parser.add_argument('--train_l', help='training labels data path')
  parser.add_argument('--test', help='test data path')
  args = parser.parse_args()

  features_path = args.train_f
  labels_path = args.train_l
  tests_path = args.test

  features = DataReader.read_features(features_path)
  labels = DataReader.read_labels(labels_path)
  tests_data = DataReader.read_features(tests_path)

  features_map = DataProcessor.split_cities(features)
  labels_map = DataProcessor.split_cities(labels, label=True)
  test_map = DataProcessor.split_cities(tests_data)

  test_map = DataProcessor.block_testdata(features_map, test_map)
  features_map = DataProcessor.block_features(features_map)

  new_train_features_path = 'new_features_train.npz'
  new_train_labels_path = 'new_labels_train.npz'
  new_test_features_path = 'new_features_test.npz'

  data_path = 'new_data'
  ensure_dir(data_path)
  np.savez(os.path.join(data_path, new_train_features_path),
            sj=features_map['sj'], iq=features_map['iq'])
  np.savez(os.path.join(data_path, new_train_labels_path),
            sj=labels_map['sj'], iq=labels_map['iq'])
  np.savez(os.path.join(data_path, new_test_features_path),
            sj=test_map['sj'], iq=test_map['iq'])

if __name__ == '__main__':
  main()