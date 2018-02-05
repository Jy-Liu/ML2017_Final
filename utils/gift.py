#!/usr/bin/env python
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
  gift = pickle.load(open(GIFT, 'rb'))
  train = gift['train']
  valid = gift['valid']
  train_mean = np.mean(train, axis=1)
  train_std = np.std(train, axis=1)
  valid_mean = np.mean(valid, axis=1)
  valid_std = np.std(valid, axis=1)
  size = np.arange(train_mean.shape[0])
  size += 1
  
  fig = plt.figure(figsize=(5*1.61803398875, 5), dpi=300)
  plt.fill_between(size, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
  plt.fill_between(size, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color='g')
  plt.plot(size, train_mean, '-', color='r', label='training')
  plt.plot(size, valid_mean, '-', color='g', label='validation')
  #plt.xticks([])
  plt.xlabel("# of trees")
  plt.ylabel("Accuracy")
  plt.legend(loc="best")
  fig.savefig('rf.png')

if __name__ == "__main__":
  print('eat accuracy.pickle and output rf.png')
  GIFT = 'accuracy.pickle'
  main()
