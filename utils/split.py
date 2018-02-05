#!/use/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    ret = []
    with open(FILE, 'r+') as f:
        for line in f:
            line = line.replace(' ', '\n')
            for i in line.split('\n'):
                ret.append(i)

    train_mean = []
    train_std = []
    valid_mean = []
    valid_std = []
    isTrain = 1
    for item in ret:
        if '+' not in item or ':' not in item:
            continue
        mean, std = item.split(':')[1].split('+')
        if isTrain:
            train_mean.append(mean)
            train_std.append(std)
        else:
            valid_mean.append(mean)
            valid_std.append(std)
        isTrain = not isTrain

    train_mean = np.array(train_mean, dtype=float)
    train_std = np.array(train_std, dtype=float)
    valid_mean = np.array(valid_mean, dtype=float)
    valid_std = np.array(valid_std,dtype=float)
    size = np.arange(train_mean.shape[0])
    size += 1
    #print(train_mean)
    #print(train_std)
    #print(valid_mean)
    #print(valid_std)
    #print(size)

    fig = plt.figure(figsize=(5*1.61803398875, 5), dpi=300)
    plt.fill_between(size, 1 - (train_mean - train_std), 1 - (train_mean + train_std), alpha=0.1, color='r')
    plt.fill_between(size, 1 - (valid_mean - valid_std), 1 - (valid_mean + valid_std), alpha=0.1, color='g')
    plt.plot(size, 1 - train_mean, '-', color='r', label='training')
    plt.plot(size, 1 - valid_mean, '-', color='g', label='validation')
    #plt.xticks([])
    plt.xlabel("# of trees")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    fig.savefig('xgboost.png')

if __name__ == '__main__':
    print('eat xgboost.txt and output xgboost.png')
    FILE = 'xgboost.txt'
    main()
