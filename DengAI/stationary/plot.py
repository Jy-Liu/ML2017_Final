import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_pred(filename, y_true, y_pred):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(y_true.shape[0]), y_true, label='$y_{true}$')
    ax.plot(np.arange(y_pred.shape[0]), y_pred, label='$y_{pred}$')
    ax.legend()
    fig.savefig(filename)
