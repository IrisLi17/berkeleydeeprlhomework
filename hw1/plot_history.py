import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    print(history.history.keys())
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='mae')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='loss')
    # plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
    #          label='Val loss')
    plt.legend()
    # plt.ylim([0, 5])
    plt.show()
