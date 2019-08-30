import numpy as np
import tensorflow as tf
import pandas as pd


if __name__ == '__main__':

    original = pd.read_csv('original_nan_whole.csv')
    original = original.fillna(-999999999999)
    original = original.values
    # reconstructed = reconstructed.values
    reconstructed = pd.read_csv('reconstructed_score_whole.csv')
    reconstructed = reconstructed.values

    dif = original - reconstructed
    dif[dif < -99999999] = 0

    dif = dif.mean(axis=1)

    # original = original.values
    # rmse = np.sqrt(np.mean((reconstructed-original)**2))
    # mse = np.sqrt((np.square(dif)).mean(axis=1))
    # print(mse)