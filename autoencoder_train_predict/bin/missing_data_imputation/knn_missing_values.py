import numpy as np
import pandas as pd
from sklearn import preprocessing
from fancyimpute import KNN
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore


def calculate_loss_mean():

    reconstructed = pd.read_csv('scale_loss/scaled_median_dataset_whole.csv')

    A = pd.read_csv('scaled_dataset_whole.csv')
    A_with_nan = pd.read_csv('nan_dataset_whole.csv')
    missing_ones_A = A_with_nan * 0
    missing_ones_A = missing_ones_A.replace(np.nan, 1)

    # Keep nan features in scaled dataset
    nans = missing_ones_A.replace(1, np.nan)
    nans = nans.replace(0.0, 1.0)
    scaled_with_missing = A.values * nans
    scaled_with_missing = scaled_with_missing.replace(np.nan, 0)

    # Mask 30% of values from A
    frac = 0.7
    sample = np.random.binomial(1, frac, size=A.shape[0] * A.shape[1])
    sample2 = sample.reshape(A.shape[0], A.shape[1])
    corrupted = A * sample2

    corrupted = corrupted.replace(0.0, np.nan)
    missing_ones_corrupted = corrupted * 0
    missing_ones_corrupted = missing_ones_corrupted.replace(np.nan, 1)

    missing_ones = np.add(missing_ones_A, missing_ones_corrupted)
    missing_ones = missing_ones.replace(2.0, 1.0)

    reconstructed = np.multiply(reconstructed.as_matrix(), missing_ones.as_matrix())
    original = np.multiply(scaled_with_missing.as_matrix(), missing_ones.as_matrix())

    rmse = np.sqrt(np.mean((reconstructed - original)**2))

    print(rmse)
    return rmse



def calculate_nrmse_loss(reconstructed, original, missing_ones):

    reconstructed_masked_value = np.multiply(reconstructed.as_matrix(), missing_ones.as_matrix())
    original_masked_value = np.multiply(original.as_matrix(), missing_ones.as_matrix())

    rmse = np.sqrt(np.mean((reconstructed_masked_value-original_masked_value)**2))

    return rmse


def knn_reconstruct():

    A = pd.read_csv('scaled_dataset_ad.csv')
    A_with_nan = pd.read_csv('nan_dataset_ad.csv')
    missing_ones_A = A_with_nan * 0
    missing_ones_A = missing_ones_A.replace(np.nan, 1)

    # K-fold cross validation for parameters
    neighbors = list(range(1, 50, 2))
    cv_scores = []

    # Keep nan features in scaled dataset
    nans = missing_ones_A.replace(1, np.nan)
    nans = nans.replace(0.0, 1.0)
    scaled_with_missing = A.values * nans
    A_nan = scaled_with_missing

    # Mask 30% of values from A
    frac = 0.8
    sample = np.random.binomial(1, frac, size=A.shape[0] * A.shape[1])
    sample2 = sample.reshape(A.shape[0], A.shape[1])
    corrupted = A * sample2

    # perform 10-fold cross validation
    for k in neighbors:

        # Use 5 nearest rows which have a feature to fill in each row's missing features
        A_filled_knn = KNN(k=k).fit_transform(A_nan)
        A_filled_knn = pd.DataFrame(A_filled_knn)
        # A_filled_knn.to_csv('knn_dataset_cn.csv')

        # Create mask for loss, both for the original matrix A and for the predicted matrix with 30%
        # masked
        corrupted = corrupted.replace(0.0, np.nan)
        missing_ones_corrupted = corrupted * 0
        missing_ones_corrupted = missing_ones_corrupted.replace(np.nan, 1)

        missing_ones = np.add(missing_ones_A, missing_ones_corrupted)
        missing_ones = missing_ones.replace(2.0, 1.0)

        loss = calculate_nrmse_loss(A_filled_knn, A, missing_ones)

        cv_scores.append(loss)

    print("Final MSE is: ", cv_scores)


if __name__ == '__main__':
    knn_reconstruct()
    # calculate_loss_mean()


