import numpy as np
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
import pandas as pd
from scipy import linalg
from scipy.linalg import svd
from sklearn import preprocessing
from scipy.stats import zscore


def calculate_nrmse_loss(reconstructed, original, missing_ones):

    reconstructed_masked_value = np.multiply(reconstructed.as_matrix(), missing_ones.as_matrix())
    original_masked_value = np.multiply(original.as_matrix(), missing_ones.as_matrix())

    rmse = np.sqrt(np.mean((reconstructed_masked_value-original_masked_value)**2))

    return rmse


def svd_reconstruct():

    A = pd.read_csv('deleted_missing_final.csv')

    # Replace nan values from array
    A = A.replace(np.nan, -99999999)
    A = A.replace(-99999999, np.nan)

    # Scale
    A = A.apply(zscore)

    # Create mask for loss
    missing_ones_A = A * 0
    missing_ones_A = missing_ones_A.replace(np.nan, 1)

    '''
    # Scale dataset
    names_A = A.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(A)
    A = pd.DataFrame(scaled_df, columns=names_A)
    '''

    # Mask 30% of values from A
    frac = 0.8
    sample = np.random.binomial(1, frac, size=A.shape[0] * A.shape[1])
    sample2 = sample.reshape(A.shape[0], A.shape[1])
    corrupted = A * sample2

    # Replace nans with 0
    A = A.replace(np.nan, 0)
    corrupted = corrupted.replace(np.nan, 0.0)

    # Singular-value decomposition
    U, s, Vh = linalg.svd(corrupted, full_matrices=False)
    # Populate Sigma with n x n diagonal matrix
    S = np.diag(s)
    # Reconstruct matrix
    B = np.dot(U, np.dot(S, Vh))
    B = pd.DataFrame(B)
    # B.to_csv('svd_dataset_cn.csv')

    # Create missing value mask
    corrupted = corrupted.replace(0.0, np.nan)
    missing_ones_corrupted = corrupted * 0
    missing_ones_corrupted = missing_ones_corrupted.replace(np.nan, 1)

    missing_ones = np.add(missing_ones_A, missing_ones_corrupted)
    missing_ones = missing_ones.replace(2.0, 1.0)

    loss = calculate_nrmse_loss(B, A, missing_ones)

    print("Final MSE is: ", loss)


if __name__ == '__main__':
    svd_reconstruct()
