import numpy as np
import pandas as pd
from scipy import linalg



def calculate_nrmse_loss(reconstructed, original, missing_ones):

    reconstructed_masked_value = np.multiply(reconstructed.as_matrix(), missing_ones.as_matrix())
    original_masked_value = np.multiply(original.as_matrix(), missing_ones.as_matrix())

    rmse = np.sqrt(np.mean((reconstructed_masked_value-original_masked_value)**2))

    return rmse


def svd_reconstruct():

    A = pd.read_csv('scaled_dataset_ad.csv')
    A_with_nan = pd.read_csv('nan_dataset_ad.csv')
    missing_ones_A = A_with_nan * 0
    missing_ones_A = missing_ones_A.replace(np.nan, 1)

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

    # Singular-value decomposition
    A_nan = A_nan.replace(np.nan, 0)
    U, s, Vh = linalg.svd(A_nan, full_matrices=False)
    # Populate Sigma with n x n diagonal matrix
    S = np.diag(s)
    # Reconstruct matrix
    B = np.dot(U, np.dot(S, Vh))
    B = pd.DataFrame(B)
    # B.to_csv('svd_dataset_cn.csv')

    # Create missing value mask
    missing_ones_corrupted = corrupted * 0
    missing_ones_corrupted = missing_ones_corrupted.replace(np.nan, 1)

    missing_ones = np.add(missing_ones_A, missing_ones_corrupted)
    missing_ones = missing_ones.replace(2.0, 1.0)

    loss = calculate_nrmse_loss(B, A, missing_ones)

    print("Final MSE is: ", loss)


if __name__ == '__main__':
    svd_reconstruct()

