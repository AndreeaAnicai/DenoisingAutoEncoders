import numpy as np
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
import pandas as pd
from scipy import linalg
from scipy.linalg import svd


def svd_reconstruct():

    A = pd.read_csv('deleted_missing_final.csv')

    # Replace nan values from array
    A = A.replace(np.nan, 0)
    A = A.replace(-99999999, 0)

    # # Singular-value decomposition
    # U, s, VT = svd(A, full_matrices=False)
    #
    # # Create m x n Sigma matrix
    # Sigma = zeros((A.shape[0], A.shape[1]))
    #
    # # Populate Sigma with n x n diagonal matrix
    # Sigma[:A.shape[1], :A.shape[1]] = diag(s)
    #
    # # Reconstruct matrix
    # B = U.dot(Sigma.dot(VT))

    # # # # # #
    U, s, Vh = linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    B = np.dot(U, np.dot(S, Vh))

    # Compute loss
    mse = (np.square(A - B)).mean(axis=None)
    mse = mse.mean()
    print(mse)


if __name__ == '__main__':
    svd_reconstruct()
