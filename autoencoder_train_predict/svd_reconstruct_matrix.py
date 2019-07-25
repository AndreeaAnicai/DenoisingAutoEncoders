import numpy as np
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
import pandas as pd
from scipy import linalg
from scipy.linalg import svd
from sklearn import preprocessing


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

    # Scale datasets

    names_A = A.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(A)
    A = pd.DataFrame(scaled_df, columns=names_A)

    B = pd.DataFrame(B)
    names_B = B.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(B)
    B = pd.DataFrame(scaled_df, columns=names_B)


    # Compute loss
    mse = (np.square(A.to_numpy() - B.to_numpy())).mean(axis=None)
    # mse = mse.mean()
    print(mse)


if __name__ == '__main__':
    svd_reconstruct()
