import numpy as np
import pandas as pd
from sklearn import preprocessing
from fancyimpute import KNN
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


def calculate_nrmse_loss(reconstructed, original, missing_ones):

    reconstructed_masked_value = np.multiply(reconstructed.as_matrix(), missing_ones.as_matrix())
    original_masked_value = np.multiply(original.as_matrix(), missing_ones.as_matrix())

    rmse = np.sqrt(np.mean((reconstructed_masked_value-original_masked_value)**2))

    return rmse


if __name__ == '__main__':

    # We use the train dataframe from Titanic dataset fancy impute removes column names.
    A = pd.read_csv('dataset_mci.csv')

    # Replace nan values from array
    A = A.replace(np.nan, -99999999)
    A = A.replace(-99999999, np.nan)

    # Scale dataset
    names_A = A.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(A)
    A = pd.DataFrame(scaled_df, columns=names_A)

    # Use 5 nearest rows which have a feature to fill in each row's missing features
    A_filled_knn = KNN(k=5).fit_transform(A)
    A_filled_knn = pd.DataFrame(A_filled_knn)

    # Mask 30% of values from predicted
    frac = 0.7
    sample = np.random.binomial(1, frac, size=A_filled_knn.shape[0] * A_filled_knn.shape[1])
    sample2 = sample.reshape(A_filled_knn.shape[0], A_filled_knn.shape[1])

    corrupted = A_filled_knn * sample2

    # Create mask for loss, both for the original matrix A and for the predicted matrix with 30% masked
    missing_ones_A = A * 0
    missing_ones_A = missing_ones_A.replace(np.nan, 1)

    corrupted = corrupted.replace(0.0, np.nan)
    missing_ones_corrupted = corrupted * 0
    missing_ones_corrupted = missing_ones_corrupted.replace(np.nan, 1)

    missing_ones = np.add(missing_ones_A, missing_ones_corrupted)

    A = A.replace(np.nan, 0)

    '''
    # Compute loss
    mse = (np.square(A.to_numpy() - A_filled_knn.to_numpy())).mean(axis=None)
    print(mse)
    mse = mse.mean()
    print("Final MSE is: ", mse)
    '''

    loss = calculate_nrmse_loss(A_filled_knn, A, missing_ones)

    print("Final MSE is: ", loss)

