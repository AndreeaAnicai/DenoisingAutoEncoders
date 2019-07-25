import numpy as np
import pandas as pd
from sklearn import preprocessing
from fancyimpute import KNN


if __name__ == '__main__':

    # We use the train dataframe from Titanic dataset fancy impute removes column names.
    A = pd.read_csv('deleted_missing_final.csv')

    # Replace nan values from array
    A = A.replace(np.nan, -99999999)
    A = A.replace(-99999999, np.nan)

    # Use 5 nearest rows which have a feature to fill in each row's missing features
    A_filled_knn = KNN(k=5).fit_transform(A)
    A_filled_knn = pd.DataFrame(A_filled_knn)

    A = A.replace(np.nan, 0)

    # Scale datasets
    names_A = A.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(A)
    A = pd.DataFrame(scaled_df, columns=names_A)

    names_A_filled = A_filled_knn.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(A_filled_knn)
    A_filled_knn = pd.DataFrame(scaled_df, columns=names_A_filled)

    # Compute loss
    mse = (np.square(A.to_numpy() - A_filled_knn.to_numpy())).mean(axis=None)
    print(mse)
    mse = mse.mean()
    print("Final MSE is: ", mse)



