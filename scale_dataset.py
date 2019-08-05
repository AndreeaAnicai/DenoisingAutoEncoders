from sklearn import preprocessing
import pandas as pd


def scale(dataset):

    names = dataset.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(dataset)
    dataset_new = pd.DataFrame(scaled_df, columns=names)

    return dataset_new


if __name__ == '__main__':

    dataset = pd.read_csv('dataset_mci.csv')
    scaled_dataset = scale(dataset)
    scaled_dataset.to_csv('scaled_dataset_mci.csv')
