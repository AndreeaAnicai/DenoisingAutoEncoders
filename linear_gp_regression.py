import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt


def lr_baseline_model():

    model = Sequential()
    model.add(Dense(400, input_dim=400, kernel_initializer='normal', activation='relu'))
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def linear_regression_prediction(X, y):

    seed = 7
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=lr_baseline_model, epochs=100, batch_size=5, verbose=0)

    # Evaluate model
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def gp_baseline_model(X, y):

    dy = 0.5 + 1.0 * np.random.random(y.shape)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    MSE = ((y_pred-y)**2).mean()

    print(MSE)


if __name__ == '__main__':

    dataframe = pd.read_csv("try_decoded_dataset_whole.csv")
    dataset = dataframe.values

    x = dataset[:, 0:400]
    y_cog13 = dataset[:, 400]

    # linear_regression_prediction(x, y_cog13)
    gp_baseline_model(x, y_cog13)



