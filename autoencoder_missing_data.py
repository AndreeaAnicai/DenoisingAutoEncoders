import os
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
import os
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import pandas as pd
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def remove_non_numeric_columns(df):
    df_numeric = df.drop(['EXAMDATE_DTIROI_04_30_14', 'update_stamp_BAIPETNMRC_09_12_16',
                          'STATUS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'OVERALLQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'TEMPQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'FRONTQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'PARQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'INSULAQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'OCCQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'BGQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'CWMQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'VENTQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'STATUS_BAIPETNMRC_09_12_16',
                          'EXAMDATE_BAIPETNMRC_09_12_16',
                          'RUNDATE_BAIPETNMRC_09_12_16',
                          'EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'VERSION_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'RUNDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'EXAMDATE_UPENNBIOMK9_04_19_17',
                          'KIT_UPENNBIOMK9_04_19_17',
                          'STDS_UPENNBIOMK9_04_19_17',
                          'RUNDATE_UPENNBIOMK9_04_19_17', 'update_stamp',
                          'update_stamp_BAIPETNMRC_09_12_16',
                          'update_stamp_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
                          'update_stamp_UPENNBIOMK9_04_19_17'], axis=1)

    data = df_numeric.values

    bit1 = data[:, 2:29]
    bit2 = data[:, 33:36]
    bit3 = data[:, 39:46]
    bit4 = data[:, 57:401]
    bit5 = data[:, 404:405]
    bit6 = data[:, 408:723]
    bit7 = data[:, 724:962]
    bit8 = data[:, 964:1206]
    bit9 = data[:, 1209:1214]
    bit10 = data[:, 1216:1444]
    bit11 = data[:, 1447:1503]

    data_numeric = np.concatenate((bit1, bit2), axis=1)
    data_numeric = np.concatenate((data_numeric, bit3), axis=1)
    data_numeric = np.concatenate((data_numeric, bit4), axis=1)
    data_numeric = np.concatenate((data_numeric, bit5), axis=1)
    data_numeric = np.concatenate((data_numeric, bit6), axis=1)
    data_numeric = np.concatenate((data_numeric, bit7), axis=1)
    data_numeric = np.concatenate((data_numeric, bit8), axis=1)
    data_numeric = np.concatenate((data_numeric, bit9), axis=1)
    data_numeric = np.concatenate((data_numeric, bit10), axis=1)
    data_numeric = np.concatenate((data_numeric, bit11), axis=1)

    return data_numeric


def impute_missing_values():
    df = pd.read_csv('merged_FINAL_cleaned_data_10_08_17_use.csv', low_memory=False)
    df_no_nan = df.replace(np.nan, -99999999, regex=True)

    # All data
    features = df_no_nan.values

    imputer = SimpleImputer(missing_values=-99999999, strategy='constant', fill_value=0)
    imputer.fit(features)
    features_zeros = imputer.fit_transform(features)

    imputer = SimpleImputer(missing_values=-99999999, strategy='most_frequent')
    imputer.fit(features)
    features_most_frequent = imputer.fit_transform(features)

    print(features_zeros)
    print(features_most_frequent)

    # Numeric only data
    features_num = remove_non_numeric_columns(df_no_nan)

    imputer = SimpleImputer(missing_values=-99999999, strategy='mean')
    imputer.fit(features_num)
    features_mean = imputer.fit_transform(features_num)

    print(features_zeros)

    df_zeros = pd.DataFrame(features_zeros)
    df_zeros.to_csv('onlyNumericValuesZeros.csv')


def extract_data(array, num_images=1):
    data = array
    data = data.reshape(num_images, 2405, 1442)
    return data


def plot_figure(data):
    plt.figure(figsize=[5, 5])

    # Display the first image in training data
    plt.subplot(121)
    # curr_img = np.reshape(data[0], (2405, 1442))
    curr_img = data
    plt.imshow(curr_img, cmap='gray')
    plt.show()


def autoencoder_f(input_img):
    # Encoder
    # Input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
    print("conv1", conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    print("pool1", pool1.shape)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
    print("conv2", conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    print("pool2", pool2.shape)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(
        pool2)  # 7 x 7 x 128 (small+thick)
    print("conv3", conv3.shape)

    # Decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
    print("conv4", conv4.shape)
    up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
    print("up1", up1.shape)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
    print("conv5", conv5.shape)
    up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
    print("up2", up2.shape)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
    print("decoded", decoded.shape)
    return decoded


def plot_loss(autoencoder_train, epochs):
    loss = autoencoder_train.history['loss']
    # val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot_predicted(test_data, pred):
    plt.figure(figsize=(20, 4))
    print("Test Images")
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(test_data[i, ..., 0], cmap='gray')
    plt.show()
    plt.figure(figsize=(20, 4))
    print("Reconstruction of Test Images")
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(pred[i, ..., 0], cmap='gray')
    plt.show()


def main():
    # Missing value = -99999999
    df_nines = pd.read_csv('csvs/onlyNumericValues.csv', low_memory=False)
    nines_array = df_nines.values
    print(nines_array.shape)
    nines_array = nines_array[:-4, :-2]

    df_zeros = pd.read_csv('csvs/onlyNumericValuesZeros.csv', low_memory=False)
    zeros_array = df_zeros.values

    print(nines_array.shape)
    # print(zeros_array.shape)

    train_data, test_data, train_labels, test_labels = train_test_split(nines_array, nines_array,
                                                                        test_size=0.2,
                                                                        random_state=13)
    # Plot data
    # plot_figure(train_data)
    # plot_figure(test_data)

    # Scale data
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)

    print(train_data.shape)
    print(test_data.shape)

    train_X, valid_X, train_ground, valid_ground = train_test_split(train_data,
                                                                    train_data,
                                                                    test_size=0.2,
                                                                    random_state=13)

    print(train_X.shape)
    print(valid_X.shape)
    print(train_ground.shape)
    print(valid_ground.shape)

    #  Reshape data
    train_data = train_data.reshape(-1, 1920, 1440, 1)
    test_data = test_data.reshape(-1, 481, 1440, 1)
    train_X = train_X.reshape(-1, 1536, 1440, 1)
    valid_X = valid_X.reshape(-1, 384, 1440, 1)
    train_ground = train_ground.reshape(-1, 1536, 1440, 1)
    valid_ground = valid_ground.reshape(-1, 384, 1440, 1)

    # Train model
    batch_size = 128
    epochs = 5
    in_channel = 1
    x, y = 1536, 1440
    input_img = Input(shape=(x, y, in_channel))

    print("input shape", train_X.shape)

    autoencoder = Model(input_img, autoencoder_f(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

    autoencoder_train = autoencoder.fit(train_X, train_ground,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1)

    # Plot results
    # Loss
    plot_loss(autoencoder_train, epochs)

    # Prediction
    pred = autoencoder.predict(test_data)
    plot_predicted(test_data, pred)


if __name__ == "__main__":
    main()
