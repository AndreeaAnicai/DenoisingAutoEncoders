import keras
from matplotlib import pyplot as plt
import panda as pd
import numpy as np
import gzip
import os
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


def plot_figure(data, labels, label_dict):
    plt.figure(figsize=[5, 5])

    # Display the first image in training data
    plt.subplot(121)
    curr_img = np.reshape(data[0], (28, 28))
    curr_lbl = labels[0]
    plt.imshow(curr_img, cmap='gray')
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()


def autoencoder_f(input_img):
    # Encoder
    # Input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small+thick)

    # Decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
    return decoded


def plot_loss(autoencoder_train, epochs):
    # model_train = load_model('autoencoder_train_model.h5')
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def plot_predicted(test_data, test_labels, label_dict, pred):
    plt.figure(figsize=(20, 4))
    print("Test Images")
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(test_data[i, ..., 0], cmap='gray')
        curr_lbl = test_labels[i]
        plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()
    plt.figure(figsize=(20, 4))
    print("Reconstruction of Test Images")
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(pred[i, ..., 0], cmap='gray')
    plt.show()


def plot_noisy(x_train_noisy, x_test_noisy):
    plt.figure(figsize=[5, 5])

    # Display the first image in training data
    plt.subplot(121)
    curr_img = np.reshape(x_train_noisy[1], (28, 28))
    plt.imshow(curr_img, cmap='gray')

    # Display the first image in testing data
    plt.subplot(122)
    curr_img = np.reshape(x_test_noisy[1], (28, 28))
    plt.imshow(curr_img, cmap='gray')


def plot_noisy_predicted(test_data, x_test_noisy, pred, test_labels=0, label_dict=0):
    plt.figure(figsize=(20, 4))
    print("Test Images")
    for i in range(10, 20, 1):
        plt.subplot(2, 10, i + 1)
        plt.imshow(test_data[i, ..., 0], cmap='gray')
        curr_lbl = test_labels[i]
        plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
    plt.show()
    plt.figure(figsize=(20, 4))
    print("Test Images with Noise")
    for i in range(10, 20, 1):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
    plt.show()

    plt.figure(figsize=(20, 4))
    print("Reconstruction of Noisy Test Images")
    for i in range(10, 20, 1):
        plt.subplot(2, 10, i + 1)
        plt.imshow(pred[i, ..., 0], cmap='gray')
    plt.show()


def main():
    '''


    # Define data
    train_data = extract_data('train-images-idx3-ubyte.gz', 60000)
    test_data = extract_data('t10k-images-idx3-ubyte.gz', 10000)
    train_labels = extract_labels('train-labels-idx1-ubyte.gz', 60000)
    test_labels = extract_labels('t10k-labels-idx1-ubyte.gz', 10000)

    # Create dictionary of target classes
    label_dict = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
    }

    plot_figure(train_data, train_labels, label_dict)
    plot_figure(test_data, test_labels, label_dict)

    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)

    # Rescaling data with maximum pixel value
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)

    :return:
    '''
    data = pd.read_csv('final_output_files/onlyNumericValuesZeros.csv')
    train_data, test_data = train_test_split(data, data, test_size=0.2, random_state=13)

    # Split data in training and validation
    train_X, valid_X, train_ground, valid_ground = train_test_split(train_data,
                                                                    train_data,
                                                                    test_size=0.2,
                                                                    random_state=13)
    # Define hyperparamters
    batch_size = 128
    epochs = 5
    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape=(x, y, inChannel))

    '''
    # Train model
    autoencoder = Model(input_img, autoencoder_f(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

    autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs,
                                        verbose=1, validation_data=(valid_X, valid_ground))

    # Save model and plot loss
    autoencoder.save('autoencoder_train_model.h5')
    plot_loss(autoencoder_train, epochs)

    # Predict model
    pred = autoencoder.predict(test_data)
    plot_predicted(test_data, test_labels, label_dict, pred)
    '''

    # Add noise to images
    noise_factor = 0.5
    x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                              size=train_X.shape)
    x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                              size=valid_X.shape)
    x_test_noisy = test_data + noise_factor * np.random.normal(loc=0.0, scale=1.0,
                                                               size=test_data.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # Plot noisy images
    plot_noisy(x_train_noisy, x_test_noisy)

    # Train on denoising
    autoencoder = Model(input_img, autoencoder_f(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
    autoencoder_train = autoencoder.fit(x_train_noisy, train_X, batch_size=batch_size,
                                        epochs=epochs, verbose=1,
                                        validation_data=(x_valid_noisy, valid_X))

    # Plot denoising
    plot_loss(autoencoder_train, epochs)

    # Predict on denoising
    pred = autoencoder.predict(x_test_noisy)

    # Plot predicted
    plot_noisy_predicted(test_data, x_test_noisy, pred)


if __name__ == "__main__":
    main()