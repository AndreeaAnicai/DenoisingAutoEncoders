import pandas as pd
import tensorflow as tf

from layers import *
import numpy as np


def encoder4_d(input):
    '''

    :param input: a dataset of size X x X
    :return: a dataset of size 1 x 100
    '''
    fc_1 = fc(input, 'fc_1', 402)
    fc_2 = fc(fc_1, 'fc_2', 201)
    fc_3 = fc(fc_2, 'fc_3', 100)
    return fc_3


def decoder4_d(input):
    '''

    :param input: a dataset of size 1 x 100
    :return: a dataset of size X x X
    '''
    fc_dec1 = fc(input, 'fc_dec1', 100)
    fc_dec2 = fc(fc_dec1, 'fc_dec2', 200)
    fc_dec3 = fc(fc_dec2, 'fc_dec3', 402)

    return fc_dec3


def autoencoder4_d(input_shape):
    '''

    :param input_shape: a dataset of size X x X
    :return:
        input_image - the original dataset passed to the encoder
        reconstricted_image - the dataset after encryption and decryption by the autoencoder function
    '''
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder4_d(input_image)
        reconstructed_image = decoder4_d(encoding)
        tf.io.write_file(
            "Reconstructed_image_whole.csv",
            reconstructed_image,
            name=None
        )
        np.savetxt("whole_encoded.csv", reconstructed_image, delimiter=",")
        # np.savetxt("Reconstructed_image_whole.csv", reconstructed_image, delimiter=",")
    return input_image, reconstructed_image


