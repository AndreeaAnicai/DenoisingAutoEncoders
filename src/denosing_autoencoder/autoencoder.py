import tensorflow as tf
from layers import *


def encoder_function(input):
    """

    Args:
        input: Original dataset
    Returns:
        layer_3: encoded representation after dataset has been through 3 layers of encoding
    """
    layer_1 = fc(input, 'layer_1', 398)
    layer_2 = fc(layer_1, 'layer_2', 201)
    layer_3 = fc(layer_2, 'layer_3', 100)
    return layer_3


def decoder_function(input):
    """

    Args:
        input: Encoded representation
    Returns:
        layer_3: reconstruction after dataset has been through 3 layers of decoding
    """
    fc_dec1 = fc(input, 'fc_dec1', 100)
    fc_dec2 = fc(fc_dec1, 'fc_dec2', 200)
    fc_dec3 = fc(fc_dec2, 'fc_dec3', 398)

    return fc_dec3


def autoencoder_denoising(input_shape):
    """

    Args:
        input_shape: the shape of the original dataset
    Returns:
        input_image: the original dataset with missing values
        reconstricted_image: the dataset after encryption and decryption by the autoencoder function
    """
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder_function(input_image)
        reconstructed_image = decoder_function(encoding)

    return input_image, reconstructed_image


