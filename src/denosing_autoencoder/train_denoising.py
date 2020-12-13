import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from autoencoder import *
from autoencoder_train_predict.autoencoder import autoencoder4_d


def train(perc_dem, perc_cog, perc_csf, perc_mri, dataset_train, dataset_test, autoencoder_function,
          sav=True, checkpoint_file='default.ckpt'):
    """
    Function that performs the training of the denoising autoencoder.
    Args:
        perc_dem: % of unmasked demographic data
        perc_cog: % of unmasked cognitive data
        perc_csf: % of unmasked CSF data
        perc_mri: % of unmasked MRI data
        dataset_train: Dataframe of training set
        dataset_test: Dataframe of testing set
        autoencoder_function: function that applies the autoencoder architecture
        sav: option to save model parameters
        checkpoint_file: output file for the model

    Return:
        loss_train_list: list of errors recorded for training dataset
        loss_test_list: list of errors recorded for testing dataset

    """

    input_image, reconstructed_image = autoencoder_function(batch_shape)
    original, loss, missing_mask = compute_rmse_loss(reconstructed_image, [batch_size,
                                                                           feature_size])
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start = time.time()
    loss_train_list = 0
    loss_test_list = 0

    with tf.Session() as session:

        # Initialise variables
        session.run(init)
        dataset_size_train = dataset_train.shape[0]
        dataset_size_test = dataset_test.shape[0]
        total_iterations = (num_epochs * dataset_size_train) // batch_size
        iterations_test = num_epochs * dataset_size_train // dataset_size_test + 1
        index_train = 0
        index_test = 0

        # Randomise subjects
        for i in range(num_epochs):
            index_train = np.append(index_train, np.random.permutation(np.arange(dataset_size_train)))
        for i in range(iterations_test):
            index_test = np.append(index_test, np.random.permutation(np.arange(dataset_size_test)))

        for step in range(total_iterations):

            # Get current batch
            temp = retrieve_batch(dataset_train, batch_size, step, index_train)
            train_batch = np.asarray(temp).astype("float32")

            # Apply masking per modality
            sample_dem = np.random.binomial(1, perc_dem, size=temp.shape[0] * 13)
            sample_dem = sample_dem.reshape(temp.shape[0], 13)

            sample_csf = np.random.binomial(1, perc_csf, size=temp.shape[0] * 3)
            sample_csf = sample_csf.reshape(temp.shape[0], 3)

            sample_mri = np.random.binomial(1, perc_mri, size=temp.shape[0] * 373)
            sample_mri = sample_mri.reshape(temp.shape[0], 373)

            sample_cog = np.random.binomial(1, perc_cog, size=temp.shape[0] * 9)
            sample_cog = sample_cog.reshape(temp.shape[0], 9)
            
            sample = np.concatenate((sample_dem, sample_csf, sample_mri), axis=1)
            sample = np.concatenate((sample, sample_cog), axis=1)

            # Create missing value mask
            mask = np.ones_like(sample) - sample
            corrupted = temp * sample
            corrupted_batch = np.asarray(corrupted).astype("float32")

            # Train denoising autoencoder
            train_loss_val, _ = session.run([loss, optimizer],
                                            feed_dict={input_image: corrupted_batch,
                                                       original: train_batch,
                                                       missing_mask: mask})
            loss_train_list = np.append(loss_train_list, train_loss_val)

            # Retrieve test batch
            temp = retrieve_batch(dataset_test, batch_size, step, index_test)
            test_batch = np.asarray(temp).astype("float32")

            # Apply masking per modality
            sample_dem = np.random.binomial(1, perc_dem, size=temp.shape[0] * 13)
            sample_dem = sample_dem.reshape(temp.shape[0], 13)

            sample_csf = np.random.binomial(1, perc_csf, size=temp.shape[0] * 3)
            sample_csf = sample_csf.reshape(temp.shape[0], 3)

            sample_mri = np.random.binomial(1, perc_mri, size=temp.shape[0] * 373)
            sample_mri = sample_mri.reshape(temp.shape[0], 373)

            sample_cog = np.random.binomial(1, perc_cog, size=temp.shape[0] * 9)
            sample_cog = sample_cog.reshape(temp.shape[0], 9)

            sample = np.concatenate((sample_dem, sample_csf, sample_mri), axis=1)
            sample = np.concatenate((sample, sample_cog), axis=1)

            # Create missing value mask
            mask = np.ones_like(sample) - sample
            corrupted = temp * sample
            corrupted_batch = np.asarray(corrupted).astype("float32")

            # Test denoising autoencoder
            test_loss_val = session.run(loss,
                                        feed_dict={input_image: corrupted_batch,
                                                   original: test_batch,
                                                   missing_mask: mask})
            loss_test_list = np.append(loss_test_list, test_loss_val)

            if step % 30 == 0:
                print(step, "/", total_iterations, train_loss_val, test_loss_val)

        if sav:
            save_path = saver.save(session, checkpoint_file)
            print(("Model saved in file: %s" % save_path))

    #  Calculate computation time
    end = time.time()
    elapsed = end - start
    print("Time elapsed %f" % elapsed)

    return loss_train_list, loss_test_list


def retrieve_batch(dataset, batch_size, step, index):
    """
    Function that retrieves a batch from the training set
    Args:
        dataset: the training set
        batch_size: size of batch to be extracted
        step: current iteration
        index: the training index
    Returns:
        new_batch: batch to be trained
    """
    start = step * batch_size
    end = ((step + 1) * batch_size)
    sel_ind = index[start:end]
    new_batch = dataset.iloc[sel_ind, :]

    return new_batch


def compute_rmse_loss(reconstructed, input_shape):
    """
    Calculate partial loss between original and imputed dataset. The loss is computed only for
    the entries that were originally missing.
    Args:
        reconstructed: dataset with imputed missing data
        input_shape: shape of dataset
    Returns:
        original: original dataset
        rmse: partial loss error
        missing_mask: 1's and 0's matrix where the 1's represented the values that were missing
        and filled in
    """
    original = tf.placeholder(tf.float32,
                              input_shape,
                              name='original')
    missing_mask = tf.placeholder(tf.float32,
                                  input_shape,
                                  name='original')

    reconstructed_masked_value = tf.multiply(reconstructed, missing_mask)
    original_masked_value = tf.multiply(original, missing_mask)

    rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(reconstructed_masked_value,
                                                                      original_masked_value)),
                                                axis=0)))

    return original, rmse, missing_mask


if __name__ == '__main__':

    input_name = 'scaled_dataset_whole.csv'
    output_path = 'imputationmodel.ckpt'

    # Read scaled dataset and create missing value mask
    input_dataset = pd.read_csv(input_name)
    input_dataset_nan = pd.read_csv('nan_dataset_whole.csv')
    mask_input_dataset = input_dataset_nan * 0
    mask_input_dataset = mask_input_dataset.replace(np.nan, 1)
    nans = mask_input_dataset.replace(1, np.nan)
    nans = nans.replace(0.0, 1.0)
    scaled_with_missing = input_dataset.values * nans

    # Set hyperparameters
    batch_size = 20
    lr = 0.01
    num_epochs = 450
    feature_size = 398

    # Set modality specific masking fraction
    perc_dem = 0.80
    perc_cog = 0.856888888
    perc_mri = 0.924317383615946
    perc_csf = 0.966

    # Replace nan values from array
    input_dataset = input_dataset.replace(np.nan, -99999999)
    input_dataset = input_dataset.replace(-99999999, np.nan)
    input_dataset = scaled_with_missing
    input_dataset = input_dataset.replace(np.nan, 0)

    # Create set for training & validation
    arr = list(range(input_dataset.shape[0]))
    random.seed(1)
    random.shuffle(arr)
    use_ind = arr[0:int(input_dataset.shape[0] * 0.75)]
    holdout_ind = arr[int(input_dataset.shape[0] * 0.75):len(arr)]
    df_use = input_dataset.iloc[use_ind]
    df_holdout = input_dataset.iloc[holdout_ind]

    # Create set for testing
    arr = list(range(df_use.shape[0]))
    random.seed(1)
    random.shuffle(arr)
    train_ind = arr[0:int(df_use.shape[0] * 0.8)]
    test_ind = arr[int(df_use.shape[0] * 0.8):len(arr)]
    dataset_train = df_use.iloc[train_ind]
    dataset_test = df_use.iloc[test_ind]

    batch_shape = (batch_size, feature_size)
    np.set_printoptions(threshold=np.inf)
    tf.reset_default_graph()

    # Train model
    loss_val_list_train, loss_val_list_test = train(perc_dem, perc_cog, perc_csf, perc_mri,
                                                    dataset_train,
                                                    dataset_test,
                                                    autoencoder_function=autoencoder4_d, sav=True,
                                                    restore=False, checkpoint_file=output_path)

    np.savetxt("trainloss_cn.csv", loss_val_list_train, delimiter="\t")
    np.savetxt("validationloss_cn.csv", loss_val_list_test, delimiter="\t")
