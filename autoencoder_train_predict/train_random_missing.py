import random
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from autoencoder import *
from sklearn import preprocessing

from autoencoder_train_predict.autoencoder import autoencoder4_d


def get_next_batch(dataset, batch_size, step, ind):
    start = step * batch_size
    end = ((step + 1) * batch_size)
    sel_ind = ind[start:end]
    newdataset = dataset.iloc[sel_ind, :]

    return newdataset


def calculate_nrmse_loss(reconstructed, input_shape):
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

'''
loss_val_list_train, loss_val_list_test
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
'''


def train(perc_dem, perc_cog, perc_csf, perc_mri, dataset_train, dataset_test, autoencoder_fun,
          restore=False, sav=True, checkpoint_file='default.ckpt'):
    input_image, reconstructed_image = autoencoder_fun(batch_shape)
    # The reconstructed image is the filled-in array

    original, loss, missing_mask = calculate_nrmse_loss(reconstructed_image,
                                                        [batch_size, feature_size])
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    start = time.time()
    loss_val_list_train = 0
    loss_val_list_test = 0

    with tf.Session() as session:

        session.run(init)
        dataset_size_train = dataset_train.shape[0]
        dataset_size_test = dataset_test.shape[0]
        print("Dataset size for training:", dataset_size_train)
        print("Dataset size for validation:", dataset_size_test)

        num_iters = (num_epochs * dataset_size_train) // batch_size
        print("Num iters:", num_iters)

        ind_train = 0
        for i in range(num_epochs):
            ind_train = np.append(ind_train, np.random.permutation(np.arange(dataset_size_train)))
        ind_test = 0

        iters = num_epochs * dataset_size_train // dataset_size_test + 1
        for i in range(iters):
            ind_test = np.append(ind_test, np.random.permutation(np.arange(dataset_size_test)))

        for step in range(num_iters):
            temp = get_next_batch(dataset_train, batch_size, step, ind_train)
            train_batch = np.asarray(temp).astype("float32")

            '''
            frac = 0.7
            sample = np.random.binomial(1, frac, size=temp.shape[0] * temp.shape[1])
            sample2 = sample.reshape(temp.shape[0], temp.shape[1])

            
            Mask according to percentage for each modality; train
            '''
            sample_dem = np.random.binomial(1, perc_dem, size=temp.shape[0] * 17)
            sample_dem = sample_dem.reshape(temp.shape[0], 17)

            sample_cog = np.random.binomial(1, perc_cog, size=temp.shape[0] * 9)
            sample_cog = sample_cog.reshape(temp.shape[0], 9)

            sample_csf = np.random.binomial(1, perc_csf, size=temp.shape[0] * 3)
            sample_csf = sample_csf.reshape(temp.shape[0], 3)

            sample_mri = np.random.binomial(1, perc_mri, size=temp.shape[0] * 373)
            sample_mri = sample_mri.reshape(temp.shape[0], 373)
            
            sample = np.concatenate((sample_dem, sample_cog, sample_csf), axis=1)
            sample = np.concatenate((sample, sample_mri), axis=1)

            missing_ones = np.ones_like(sample) - sample

            corrupted = temp * sample

            # Added this
            corrupted[corrupted == -99999999] = 0.0

            corrupted_batch = np.asarray(corrupted).astype("float32")

            train_loss_val, _ = session.run([loss, optimizer],
                                            feed_dict={input_image: corrupted_batch,
                                                       original: train_batch,
                                                       missing_mask: missing_ones})
            loss_val_list_train = np.append(loss_val_list_train, train_loss_val)

            temp = get_next_batch(dataset_test, batch_size, step, ind_test)
            test_batch = np.asarray(temp).astype("float32")

            '''
            frac = 0.7
            sample = np.random.binomial(1, frac, size=temp.shape[0] * temp.shape[1])
            sample2 = sample.reshape(temp.shape[0], temp.shape[1])

            
            Mask according to percentage for each modality; test
            '''

            sample_dem = np.random.binomial(1, perc_dem, size=temp.shape[0] * 17)
            sample_dem = sample_dem.reshape(temp.shape[0], 17)

            sample_cog = np.random.binomial(1, perc_cog, size=temp.shape[0] * 9)
            sample_cog = sample_cog.reshape(temp.shape[0], 9)

            sample_csf = np.random.binomial(1, perc_csf, size=temp.shape[0] * 3)
            sample_csf = sample_csf.reshape(temp.shape[0], 3)

            sample_mri = np.random.binomial(1, perc_mri, size=temp.shape[0] * 373)
            sample_mri = sample_mri.reshape(temp.shape[0], 373)

            sample = np.concatenate((sample_dem, sample_cog, sample_csf), axis=1)
            sample = np.concatenate((sample, sample_mri), axis=1)

            missing_ones = np.ones_like(sample) - sample
            corrupted = temp * sample
            corrupted_batch = np.asarray(corrupted).astype("float32")

            test_loss_val = session.run(loss,
                                        feed_dict={input_image: corrupted_batch,
                                                   original: test_batch,
                                                   missing_mask: missing_ones})
            loss_val_list_test = np.append(loss_val_list_test, test_loss_val)

            if step % 30 == 0:
                print(step, "/", num_iters, train_loss_val, test_loss_val)

        if sav:
            save_path = saver.save(session, checkpoint_file)
            print(("Model saved in file: %s" % save_path))

    end = time.time()
    el = end - start
    print(("Time elapsed %f" % el))

    return loss_val_list_train, loss_val_list_test


# [3]###################################
# tf.reset_default_graph()
# with tf.Graph().as_default():

if __name__ == '__main__':
    # input_name = sys.argv[1]  # 'rna_naremoved_logtransformed_normalized.csv'
    # output_path = sys.argv[2]  # 'imputationmodel.ckpt'
    # feature_size = sys.argv[3]  # Dimension of the feature, 17176
    # nonmissing_perc = sys.argv[4]  # Percent of non-missing elements in the data, 0.7
    # batch_size = sys.argv[5]  # 128
    # lr = sys.argv[6]  # 0.1
    # num_epochs = sys.argv[7]  # 450

    input_name = 'deleted_missing_final.csv'
    output_path = 'imputationmodel.ckpt'
    feature_size = 402

    # nonmissing_perc = 0.7
    perc_dem = 0.7
    perc_cog = 0.28
    perc_csf = 0.83
    perc_mri = 0.62

    batch_size = 20
    lr = 0.01
    num_epochs = 100

    df = pd.read_csv(input_name)

    # Replace nan values from array
    df = df.replace(np.nan, 0)
    df = df.replace(-99999999, 0)
    # df.drop(df.columns[[0]], axis=1, inplace=True)

    # Create set for training & validation
    arr = list(range(df.shape[0]))
    random.seed(1)
    random.shuffle(arr)
    use_ind = arr[0:int(df.shape[0] * 0.75)]
    holdout_ind = arr[int(df.shape[0] * 0.75):len(arr)]
    df_use = df.iloc[use_ind]
    df_holdout = df.iloc[holdout_ind]

    # Create set for testing
    arr = list(range(df_use.shape[0]))
    random.seed(1)
    random.shuffle(arr)
    train_ind = arr[0:int(df_use.shape[0] * 0.8)]
    test_ind = arr[int(df_use.shape[0] * 0.8):len(arr)]
    dataset_train = df_use.iloc[train_ind]
    dataset_test = df_use.iloc[test_ind]

    # Scale datasets
    names_train = dataset_train.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(dataset_train)
    dataset_train = pd.DataFrame(scaled_df, columns=names_train)

    names_test = dataset_test.columns
    scaled_df = scaler.fit_transform(dataset_test)
    dataset_test = pd.DataFrame(scaled_df, columns=names_test)

    # print("The number of nan values in train is: ", np.count_nonzero(np.isnan(dataset_train)))

    batch_shape = (batch_size, feature_size)
    np.set_printoptions(threshold=np.inf)
    tf.reset_default_graph()

    # Train model
    loss_val_list_train, loss_val_list_test = train(perc_dem, perc_cog, perc_csf, perc_mri,
                                                    dataset_train,
                                                    dataset_test,
                                                    autoencoder_fun=autoencoder4_d, sav=True,
                                                    restore=False, checkpoint_file=output_path)

    np.savetxt("trainloss.csv", loss_val_list_train, delimiter="\t")
    np.savetxt("validationloss.csv", loss_val_list_test, delimiter="\t")
