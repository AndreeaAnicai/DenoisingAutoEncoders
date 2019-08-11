import os
import sys

import tensorflow as tf
import pandas as pd
import time
from autoencoder import *
import random
import numpy as np
from sklearn import preprocessing
import csv


from autoencoder_train_predict.autoencoder import autoencoder4_d


def reconstruct_loss(dataset_test_uncorrutped, dataset_test, autoencoder_fun,
                     checkpoint_file='default.ckpt', missing_ind=None):
    input_image, reconstructed_image = autoencoder_fun(batch_shape)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as session:

            print(("Loading variables from '%s'." % checkpoint_file))
            saver.restore(session, checkpoint_file)
            print('restored')    
            dataset_size = dataset_test.shape[0]
            print("Dataset size:", dataset_size)            
            
            dataset_test = np.asarray(dataset_test).astype("float32")
            dataset_test_uncorrutped = np.asarray(dataset_test_uncorrutped).astype("float32")
             
            reconstruct = session.run(reconstructed_image, feed_dict={input_image: dataset_test})
            recon_df = pd.DataFrame(reconstruct)

            recon_df.to_csv('decoded_dataset_cn.csv', mode='a', header=False)

            loss = rmse_loss(reconstruct, dataset_test_uncorrutped, missing_ind)

    return loss


def rmse_loss(reconstructed, original, missing_ind):
    # Add if statement 'if -9999999 then ignore?'

    rmse = np.sqrt(((reconstructed[0, missing_ind] - original[0, missing_ind]) ** 2).mean())
    return rmse


def mask_dfrow(row, perc):
    sample = np.random.binomial(1, perc, size=row.size)
    corrupted = row*sample

    return corrupted


if __name__ == '__main__':

        input_name = 'dataset_cn.csv'
        output_path = 'testloss_final_dataset.csv'
        model_path = 'models/imputationmodel.ckpt'
        feature_size = 402
        nonmissing_perc = 1.0

        holdout_cohort = pd.read_csv(input_name)
        holdout_cohort = holdout_cohort.replace(np.nan, 0)
        holdout_cohort = holdout_cohort.replace(-99999999, 0)

        # Scale datasets
        names_holdout = holdout_cohort.columns
        scaler = preprocessing.StandardScaler()
        scaled_df = scaler.fit_transform(holdout_cohort)
        holdout_cohort = pd.DataFrame(scaled_df, columns=names_holdout)

        np.random.seed(1)
        corrupted_holdout_cohort = holdout_cohort.apply(mask_dfrow, perc=nonmissing_perc, axis=1)
        
        loss_list = 0
        for i in range(0, corrupted_holdout_cohort.shape[0]):
            cur_test = corrupted_holdout_cohort.iloc[i:i+1, :]
            true_cur_test = holdout_cohort.iloc[i:i+1, :]
        
            missing_index = np.where(cur_test.iloc[0, :] == 0)[0]
            batch_shape = (1, feature_size)
            np.set_printoptions(threshold=np.inf)
            tf.reset_default_graph()
            loss_val = reconstruct_loss(true_cur_test, cur_test, autoencoder4_d, model_path,
                                        missing_index)
            
            loss_list = np.append(loss_list, loss_val)
            print(loss_val)
            if i % 5 == 0:
                np.savetxt(output_path, loss_list, delimiter="\t")
