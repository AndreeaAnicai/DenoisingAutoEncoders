import numpy as np
import pandas as pd


def naive_mean(input_scaled, input_with_nan, reconstructed, fraction_masking):
    """
    Function that applies mean / median / zero imputation on a dataset with missing entries in
    order to achieve missing data imputation.

    Args:
        input_scaled: a Pandas Dataframe with the input dataset that requires missing data
        imputation, after z-score normalisation of features.

        input_with_nan: a Pandas Dataframe with the input dataset before z-score normalisation,
        where the missing values are replaced with Numpy NaNs

        fraction_masking: percentage of dataset that remains unmasked (for 20% maksing -> frac =
        0.8)

    Returns:
        loss: the RMSE computed only with the imputed missing values
    """

    # Create mask of existing missing data
    mask_input_dataset = input_with_nan * 0
    mask_input_dataset = mask_input_dataset.replace(np.nan, 1)
    nans = mask_input_dataset.replace(1, np.nan)
    nans = nans.replace(0.0, 1.0)
    dataset = input_scaled.values * nans
    dataset = dataset.replace(np.nan, 0)

    # Apply frac% masking
    sample = np.random.binomial(1, fraction_masking, size=input_scaled.shape[0] * input_scaled.shape[1])
    sample2 = sample.reshape(input_scaled.shape[0], input_scaled.shape[1])
    corrupted = input_scaled * sample2
    corrupted = corrupted.replace(0.0, np.nan)
    mask_corrupted = corrupted * 0
    mask_corrupted = mask_corrupted.replace(np.nan, 1)
    final_mask = np.add(mask_input_dataset, mask_corrupted)
    final_mask = final_mask.replace(2.0, 1.0)

    # Calculate loss
    reconstructed = np.multiply(reconstructed.as_matrix(), final_mask.as_matrix())
    original = np.multiply(dataset.as_matrix(), final_mask.as_matrix())

    loss = np.sqrt(np.mean((reconstructed - original)**2))

    return loss


if __name__ == '__main__':
    input_dataset = pd.read_csv('scaled_dataset_whole.csv')
    input_with_nans = pd.read_csv('nan_dataset_whole.csv')
    reconstructed = pd.read_csv('scale_loss/scaled_median_dataset_whole.csv')
    frac = 0.8

    loss_error = naive_mean(input_dataset, input_with_nans, reconstructed, frac)

    print("Loss error for SVD is ", loss_error)