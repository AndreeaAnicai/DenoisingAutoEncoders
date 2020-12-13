import numpy as np
import pandas as pd
from scipy import linalg


def svd_imputation(input_scaled, input_with_nan, fraction_masking):
    """
    Function that applies SVD (non-iterative) on a dataset with missing entries in order to achieve
    missing data imputation.

    Args:
        input_scaled: a Pandas Dataframe with the input dataset that requires missing data
        imputation, after z-score normalisation of features.

        input_with_nan: a Pandas Dataframe with the input dataset before z-score normalisation,
        where the missing values are replaced with Numpy NaNs

        fraction_masking: percentage of dataset that remains unmasked (for 20% maksing -> frac = 0.8)

    Returns:
        loss: the RMSE computed only with the imputed missing values
    """

    # Create mask of existing missing data
    mask_input_dataset = input_with_nan * 0
    mask_input_dataset = mask_input_dataset.replace(np.nan, 1)
    nans = mask_input_dataset.replace(1, np.nan)
    nans = nans.replace(0.0, 1.0)
    dataset = input_scaled.values * nans

    # Apply frac% masking
    sample = np.random.binomial(1, fraction_masking, size=input_scaled.shape[0] * input_scaled.shape[1])
    sample2 = sample.reshape(input_scaled.shape[0], input_scaled.shape[1])
    corrupted = input_scaled * sample2

    # Perform matrix singular-value decomposition
    dataset = dataset.replace(np.nan, 0)
    U, s, Vh = linalg.svd(dataset, full_matrices=False)
    S = np.diag(s)
    imputed_dataset = np.dot(U, np.dot(S, Vh))
    imputed_dataset = pd.DataFrame(imputed_dataset)

    # Create missing value mask
    mask_corrupted_dataset = corrupted * 0
    mask_corrupted_dataset = mask_corrupted_dataset.replace(np.nan, 1)
    final_mask = np.add(mask_input_dataset, mask_corrupted_dataset)
    final_mask = final_mask.replace(2.0, 1.0)

    loss = compute_rmse_loss(imputed_dataset, input_scaled, final_mask)

    '''
    # Save matrix with imputed values
    imputed_dataset.to_csv('svd_dataset_whole.csv')
    '''

    return loss


def compute_rmse_loss(reconstructed, original, mask):
    """
    Function that computes the partial RMSE loss between the original matrix and the 
    reconstructed dataset after imputation.
    
    Args:
        reconstructed: a Pandas Dataframe with the original dataset after SVD imputation
        
        original: a Pandas Dataframe with the original dataset
        
        mask: a Pandas Dataframe of 1's and 0s, with the 1's marking the location of where the 
        original missing values were. We only compute loss for these features.
        
    Returns:
        rmse: the loss error
    """

    reconstructed_masked_value = np.multiply(reconstructed.as_matrix(), mask.as_matrix())
    original_masked_value = np.multiply(original.as_matrix(), mask.as_matrix())
    rmse = np.sqrt(np.mean((reconstructed_masked_value-original_masked_value)**2))

    return rmse


if __name__ == '__main__':

    input_scaled = pd.read_csv('scaled_dataset_whole.csv')
    input_with_nans = pd.read_csv('nan_dataset_whole.csv')
    frac = 0.8

    loss_error = svd_imputation(input_scaled, input_with_nans, frac)

    print("Loss error for SVD is ", loss_error)

