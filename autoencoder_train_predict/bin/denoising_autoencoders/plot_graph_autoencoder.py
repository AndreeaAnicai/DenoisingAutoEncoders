import pandas as pd
from matplotlib import pyplot as plt


def plot_loss():
    loss = pd.read_csv('../../data/final_output_files/trainloss.csv')
    val_loss = pd.read_csv('../../data/final_output_files/validationloss.csv')


    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training vs validation loss for dataset')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_pred():
    loss_pred = pd.read_csv('testloss_final_dataset.csv')
    epoch_count = range(1, len(loss_pred) + 1)

    plt.plot(epoch_count, loss_pred, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    plot_loss()
    # plot_pred()
