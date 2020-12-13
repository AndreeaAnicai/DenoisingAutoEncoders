import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import csv


def compute_stats():

    df = pd.read_csv('merged_FINAL_cleaned_data_10_08_17_use.csv', low_memory=False)
    df = df.replace(np.nan, '-99999999', regex=True)

    # df_col_names = pd.DataFrame(list(df.columns.values))
    # df_col_names.to_csv("columnNames.csv")

    df_missing = pd.DataFrame(df.isin([-99999999]).sum(axis=0))
    df_missing['TotalEntries'] = 2405
    df_missing.columns = ['MissingValues', 'TotalEntries']
    df_missing.to_csv("missingValues.csv")

    percentages = {}

    df_m = pd.read_csv('missingValues.csv', low_memory=False)
    for row in df_m.itertuples():
        if row[2] != 0:
            # print(row[1], row[2]/row[3]*100)
            perc = row[2]/row[3]*100
            percentages[row[1]] = {(row[2], row[3], perc)}

    with open('missingValuesPercentages.csv', 'w') as f:
        for key in percentages.keys():
            f.write("%s,%s\n" % (key, percentages[key]))


if __name__ == "__main__":
    compute_stats()
