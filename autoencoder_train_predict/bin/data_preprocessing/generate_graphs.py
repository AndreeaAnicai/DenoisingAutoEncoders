import matplotlib.pyplot as plt;
import numpy as np
import pandas as pd

plt.rcdefaults()


def main():
    df = pd.read_csv('../../data/final_output_files/onlyNumericValuesZeros.csv', low_memory=False)
    df = df.replace(np.nan, '-99999999', regex=True)
    df2 = df[['RID', 'VISCODE', 'SITE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT',
              'PTMARRY', 'APOE4', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE']]
    df2.to_csv('relevantColumnsExtended.csv')

    df = pd.read_csv('../../data/final_output_files/relevantColumnsExtended.csv')
    number_of_rows = len(df.index)

    unique_rid_list = get_unique_rid_list(df)

    full_stats = {}
    for i in unique_rid_list:
        rid_group = row_filter_function(df, i)
        rid_dict = {}
        rid_dict['ADAS11'] = get_missing_stats(rid_group, 'ADAS11')
        rid_dict['ADAS13'] = get_missing_stats(rid_group, 'ADAS13')
        rid_dict['MMSE'] = get_missing_stats(rid_group, 'MMSE')
        rid_dict['CDRSB'] = get_missing_stats(rid_group, 'CDRSB')
        full_stats[i] = rid_dict

    # data_completeness_plot(full_stats)

    # MMSE_total_missing, MMSE_total_non_missing = calculate_missing_totals(full_stats, 'MMSE')
    # ADAS11_total_missing, ADAS11_total_non_missing = calculate_missing_totals(full_stats, 'ADAS11')
    # ADAS13_total_missing, ADAS13_total_non_missing = calculate_missing_totals(full_stats, 'ADAS13')
    # CDRSB_total_missing, CDRSB_total_non_missing = calculate_missing_totals(full_stats, 'CDRSB')

    # plot_pie_chart(MMSE_total_missing, MMSE_total_non_missing)
    # plot_pie_chart(ADAS11_total_missing, ADAS11_total_non_missing)
    # plot_pie_chart(ADAS13_total_missing, ADAS13_total_non_missing)
    # plot_pie_chart(CDRSB_total_missing, CDRSB_total_non_missing)

    breakpoint = 0

    # plot_graph_for_column(column_stats[13], number_of_rows)


def get_unique_rid_list(df):
    rid_list = df['RID']
    rid_list = rid_list.to_list()
    return list(set(rid_list))


def get_missing_stats(rid_group, column):
    rid_group_length = len(rid_group[column])
    rid_group_stats = rid_group[column].value_counts().to_dict()
    missing_values, non_missing_values = get_value_missing_value_stats(rid_group_stats,
                                                                       rid_group_length)
    return {'missing_values': missing_values, 'non_missing_values': non_missing_values}


# get all information for person with rid specified
def row_filter_function(df, rid):
    grouped_by_individual_filter = df['RID'] == rid
    grouped_by_individual_df = df[grouped_by_individual_filter]
    return grouped_by_individual_df


def get_value_missing_value_stats(rid_stats, rid_group_length):
    missing_values = 0

    if -99999999 in rid_stats.keys():
        missing_values = rid_stats[-99999999]

    non_missing_values = rid_group_length - missing_values
    return missing_values, non_missing_values


def get_full_column_stats(df):
    num_columns = len(df.columns)
    column_stats = []
    for i in range(len(df.columns)):
        value_counts = df.iloc[:, i].value_counts()
        column_stats.append(value_counts.to_dict())
    return column_stats


def calculate_missing_totals(rid_stats_dict, test):
    total_missing = 0
    total_non_missing = 0

    for rid, testType in rid_stats_dict.items():
        total_missing += testType[test]['missing_values']
        total_non_missing += testType[test]['non_missing_values']
    return total_missing, total_non_missing


def plot_pie_chart(test_missing, test_non_missing):
    # plot pie chart
    labels = 'Missing', 'Non-missing'
    sizes = [test_missing, test_non_missing]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('CDRSB Cognitive Test Data')

    plt.show()


def plot_graph_for_column(column_stats_dict, number_of_rows):
    missing_values = 0

    if -99999999 in column_stats_dict.keys():
        missing_values = column_stats_dict[-99999999]

    non_missing_values = number_of_rows - missing_values

    objects = ('Missing Values', 'Non-Missing Values')

    y_pos = np.arange(len(objects))
    performance = [missing_values, non_missing_values]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Session Data')
    plt.title('Patient Missing Data')

    plt.show()
    l = 0


def data_completeness_plot(rid_stats_dict):
    test1 = 'ADAS11'
    test2 = 'ADAS13'
    test3 = 'MMSE'
    test4 = 'CDRSB'

    # data to plot
    n_groups = len(rid_stats_dict)

    missing_ADAS11 = []
    missing_ADAS13 = []
    missing_MMSE = []
    missing_CDRSB = []
    non_missing_ADAS11 = []
    non_missing_ADAS13 = []
    non_missing_MMSE = []
    non_missing_CDRSB = []

    for rid, test in rid_stats_dict.items():
        missing_ADAS11.append(test[test1]['missing_values'])
        non_missing_ADAS11.append(test[test1]['non_missing_values'])

        missing_ADAS13.append(test[test2]['missing_values'])
        non_missing_ADAS13.append(test[test2]['non_missing_values'])

        missing_MMSE.append(test[test3]['missing_values'])
        non_missing_MMSE.append(test[test3]['non_missing_values'])

        missing_CDRSB.append(test[test3]['missing_values'])
        non_missing_CDRSB.append(test[test3]['non_missing_values'])

    missing_ADAS11 = tuple(missing_ADAS11)
    missing_ADAS13 = tuple(missing_ADAS13)
    missing_MMSE = tuple(missing_MMSE)
    missing_CDRSB = tuple(missing_CDRSB)
    non_missing_ADAS11 = tuple(non_missing_ADAS11)
    non_missing_ADAS13 = tuple(non_missing_ADAS13)
    non_missing_MMSE = tuple(non_missing_MMSE)
    non_missing_CDRSB = tuple(non_missing_CDRSB)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, missing_CDRSB, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Missing')

    rects2 = plt.bar(index + bar_width, non_missing_CDRSB, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Non-Missing')

    plt.xlabel('Participant')
    plt.ylabel('Session Data')
    plt.title('CDRSB Cognitive Test Data')
    # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
