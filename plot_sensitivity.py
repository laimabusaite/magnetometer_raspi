
import numpy as np
import pandas as pd


if __name__ == '__main__':

    filename = 'tables/table_sensitivity.csv'

    dataframe = pd.read_csv(filename, header=0, index_col=0)
    dataframe.drop(dataframe.loc[dataframe['avg'] == 4].index, axis=0, inplace=True)
    print(dataframe.head())
    print(set(dataframe['avg']))

    dataframe_sensitivity = dataframe[['B index', 'sensitivity (nT/Hz1/2)']]

    print(dataframe_sensitivity)

    sensitivity_mean = dataframe_sensitivity['sensitivity (nT/Hz1/2)'].mean()
    sensitivity_std = dataframe_sensitivity['sensitivity (nT/Hz1/2)'].std() / (len(dataframe_sensitivity) - 1)

    dataframe_mean = dataframe_sensitivity.groupby(by='B index').mean()
    dataframe_mean['sensitivity stderr (nT/Hz1/2)'] = dataframe_sensitivity.groupby(by='B index').std() / (len(dataframe_sensitivity) - 1)
    dataframe_mean.loc['mean'] = dataframe_mean.mean()
    dataframe_mean.loc['mean', 'sensitivity stderr (nT/Hz1/2)'] = sensitivity_std
    print(dataframe_mean)
    print(dataframe_mean['sensitivity (nT/Hz1/2)'].mean(), sensitivity_mean, sensitivity_std)

    dataframe_mean.to_csv('tables/table_sensitivity_short.csv')
    # dataframe_mean.to_excel('tables/table_sensitivity_short.xlsx')

