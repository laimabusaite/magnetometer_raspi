import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import utilities


if __name__ == '__main__':

    filename = 'tables/table_accuracy.csv'

    dataframe = pd.read_csv(filename, header=[0, 1, 2], index_col=[0, 1, 2, 3])

    print(dataframe)

    print(dataframe.index.get_level_values(1))
    print(dataframe.index.get_level_values(2))
    idx = pd.IndexSlice
    dataframe_mean = dataframe.loc[idx[:,:], idx[:, :, 'B (mT)']]
    dataframe_std = dataframe.loc[idx[:, :], idx[:, :, 'B_std (mT)']]
    # .loc[idx[:, :, ["C1", "C3"]], idx[:, "foo"]]
    col_num = len(dataframe_mean.columns)
    x_data = np.zeros(col_num)
    avg_data = np.zeros(col_num)
    dev_data = np.zeros(col_num)
    for idx_col, col in enumerate(dataframe_mean.columns):
        print(col)
        x_data[idx_col] = float(col[0]) * float(col[1])
        avg_data[idx_col] = int(col[1])
        dev_data[idx_col] = int(col[0])

    print(x_data)

    dataframe_mean.reset_index(level='B coil std (mT)', inplace=True, drop=True)
    dataframe_mean.reset_index(level='B coil (mT)', inplace=True)
    dataframe_std.reset_index(level='B coil std (mT)', inplace=True)
    dataframe_std.reset_index(level='B coil (mT)', inplace=True, drop=True)
    # # dataframe_mean.reset_index(level=0, inplace=True)
    print(dataframe_mean)
    # print(len(dataframe.columns))

    print(dataframe_mean.loc[(0,'Bx'), :])
    plt.figure(figsize=(5, 4))
    for idx_folder, B_folder in enumerate([0, 1, 3, 5]):
        dataframe_temp = pd.DataFrame(data={'x_data':x_data})
        idx = pd.IndexSlice
        dataframe_temp['dev (MHz)'] = dev_data
        dataframe_temp['avg'] = avg_data
        dataframe_temp['Bx_coil'] = dataframe_mean.loc[(B_folder,'Bx'), idx['B coil (mT)', :, :]].values[0]
        dataframe_temp['By_coil'] = dataframe_mean.loc[(B_folder, 'By'), idx['B coil (mT)', :, :]].values[0]
        dataframe_temp['Bz_coil'] = dataframe_mean.loc[(B_folder, 'Bz'), idx['B coil (mT)', :, :]].values[0]
        dataframe_temp['Bmod_coil'] = dataframe_mean.loc[(B_folder, 'Bmod'), idx['B coil (mT)', :, :]].values[0]
        dataframe_temp['Bx_measured'] = dataframe_mean.loc[(B_folder,'Bx'), idx[:, :, 'B (mT)']].values
        dataframe_temp['By_measured'] = dataframe_mean.loc[(B_folder, 'By'), idx[:, :, 'B (mT)']].values
        dataframe_temp['Bz_measured'] = dataframe_mean.loc[(B_folder, 'Bz'), idx[:, :, 'B (mT)']].values
        dataframe_temp['Bmod_measured'] = dataframe_mean.loc[(B_folder, 'Bmod'), idx[:, :, 'B (mT)']].values

        dataframe_temp['Bx_coil_std'] = dataframe_std.loc[(B_folder, 'Bx'), idx['B coil std (mT)', :, :]].values[0]
        dataframe_temp['By_coil_std'] = dataframe_std.loc[(B_folder, 'By'), idx['B coil std (mT)', :, :]].values[0]
        dataframe_temp['Bz_coil_std'] = dataframe_std.loc[(B_folder, 'Bz'), idx['B coil std (mT)', :, :]].values[0]
        dataframe_temp['Bmod_coil_std'] = dataframe_std.loc[(B_folder, 'Bmod'), idx['B coil std (mT)', :, :]].values[0]
        dataframe_temp['Bx_measured_std'] = dataframe_std.loc[(B_folder, 'Bx'), idx[:, :, 'B_std (mT)']].values
        dataframe_temp['By_measured_std'] = dataframe_std.loc[(B_folder, 'By'), idx[:, :, 'B_std (mT)']].values
        dataframe_temp['Bz_measured_std'] = dataframe_std.loc[(B_folder, 'Bz'), idx[:, :, 'B_std (mT)']].values
        dataframe_temp['Bmod_measured_std'] = dataframe_std.loc[(B_folder, 'Bmod'), idx[:, :, 'B_std (mT)']].values

        dataframe_temp.sort_values(by='x_data', inplace=True)

        # print(dataframe_temp)

        p = plt.errorbar(x=dataframe_temp['x_data'], y=dataframe_temp['Bmod_coil'], label='B coil')
        color = p[0].get_color()
        plt.axhline(dataframe_temp['Bmod_coil'].mean(), c=color)

        plt.errorbar(x=dataframe_temp['x_data'], y=dataframe_temp['Bmod_measured'],
                     yerr=dataframe_temp['Bmod_measured_std'], capsize=2, marker='o', ls='', ms=4, c=color, label='B measured')
        # plt.xlabel(r'Number of averages $\times$ frequency range (MHz)')
        plt.xlabel(r'Number of averages $\times$ number of measured points')
        plt.ylabel(r'Magnetic field |B| (mT)')
        plt.xlim(0, 2200)
        plt.tight_layout()

        print("ass",B_folder,dataframe_temp['Bmod_measured'].mean()-dataframe_temp['Bmod_coil'].mean())

    plt.legend()
    plt.savefig('tables/accuracy_vs_devavg.png')
    plt.savefig('tables/accuracy_vs_devavg.pdf')
    plt.show()