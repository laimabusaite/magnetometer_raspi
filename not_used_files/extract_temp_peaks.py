import os
import re
import time

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import fit_odmr
import utilities
from detect_peaks import detect_peaks_weighted, detect_peaks

if __name__ == '__main__':

    dir_name = 'RQnc/temp/fan_23_degrees'

    filename_odmr_list = glob.glob(f'{dir_name}/fan*.dat')
    filename_odmr_list.sort(key=lambda x: f"{x.split('_')[-2]}_{x.split('_')[-1]}")

    filename_array = np.reshape(filename_odmr_list, (-1, 4)).transpose()
    print('filename_array')
    print(filename_array.shape)
    print(filename_array[:, 0])

    print(len(filename_odmr_list))
    print(filename_odmr_list[:10])

    peak_array = np.zeros_like(filename_array, dtype='float')
    for idx in range(filename_array.shape[1]):
        # if idx > 10:
        #     break
        filename_list = filename_array[:, idx]
        for idx2, filename in enumerate(filename_list):
            dataframe = pd.read_csv(filename, names=['MW', 'ODMR'], sep='\t')
            # print(dataframe.head())
            peak = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=False)
            peak_array[idx2, idx] = peak
        print(idx, peak_array[:, idx])

    dataframe = pd.DataFrame(data=peak_array[:, :].transpose())
    print(dataframe)
    dataframe.to_csv('tables/temp_peaks.csv')

    print(peak_array[:, 0:12])

    plt.plot(peak_array[:, :].transpose())
    plt.show()


