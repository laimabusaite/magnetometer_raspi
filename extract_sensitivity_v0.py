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
from detect_peaks import detect_peaks_weighted

def extract_width(x_data, y_data, debug=False):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x0_init = detect_peaks_weighted(x_data, y_data)  # np.mean(x_data)
    y0_init = max(y_data)
    amplitude_init = min(y_data) - max(y_data)
    gamma_init = 5
    # try:
    popt, pconv = curve_fit(utilities.lorentz, x_data, y_data, p0=[x0_init, amplitude_init, gamma_init, y0_init],
                            bounds=((x_data[0], -1, 4.0, 0.8), (x_data[-1], -0.001, 20.0, 1.6)))

    x_fitted = np.linspace(x_data[0], x_data[-1], 1000)
    y_fitted = utilities.lorentz(x_fitted, popt[0], popt[1], popt[2], popt[3])
    if debug:
        plt.plot(x_fitted, y_fitted)
    # except Exception as e:
    #     print(e)

    return popt

if __name__ == '__main__':

    dir_name = 'RQnc/arb1'
    folder_list = sorted(glob.glob(f'{dir_name}/*/'))
    print(folder_list)



    idx_folder = 0
    folder = folder_list[idx_folder]

    # filename_fullscan_list = glob.glob(f'{folder}full*.dat')
    # filename_fullscan_list.sort(key=lambda x: f"{x.split('_')[-2]}_{x.split('_')[-1]}")
    # for idx_full, filename_fullscan in enumerate(filename_fullscan_list[:]):
    #     dataframe_odmr = pd.read_csv(filename_fullscan, names=['MW', 'ODMR'], sep='\t')
    #     peak_conditions = [(2500,2750), (2800,3100), (3170,3300), (3340,3500)]
    #     x_data = np.array(dataframe_odmr['MW'])
    #     y_data = np.array(dataframe_odmr['ODMR'])
    #     plt.plot(x_data, y_data, c='k')
    #     # B = 200
    #     # theta = 80
    #     # phi = 20
    #     # Blab = utilities.CartesianVector(B, theta, phi)
    #     # print(Blab)
    #     # init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2], 'glor': 10., 'D': 2870.00, 'Mz1': 1.67,
    #     #                'Mz2': 1.77, 'Mz3': 1.83, 'Mz4': 2.04}
    #     # parameters = fit_odmr.fit_full_odmr(x_data, y_data, init_params=init_params,
    #     #                                     debug=False, save=False)
    #
    #     for idx_peak, peak_range in enumerate(peak_conditions):
    #         print(peak_range)
    #         x_data_peak = dataframe_odmr[(dataframe_odmr['MW'] >= peak_range[0]) &
    #                                      (dataframe_odmr['MW'] <= peak_range[1])]['MW']
    #         y_data_peak = dataframe_odmr[(dataframe_odmr['MW'] >= peak_range[0]) &
    #                                      (dataframe_odmr['MW'] <= peak_range[1])]['ODMR']
    #         [x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted] = extract_width(x_data_peak, y_data_peak, debug=True)


    filename_list = glob.glob(f'{folder}*avg*.dat')
    filename_list.sort(key=lambda x: f"{x.split('_')[-2]}_{x.split('_')[-1]}")
    print(filename_list)
    dev_prev = 0
    avg_prev = 0
    dataframe_temp = pd.DataFrame()
    idx_a = 0
    for idx, filename in enumerate(filename_list[:]):
        print(filename)
        dev_idx = filename.find('_dev')
        avg_idx = filename.find('_avg')
        dev_idx2 = filename[avg_idx:].find('_dev') + avg_idx
        peak_idx = filename.find('peak')
        date_idx = filename.find('_2021-')
        dev = float(filename[dev_idx + 4:avg_idx])
        avg = int(filename[avg_idx + 4:dev_idx2])
        fr_centr = float(filename[peak_idx + 4:date_idx])
        print(dev, avg, fr_centr)

        dataframe_odmr = pd.read_csv(filename, names=['MW', 'ODMR'], sep='\t')
        # print(dataframe_odmr.head())

        if (dev_prev != dev) | (avg_prev != avg):
            # dataframe_temp = pd.DataFrame()
            plt.figure()
            idx_a += 1

        plt.title(f'{avg} avg, dev = {dev} MHz')
        x_data = np.array(dataframe_odmr['MW'])
        y_data = np.array(dataframe_odmr['ODMR'])
        plt.plot(x_data, y_data, marker='o')
        [x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted] = extract_width(x_data, y_data, debug=True)
        print(x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted)

        x_fitted = np.linspace(x_data[0] - 100, x_data[-1] + 100, 1000)
        y_fitted = utilities.lorentz(x_fitted, x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted)
        plt.plot(x_fitted, y_fitted)
        contrast = (y0_fitted - min(y_fitted)) / y0_fitted
        contrast1 = (max(y_fitted) - min(y_fitted))/max(y_fitted)
        contrast2 = amplitude_fitted / gamma_fitted / 2.
        print('contrast', contrast, contrast1, contrast2)

        peak_conditions = [(2500, 2750), (2800, 3100), (3170, 3300), (3340, 3500)]

        # idx1 = 0
        for idx1, peak_range in enumerate(peak_conditions):
            if peak_range[0] <= x0_fitted <= peak_range[1]:
                peak_nr = (idx1 + 1) * 2

        dict_temp = {
            'avg': avg,
            'dev': dev,
            'peak': x0_fitted,
            'width': gamma_fitted,
            'contrast': contrast,
            'peak nr': peak_nr
        }
        dataframe_temp = dataframe_temp.append(dict_temp, ignore_index=True)
        print(dataframe_temp)

        dev_prev = dev
        avg_prev = avg


    plt.show()