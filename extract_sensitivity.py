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


def import_coil_magnetic_field(filename_coil):
    # import coil magnetic field
    ####################################################################################################################
    excel_coil = pd.ExcelFile(filename_coil)
    column_names = ['set Ix, mA', 'set Iy, mA', 'set Iz, mA']
    dataframe_coil = pd.read_excel(excel_coil, header=0)
    dataframe_coil['Bx_coil'], dataframe_coil['By_coil'], dataframe_coil['Bz_coil'] = utilities.amper2gauss_array(
        dataframe_coil[column_names[0]], dataframe_coil[column_names[1]], dataframe_coil[column_names[2]])
    dataframe_coil['Ix err'] = 0.5
    dataframe_coil['Iy err'] = 0.5
    dataframe_coil['Iz err'] = 0.1
    dataframe_coil['Bx_coil_std'], dataframe_coil['By_coil_std'], dataframe_coil[
        'Bz_coil_std'] = utilities.amper2gauss_array(
        dataframe_coil['Ix err'], dataframe_coil['Iy err'], dataframe_coil['Iz err'])
    dataframe_coil *= 0.1
    dataframe_coil['Bmod_coil'] = np.sqrt(
        dataframe_coil['Bx_coil'] ** 2 + dataframe_coil['By_coil'] ** 2 + dataframe_coil['Bz_coil'] ** 2)
    dataframe_coil['Bmod_coil_std'] = np.sqrt(
        dataframe_coil['Bx_coil_std'] ** 2 + dataframe_coil['By_coil_std'] ** 2 + dataframe_coil['Bz_coil_std'] ** 2)
    ####################################################################################################################
    return dataframe_coil


if __name__ == '__main__':

    dir_name = 'RQnc/arb1'
    folder_list = sorted(glob.glob(f'{dir_name}/*/'))
    print(folder_list)

    filename_coil = glob.glob(f'{dir_name}/*.xls')[0]
    # print(filename_coil)
    dataframe_coil = import_coil_magnetic_field(filename_coil)
    print(dataframe_coil)

    # dev_array = np.array([16, 20, 32])
    # avg_array = np.array([4, 8, 16, 64])
    # # metric_array = np.array(['rate (Hz)'])
    # index_col = pd.MultiIndex.from_product([dev_array, avg_array], names=['dev (MHz)', 'avg'])

    dataframe_sensitivity = pd.DataFrame()

    # idx_folder = 0
    # folder = folder_list[idx_folder]
    for idx_folder, folder in enumerate(folder_list):
        folder_split = re.split('/', folder)
        B_set = float(folder_split[2][:-1])
        print(B_set)

        log_file_list = sorted(glob.glob(f'{folder}/*.log'))
        # log_file_list.sort(key=os.path.getmtime)
        # print(log_file_list)
        log_file_a_list = []
        log_file_summary_list = []
        for log_file in log_file_list:
            if 'summary' in log_file:
                log_file_summary_list.append(log_file)
            else:
                log_file_a_list.append(log_file)

        print(log_file_a_list)
        for idx, log_file in enumerate(log_file_a_list):
            print(log_file)
            dev_idx = log_file.find('_dev')
            avg_idx = log_file.find('_avg')
            dot_idx = log_file.find('.log')
            # print('avg find', dev_idx, avg_idx, dot_idx)
            # print(log_file[dev_idx + 4:avg_idx], log_file[avg_idx + 4:dot_idx])
            log_file_split = re.split('/', log_file)
            # print(log_file_split)
            B_gauss = float(log_file_split[2][:-1])
            a = log_file_split[3]
            dev = int(log_file[dev_idx + 4:avg_idx])  # float(log_file_split[4])
            avg = int(log_file[avg_idx + 4:dot_idx])  # float(log_file_split[5])

            B_dataframe_temp = pd.read_csv(log_file, names=['Bx', 'By', 'Bz', 'B'], delimiter='\t')
            [log_file_head, log_file_tail] = os.path.split(log_file)
            print(log_file[:-4])

            filename_odmr_list = glob.glob(f'{log_file[:-4]}*.dat')
            filename_odmr_list.sort(key=lambda x: f"{x.split('_')[-2]}_{x.split('_')[-1]}")
            print(filename_odmr_list)

            plt.figure(f'{avg} avg, dev = {dev} MHz')
            plt.title(f'{avg} avg, dev = {dev} MHz')

            peak_conditions = [(2500, 2750), (2800, 3100), (3170, 3300), (3340, 3500)]
            contrast_array = np.empty((4, len(filename_odmr_list[:])//4))
            width_array = np.empty((4, len(filename_odmr_list[:])//4))
            for idx_odmr, filename in enumerate(filename_odmr_list[:]):
                print(idx_odmr, filename)

                peak_idx = filename.find('peak')
                date_idx = filename.find('_2021-')
                fr_centr = float(filename[peak_idx + 4:date_idx])
                print(dev, avg, fr_centr)

                dataframe_odmr = pd.read_csv(filename, names=['MW', 'ODMR'], sep='\t')
                x_data = np.array(dataframe_odmr['MW'])
                y_data = np.array(dataframe_odmr['ODMR'])
                [x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted] = extract_width(x_data, y_data, debug=True)
                print(x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted)

                x_fitted = np.linspace(x_data[0] - 100, x_data[-1] + 100, 1000)
                y_fitted = utilities.lorentz(x_fitted, x0_fitted, amplitude_fitted, gamma_fitted, y0_fitted)
                contrast = (y0_fitted - min(y_fitted)) / y0_fitted

                for idx1, peak_range in enumerate(peak_conditions):
                    if peak_range[0] <= x0_fitted <= peak_range[1]:
                        peak_nr = (idx1 + 1) * 2
                        contrast_array[idx1,idx_odmr//4] = contrast
                        width_array[idx1,idx_odmr//4] = gamma_fitted

                plt.plot(x_data, y_data, marker='o')
                plt.plot(x_fitted, y_fitted)

            width_mean = np.mean(width_array, axis=1)
            contrast_mean = np.mean(contrast_array, axis=1)
            width_std = np.std(width_array, axis=1) / np.sqrt(len(width_array) - 1)
            contrast_std = np.std(contrast_array, axis=1) / np.sqrt(len(contrast_array) - 1)
            print(width_array)
            print(width_array.shape)
            print(width_mean)

            B_mean = B_dataframe_temp.mean()
            B_std = B_dataframe_temp.std() / np.sqrt(len(B_dataframe_temp) - 1)

            data = {
                'B index': B_set,
                'Bx coil (mT)' : dataframe_coil.loc[idx_folder, 'Bx_coil'],
                'By coil (mT)': dataframe_coil.loc[idx_folder, 'By_coil'],
                'Bz coil (mT)': dataframe_coil.loc[idx_folder, 'Bz_coil'],
                'Bmod coil (mT)': dataframe_coil.loc[idx_folder, 'Bmod_coil'],
                'Bx coil std (mT)': dataframe_coil.loc[idx_folder, 'Bx_coil_std'],
                'By coil std (mT)': dataframe_coil.loc[idx_folder, 'By_coil_std'],
                'Bz coil std (mT)': dataframe_coil.loc[idx_folder, 'Bz_coil_std'],
                'Bmod coil std (mT)': dataframe_coil.loc[idx_folder, 'Bmod_coil_std'],
                'Bx measured (mT)' : B_mean['Bx'] * 0.1,
                'By measured (mT)': B_mean['By'] * 0.1,
                'Bz measured (mT)': B_mean['Bz'] * 0.1,
                'Bmod measured (mT)': B_mean['B'] * 0.1,
                'Bx measured std (mT)': B_std['Bx'] * 0.1,
                'By measured std (mT)': B_std['By'] * 0.1,
                'Bz measured std (mT)': B_std['Bz'] * 0.1,
                'Bmod measured std (mT)': B_std['B'] * 0.1,
                'avg': avg,
                'dev (MHz)': dev,
                'contrast 2': contrast_mean[0],
                'contrast 4': contrast_mean[1],
                'contrast 6': contrast_mean[2],
                'contrast 8': contrast_mean[3],
                'contrast std 2': contrast_std[0],
                'contrast std 4': contrast_std[1],
                'contrast std 6': contrast_std[2],
                'contrast std 8': contrast_std[3],
                'width 2': width_mean[0],
                'width 4': width_mean[1],
                'width 6': width_mean[2],
                'width 8': width_mean[3],
                'width std 2': width_std[0],
                'width std 4': width_std[1],
                'width std 6': width_std[2],
                'width std 8': width_std[3],

            }
            print(data)
            dataframe_sensitivity = dataframe_sensitivity.append(data, ignore_index=True)

    print(dataframe_sensitivity)

    dataframe_sensitivity.to_csv('tables/table_sensitivity.csv')

    plt.show()