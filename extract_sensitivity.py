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

    idx_folder = 0
    folder = folder_list[idx_folder]

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

            plt.plot(x_data, y_data, marker='o')
            plt.plot(x_fitted, y_fitted)

    plt.show()