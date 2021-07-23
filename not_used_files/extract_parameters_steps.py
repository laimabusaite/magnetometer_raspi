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

from extract_sensitivity import import_coil_magnetic_field

if __name__ == '__main__':

    dir_name = 'RQnc/steps'

    folder_list = sorted(glob.glob(f'{dir_name}/*/'))
    print(folder_list)

    filename_coil = glob.glob(f'{dir_name}/*.xls')[0]
    # print(filename_coil)
    dataframe_coil = import_coil_magnetic_field(filename_coil)
    print(dataframe_coil)

    dataframe_steps = pd.DataFrame()

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
            g_idx = log_file.find('G/')
            mhz_idx = log_file.find('MHz')
            dev_idx = log_file.find('_dev')
            avg_idx = log_file.find('_avg')
            dot_idx = log_file.find('.log')
            # print('avg find', dev_idx, avg_idx, dot_idx)
            # print(log_file[dev_idx + 4:avg_idx], log_file[avg_idx + 4:dot_idx])
            log_file_split = re.split('/', log_file)
            # print(log_file_split)
            B_gauss = float(log_file_split[2][:-1])
            a = log_file_split[3]
            step = float(log_file[g_idx + 2:mhz_idx])
            dev = int(log_file[dev_idx + 4:avg_idx])  # float(log_file_split[4])
            avg = int(log_file[avg_idx + 4:dot_idx])  # float(log_file_split[5])

            print(f'B set = {B_set}, step = {step}, dev = {dev}, avg = {avg}')

            B_dataframe_temp = pd.read_csv(log_file, names=['Bx', 'By', 'Bz', 'B'], delimiter='\t')

            B_mean = B_dataframe_temp.mean()
            B_std = B_dataframe_temp.std() / np.sqrt(len(B_dataframe_temp) - 1)

            data = {
                'B index': B_set,
                'Bx coil (mT)': dataframe_coil.loc[idx_folder, 'Bx_coil'],
                'By coil (mT)': dataframe_coil.loc[idx_folder, 'By_coil'],
                'Bz coil (mT)': dataframe_coil.loc[idx_folder, 'Bz_coil'],
                'Bmod coil (mT)': dataframe_coil.loc[idx_folder, 'Bmod_coil'],
                'Bx coil stderr (mT)': dataframe_coil.loc[idx_folder, 'Bx_coil_std'],
                'By coil stderr (mT)': dataframe_coil.loc[idx_folder, 'By_coil_std'],
                'Bz coil stderr (mT)': dataframe_coil.loc[idx_folder, 'Bz_coil_std'],
                'Bmod coil stderr (mT)': dataframe_coil.loc[idx_folder, 'Bmod_coil_std'],
                'Bx measured (mT)': B_mean['Bx'] * 0.1,
                'By measured (mT)': B_mean['By'] * 0.1,
                'Bz measured (mT)': B_mean['Bz'] * 0.1,
                'Bmod measured (mT)': B_mean['B'] * 0.1,
                'Bx measured stderr (mT)': B_std['Bx'] * 0.1,
                'By measured stderr (mT)': B_std['By'] * 0.1,
                'Bz measured stderr (mT)': B_std['Bz'] * 0.1,
                'Bmod measured stderr (mT)': B_std['B'] * 0.1,
                'avg': avg,
                'dev (MHz)': dev,
                'step (MHz)': step,
                'point count': avg * dev / step

            }
            print(data)
            dataframe_steps = dataframe_steps.append(data, ignore_index=True)

    print(dataframe_steps)

    dataframe_steps.to_csv('tables/table_steps.csv', index=False)