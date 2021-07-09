import os
import re
import time

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

import utilities

if __name__ == '__main__':

    dir_name = 'RQnc/arb'
    folder_list = sorted(glob.glob(f'{dir_name}/*/'))
    # folder_list.sort(key=os.path.getmtime)
    print(folder_list)

    #import coil magnetic field
    filename_coil = glob.glob(f'{dir_name}/*.xls')[0]
    print(filename_coil)
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
    print(dataframe_coil)



    folder = folder_list[0]

    log_file_list = sorted(glob.glob(f'{folder}/*.log'))
    # log_file_list.sort(key=os.path.getmtime)
    print(log_file_list)
    log_file_a_list = []
    log_file_summary_list = []
    for log_file in log_file_list:
        if 'summary' in log_file:
            log_file_summary_list.append(log_file)
        else:
            log_file_a_list.append(log_file)

    print(log_file_a_list)
    print(log_file_summary_list)



    Bx_list = []
    By_list = []
    Bz_list = []
    B_list = []
    Bx_std_list = []
    By_std_list = []
    Bz_std_list = []
    B_std_list = []

    avg_list = []
    dev_list = []

    dataframe_list = []
    for idx, log_file in enumerate(log_file_a_list):
        # print(log_file, time.ctime(os.path.getctime(log_file)))
        log_file_split = re.split('G/a|/|_dev|_avg|.log',log_file)
        print(log_file_split)
        B_gauss = float(log_file_split[2])
        a = log_file_split[3]
        dev = float(log_file_split[4])
        avg = float(log_file_split[5])
        avg_list.append(avg)
        dev_list.append(dev)

        log_summary = log_file_summary_list[idx]
        same = log_file[:-4] in log_summary
        print(same)

        B_dataframe = pd.read_csv(log_file, names=['Bx', 'By', 'Bz', 'B'], delimiter='\t')
        dataframe_list.append(B_dataframe)

    dataframe_B = pd.concat(dataframe_list, ignore_index=True)
    print(dataframe_B)

    B_mean = dataframe_B.mean()
    # B_mean['Bmod'] = np.sqrt(B_mean['Bx']**2+B_mean['By']**2+B_mean['Bz']**2)
    print(B_mean)
    B_std = dataframe_B.std() / np.sqrt(len(dataframe_B) - 1)
    print(B_std)
    #
    # print(B_mean - B_std)
    #
    # print(B_mean + B_std)


