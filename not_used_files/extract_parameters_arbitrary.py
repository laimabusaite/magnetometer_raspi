import os
import re
import time

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

import utilities

if __name__ == '__main__':

    dir_name = 'RQnc/arb1'
    folder_list = sorted(glob.glob(f'{dir_name}/*/'))
    # folder_list.sort(key=os.path.getmtime)
    print(folder_list)

    #import coil magnetic field
    filename_coil = glob.glob(f'{dir_name}/*.xls')[0]
    print(filename_coil)
    excel_coil = pd.ExcelFile(filename_coil)
    column_names = ['set Ix, mA', 'set Iy, mA', 'set Iz, mA']
    dataframe_coil = pd.read_excel(excel_coil, header=0)
    dataframe_coil['Bx_coil'], dataframe_coil['By_coil'], dataframe_coil['Bz_coil'] =utilities.amper2gauss_array(
        dataframe_coil[column_names[0]], dataframe_coil[column_names[1]], dataframe_coil[column_names[2]])
    dataframe_coil['Ix err'] = 0.5
    dataframe_coil['Iy err'] = 0.5
    dataframe_coil['Iz err'] = 0.1
    dataframe_coil['Bx_coil_std'], dataframe_coil['By_coil_std'], dataframe_coil[
        'Bz_coil_std'] = utilities.amper2gauss_array(
        dataframe_coil['Ix err'], dataframe_coil['Iy err'], dataframe_coil['Iz err'])
    # dataframe_coil *= 0.1
    print(dataframe_coil)


    idx_folder = 3
    folder = folder_list[idx_folder]

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
    time_list = []
    rate_list = []

    dataframe_list = []
    for idx, log_file in enumerate(log_file_a_list):
        # print(log_file, time.ctime(os.path.getctime(log_file)))
        dev_idx = log_file.find('_dev')
        avg_idx = log_file.find('_avg')
        dot_idx = log_file.find('.log')
        print('avg find', dev_idx, avg_idx, dot_idx)
        print(log_file[dev_idx+4:avg_idx], log_file[avg_idx+4:dot_idx])
        log_file_split = re.split('G/a|/|_dev|_avg|.log|G', log_file)
        print(log_file_split)
        B_gauss = float(log_file_split[2])
        a = log_file_split[3]
        dev = float(log_file[dev_idx+4:avg_idx]) #float(log_file_split[4])
        avg = float(log_file[avg_idx+4:dot_idx]) #float(log_file_split[5])
        avg_list.append(avg)
        dev_list.append(dev)

        log_summary = log_file_summary_list[idx]
        same = log_file[:-4] in log_summary
        print(same)

        B_dataframe_temp = pd.read_csv(log_file, names=['Bx', 'By', 'Bz', 'B'], delimiter='\t')
        dataframe_list.append(B_dataframe_temp)
        B_mean = B_dataframe_temp.mean()
        B_std = B_dataframe_temp.std() / np.sqrt(len(B_dataframe_temp) - 1)

        B_list.append(0.1*B_mean['B'])
        Bx_list.append(0.1*B_mean['Bx'])
        By_list.append(0.1*B_mean['By'])
        Bz_list.append(0.1*B_mean['Bz'])
        B_std_list.append(0.1*B_std['B'])
        Bx_std_list.append(0.1*B_std['Bx'])
        By_std_list.append(0.1*B_std['By'])
        Bz_std_list.append(0.1*B_std['Bz'])

        #extract from summary
        a_file = open(log_summary, 'r')
        summary = a_file.readlines()
        summary_split = re.split(' ', summary[0])
        print(summary)
        print(summary_split)
        total_time = float(summary_split[2])
        rate = float(summary_split[16])
        a_file.close()
        print(f'Total time: {total_time} s, Rate: {rate} Hz')
        time_list.append(total_time)
        rate_list.append(rate)


    data = {
        # 'Bx_coil': dataframe_coil['Bx_coil'].values,
        # 'By_coil': dataframe_coil['By_coil'].values,
        # 'Bz_coil': dataframe_coil['Bz_coil'].values,
        # 'B_coil': B_coil_list,
        # 'Bx_coil_std': Bx_coil_list,
        # 'By_coil_std': By_coil_list,
        # 'Bz_coil_std': Bz_coil_list,
        # 'B_coil_std': B_coil_list,
        'Bx_measured (mT)': Bx_list,
        'By_measured (mT)': By_list,
        'Bz_measured (mT)': Bz_list,
        'B_measured (mT)': B_list,
        'Bx_measured_std (mT)': Bx_std_list,
        'By_measured_std (mT)': By_std_list,
        'Bz_measured_std (mT)': Bz_std_list,
        'B_measured_std (mT)': B_std_list,
        'avg': avg_list,
        'dev (MHz)': dev_list,
        'total time (s)': time_list,
        'rate (Hz)': rate_list
    }

    dataframe = pd.DataFrame(data)
    dataframe['Bx_coil (mT)'] = dataframe_coil.loc[0, 'Bx_coil'] * 0.1
    dataframe['By_coil (mT)'] = dataframe_coil.loc[0, 'By_coil'] * 0.1
    dataframe['Bz_coil (mT)'] = dataframe_coil.loc[0, 'Bz_coil'] * 0.1
    # dataframe['B_coil'] = dataframe_coil.loc[0, 'B_coil']
    dataframe['Bx_coil_std (mT)'] = dataframe_coil.loc[0, 'Bx_coil_std'] * 0.1
    dataframe['By_coil_std (mT)'] = dataframe_coil.loc[0, 'By_coil_std'] * 0.1
    dataframe['Bz_coil_std (mT)'] = dataframe_coil.loc[0, 'Bz_coil_std'] * 0.1
    # dataframe['B_coil_std'] = dataframe_coil.loc[0, 'B_coil_std']
    # print(dataframe)

    #
    # dataframe_B = pd.concat(dataframe_list, ignore_index=True)
    # print(dataframe_B)
    #
    # B_mean = dataframe_B.mean()
    # # B_mean['Bmod'] = np.sqrt(B_mean['Bx']**2+B_mean['By']**2+B_mean['Bz']**2)
    # print('B_mean')
    # print(B_mean)
    # B_std = dataframe_B.std() / np.sqrt(len(dataframe_B) - 1)
    # print('B_std')
    # print(B_std)
    # print('B_mean - B_std')
    # print(B_mean - B_std)
    # print('B_mean + B_std')
    # print(B_mean + B_std)
    #
    # B_mean['Bmod'] = np.sqrt(B_mean['Bx']**2+B_mean['By']**2+B_mean['Bz']**2)
    # print('B_mean')
    # print(B_mean)


    dataframe_dev20 = dataframe[dataframe['dev (MHz)']==20]
    # print(dataframe_dev20)

    dev_set = set(dataframe['dev (MHz)'])

    print(dev_set)

    for dev in dev_set:
        dataframe_dev = dataframe[dataframe['dev (MHz)'] == dev]
        print(dataframe_dev[['avg', 'rate (Hz)', 'dev (MHz)']])
        plt.scatter(dataframe_dev['avg'], dataframe_dev['rate (Hz)'], label=f'dev = {dev} MHz')

        plt.xlabel('Averages')
        plt.ylabel('Rate (Hz)')

    print(dataframe[['avg', 'rate (Hz)', 'dev (MHz)']])
    #
    # dataframe_rate = pd.DataFrame([])
    # dataframe_rate

    plt.legend()
    plt.show()


