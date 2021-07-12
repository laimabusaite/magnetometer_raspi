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
    print(folder_list)

    # import coil magnetic field
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
    # dataframe_coil *= 0.1
    print(dataframe_coil)

    # dataframe = pd.DataFrame()
    # dataframe_list = []
    col_list = []
    for dev in [16,20,32]:
        for avg in [4,8,16,64]:
            # col_list.append((f'dev {dev} MHz', f'avg {avg}'))
            col_list.append((dev, avg))

    print(col_list)

    dev_array = np.array([16, 20, 32])
    avg_array = np.array([4, 8, 16, 64])
    metric_array = np.array(['rate (Hz)'])

    index_col = pd.MultiIndex.from_product([dev_array, avg_array], names=['dev (MHz)', 'avg'])

    dataframe_rate = pd.DataFrame(index=[0, 1, 3, 5], columns=index_col)
    print(dataframe_rate)

    B_array = np.array(['Bx', 'By', 'Bz', 'Bmod'])
    index_col = pd.MultiIndex.from_product([dev_array, avg_array, B_array], names=['dev (MHz)', 'avg', 'B_std (mT)'])
    dataframe_accuracy = pd.DataFrame(index=[0, 1, 3, 5], columns=index_col)

    # dataframe_rate = pd.DataFrame(index=[0, 1, 3, 5], columns=['dev (MHz)', 'avg', 'rate (Hz)'])
    dataframe_list = []
    for idx_folder, folder in enumerate(folder_list):
        # folder = folder_list[idx_folder]

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

        # print(log_file_a_list)
        # print(log_file_summary_list)

        for idx, log_file in enumerate(log_file_a_list):
            # print(log_file, time.ctime(os.path.getctime(log_file)))
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
            # print(len(B_dataframe_temp))
            if len(B_dataframe_temp) == 10:
                # dataframe_list.append(B_dataframe_temp)
                B_mean = 0.1 * B_dataframe_temp.mean()
                B_std = 0.1 * B_dataframe_temp.std() / np.sqrt(len(B_dataframe_temp) - 1)

                log_summary = log_file_summary_list[idx]
                same = log_file[:-4] in log_summary
                # print(same)
                # extract from summary
                a_file = open(log_summary, 'r')
                summary = a_file.readlines()
                summary_split = re.split(' ', summary[0])
                # print(summary)
                # print(summary_split)
                total_time = float(summary_split[2])
                rate = 10./ total_time # float(summary_split[16]) #
                a_file.close()
                # print(f'Total time: {total_time} s, Rate: {rate} Hz')

                data_temp = {
                    'B_folder': [B_gauss],
                    'Bx_measured (mT)': [B_mean['Bx']],
                    'By_measured (mT)': [B_mean['By']],
                    'Bz_measured (mT)': [B_mean['Bz']],
                    'B_measured (mT)': [B_mean['B']],
                    'Bx_measured_std (mT)': [B_std['Bx']],
                    'By_measured_std (mT)': [B_std['By']],
                    'Bz_measured_std (mT)': [B_std['Bz']],
                    'B_measured_std (mT)': [B_std['B']],
                    'avg': [avg],
                    'dev (MHz)': [dev],
                    'total time (s)': [total_time],
                    'rate (Hz)': [rate]
                }
                print(B_gauss, dev, rate)
                dataframe_temp = pd.DataFrame(data=data_temp)
                # dataframe_rate.append(dataframe_temp)
                dataframe_list.append(dataframe_temp)
                # dataframe_rate.loc[B_gauss, ('rate (Hz)', f'dev={dev} MHz', f'{avg} avg')] = rate
                dataframe_rate.loc[B_gauss, (dev, avg)] = rate
                # dataframe_rate.loc[B_gauss, 'dev (MHz)'] = dev
                # dataframe_rate.loc[B_gauss, 'avg'] = avg
                # dataframe_rate.loc[B_gauss, 'rate (Hz)'] = rate
                dataframe_accuracy.loc[B_gauss, (dev, avg, 'Bx')] = B_std['Bx']
                dataframe_accuracy.loc[B_gauss, (dev, avg, 'By')] = B_std['By']
                dataframe_accuracy.loc[B_gauss, (dev, avg, 'Bz')] = B_std['Bz']
                dataframe_accuracy.loc[B_gauss, (dev, avg, 'Bmod')] = B_std['B']

    # dataframe_rate = dataframe_rate.reindex(sorted(dataframe_rate.columns), axis=1)
    # dataframe_rate = dataframe_rate.sort_index(axis=1)
    print(dataframe_rate)
    # print(dataframe_rate.T)

    print(dataframe_accuracy)



    # dataframe_full = pd.concat(dataframe_list, ignore_index=True)
    # # dataframe_full = dataframe_full.sort_index()
    # print(dataframe_full)
    # dataframe_B_rate = dataframe_full[['avg', 'dev (MHz)', 'rate (Hz)', 'B_folder']]
    # dataframe_B_rate = dataframe_B_rate.set_index(['avg', 'dev (MHz)'])
    # dataframe_B_rate = dataframe_B_rate.sort_index()
    # dataframe_B_rate_tr = dataframe_B_rate#.T
    # print(dataframe_B_rate_tr)
    # dataframe_B_rate_tr = dataframe_B_rate_tr.set_index('B_folder')



    dataframe_rate.to_csv('tables/table_rate.csv')
    dataframe_accuracy.to_csv('tables/table_accuracy.csv')

    # print(dataframe_rate.T)