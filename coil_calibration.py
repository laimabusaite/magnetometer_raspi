import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':

    dir0 = 'coil_calibration'

    slope_list = []
    intercept_list = []

    dir_axis_list = ['x', 'y', 'z']
    for dir_axis in dir_axis_list:
        fig = plt.figure(f'{dir_axis}-axis')
        ax1 = plt.subplot(1, 1, 1)
        fig2 = plt.figure(f'{dir_axis}-axis hist')
        ax2 = plt.subplot(1, 1, 1)
        filepath0_list = glob.glob(f'{dir0}/{dir_axis}/*.txt')
        print(filepath0_list)

        current_x_list = []
        current_y_list = []
        current_z_list = []
        Bx_list = []
        By_list = []
        Bz_list = []
        Bmod_list = []
        Bx_std_list = []
        By_std_list = []
        Bz_std_list = []
        Bmod_std_list = []
        Bmod_np_std_list= []
        for filepath in filepath0_list[:]:
            head_tail = os.path.split(filepath)
            filename = head_tail[1]
            # print(filename)
            filename_split = re.split('_|.txt', filename)
            # print(filename_split)

            current_x = float(filename_split[0])
            current_y = float(filename_split[1])
            current_z = float(filename_split[2])
            print(current_x, current_y, current_z)
            current_x_list.append(current_x)
            current_y_list.append(current_y)
            current_z_list.append(current_z)

            dataframe = pd.read_csv(filepath, sep='\s+', skiprows=2, header=0)
            print(dataframe.head())
            print(dataframe.columns)
            dataframe.plot(x='time[s]',y=['Bx[mT]', 'By[mT]', 'Bz[mT]', 'Bmod[mT]'], ax=ax1)
            dataframe['Bmod[mT]'].plot.hist(ax=ax2, bins=20)


            dataframe_mean = dataframe.mean()
            # print(dataframe_mean)
            Bx_mean = dataframe_mean[1]
            Bx_list.append(Bx_mean)
            # print(Bx_mean)
            By_mean = dataframe_mean[2]
            By_list.append(By_mean)
            # print(By_mean)
            Bz_mean = dataframe_mean[3]
            Bz_list.append(Bz_mean)
            # print(Bz_mean)
            Bmod_mean = dataframe_mean[4]
            Bmod_list.append(Bmod_mean)
            # print(Bmod_mean)


            dataframe_std = dataframe.std()
            # print(dataframe_mean)
            Bx_std = dataframe_std[1] / (len(dataframe_std) - 1)
            Bx_std_list.append(Bx_std)
            # print(Bx_mean)
            By_std = dataframe_std[2] / (len(dataframe_std) - 1)
            By_std_list.append(By_std)
            # print(By_mean)
            Bz_std = dataframe_std[3] / (len(dataframe_std) - 1)
            Bz_std_list.append(Bz_std)
            # print(Bz_mean)
            Bmod_std = dataframe_std[4] / (len(dataframe_std) - 1)
            Bmod_std_list.append(Bmod_std)

            Bmod_np_std = np.std(np.array(dataframe['Bmod[mT]'])) / (len(dataframe_std) - 1)
            Bmod_np_std_list.append(Bmod_std)
            # print(Bmod_mean)

            # print(Bx_mean, By_mean, Bz_mean, Bmod_mean)
            # print(Bx_std, By_std, Bz_std, Bmod_std)
            # print(Bx_std/Bx_mean, By_std/By_mean, Bz_std/Bz_mean, Bmod_std/Bmod_mean)

        data_axis = {
            'current_x_coil': current_x_list,
            'current_y_coil': current_y_list,
            'current_z_coil': current_z_list,
            'Bx_probe': Bx_list,
            'By_probe': By_list,
            'Bz_probe': Bz_list,
            'Bmod_probe': Bmod_list,
            'Bx_std_probe': Bx_std_list,
            'By_std_probe': By_std_list,
            'Bz_std_probe': Bz_std_list,
            'Bmod_std_probe': Bmod_std_list,
            'Bmod_np_std_probe': Bmod_np_std_list
        }
        dataframe_axis = pd.DataFrame(data=data_axis)
        dataframe_axis.sort_values(by=f'current_{dir_axis}_coil', inplace=True)
        dataframe_axis.reset_index(inplace=True, drop=True)
        # print(dataframe_axis)
        dataframe_axis0 = dataframe_axis[dataframe_axis[f'current_{dir_axis}_coil'] == 0.0]
        # print(dataframe_axis0)

        data0 = pd.DataFrame(dataframe_axis0.mean())
        print(data0)

        dataframe_axis = dataframe_axis.append(data0.T)
        dataframe_axis = dataframe_axis - dataframe_axis0.mean()
        # dataframe_axis[dataframe_axis[f'current_{dir_axis}_coil']==0.0][0] = dataframe_axis0.mean()
        index0 = dataframe_axis0.index
        print(index0)
        dataframe_axis.drop(index=index0, inplace=True)

        dataframe_axis.sort_values(by=f'current_{dir_axis}_coil', inplace=True)
        dataframe_axis.reset_index(inplace=True, drop=True)

        dataframe_axis['Bmod_probe'] = np.sqrt(dataframe_axis['Bx_probe'] ** 2 +
                                               dataframe_axis['By_probe'] ** 2 +
                                               dataframe_axis['Bz_probe'] ** 2)

        index0 = dataframe_axis[dataframe_axis[f'current_{dir_axis}_coil'] == 0.0].index[0]

        dataframe_axis.loc[:index0,'Bmod_probe'] *= -1.



        # fit linear
        slope, intercept = np.polyfit(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bmod_probe'], 1)
        print(f'slope = {slope}, intercept = {intercept}')
        slope_list.append(slope)
        intercept_list.append(intercept)
        dataframe_axis['Bmod_linear'] = slope * dataframe_axis[f'current_{dir_axis}_coil'] + intercept
        dataframe_axis['slope'] = slope
        dataframe_axis['intercept'] = intercept

        plt.figure()
        plt.title(f'{dir_axis}-axis')
        # plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bx_probe'], label='Bx')
        # plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['By_probe'], label='By')
        # plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bz_probe'], label='Bz')
        # plt.scatter(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bmod_probe'], label='Bmod')

        plt.errorbar(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bx_probe'],
                     yerr=dataframe_axis['Bx_std_probe'], label='Bx', marker='o', ms=5, capsize=2)
        plt.errorbar(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['By_probe'],
                     yerr=dataframe_axis['By_std_probe'], label='By', marker='o', ms=5, capsize=2)
        plt.errorbar(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bz_probe'],
                     yerr=dataframe_axis['Bz_std_probe'], label='Bz', marker='o', ms=5, capsize=2)
        plt.errorbar(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bmod_probe'],
                     yerr=dataframe_axis['Bmod_std_probe'], label='Bz', marker='o', ms=5, capsize=2)


        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bmod_linear'], label=f'Blinear {slope:.5f} x + {intercept:.5f}')

        plt.xlabel('Current, mA')
        plt.ylabel('B, mT')
        plt.legend()

        print(dataframe_axis)
        # dataframe_axis.to_csv(f'coil_calibration/extracted_data/{dir_axis}_coil.dat', index=False)

        plt.figure(f'{dir_axis}-axis std')
        plt.title(f'{dir_axis}-axis std')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], np.abs(dataframe_axis['Bx_std_probe']), label='Bx std')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], np.abs(dataframe_axis['By_std_probe']), label='By std')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], np.abs(dataframe_axis['Bz_std_probe']), label='Bz std')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], np.abs(dataframe_axis['Bmod_std_probe']), label='Bmod std')
        plt.scatter(dataframe_axis[f'current_{dir_axis}_coil'], np.abs(dataframe_axis['Bmod_np_std_probe']), label='Bmod np std')
        plt.xlabel('Current, mA')
        plt.ylabel('B std, mT')
        plt.legend()



    linear_parameters = {'slope' : slope_list,
                         'intercept': intercept_list}

    # a_file = open(f'coil_calibration/extracted_data/linear_parameters.json', "w")
    # json.dump(linear_parameters, a_file)
    # a_file.close()

    plt.show()
