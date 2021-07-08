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
        filepath0_list = glob.glob(f'{dir0}/{dir_axis}/*.txt')
        print(filepath0_list)

        current_x_list = []
        current_y_list = []
        current_z_list = []
        Bx_list = []
        By_list = []
        Bz_list = []
        Bmod_list = []
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

            dataframe = pd.read_csv(filepath, sep='\t', skiprows=2, header=0)
            # print(dataframe)

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

            print(Bx_mean, By_mean, Bz_mean, Bmod_mean)

        data_axis = {
            'current_x_coil': current_x_list,
            'current_y_coil': current_y_list,
            'current_z_coil': current_z_list,
            'Bx_probe': Bx_list,
            'By_probe': By_list,
            'Bz_probe': Bz_list,
            'Bmod_probe': Bmod_list,
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
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bx_probe'], label='Bx')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['By_probe'], label='By')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bz_probe'], label='Bz')
        plt.scatter(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bmod_probe'], label='Bmod')
        plt.plot(dataframe_axis[f'current_{dir_axis}_coil'], dataframe_axis['Bmod_linear'], label=f'Blinear {slope:.5f} x + {intercept:.5f}')

        plt.xlabel('Current, mA')
        plt.ylabel('B, mT')
        plt.legend()

        print(dataframe_axis)
        # dataframe_axis.to_csv(f'coil_calibration/extracted_data/{dir_axis}_coil.dat', index=False)


    linear_parameters = {'slope' : slope_list,
                         'intercept': intercept_list}

    # a_file = open(f'coil_calibration/extracted_data/linear_parameters.json', "w")
    # json.dump(linear_parameters, a_file)
    # a_file.close()

    plt.show()
