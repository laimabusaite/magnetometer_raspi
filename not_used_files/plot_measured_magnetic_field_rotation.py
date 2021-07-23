from glob import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

import utilities

if __name__ == '__main__':
    folders = glob("R*/")

    print(folders)
    axis_list = []
    current_x_list = []
    current_y_list = []
    current_z_list = []
    Bx_coil_list = []
    By_coil_list = []
    Bz_coil_list = []
    B_coil_list = []
    Bx_list = []
    By_list = []
    Bz_list = []
    B_list = []
    Bx_std_list = []
    By_std_list = []
    Bz_std_list = []
    B_std_list = []
    for folder in folders:
        folder_split = re.split('R1|RQ1|_', folder)
        print(folder_split)
        # print(float(folder_split[4][:-1]))

        rotation_axis = folder_split[1]
        if rotation_axis == '':
            rotation_axis = 'z'

        current_x = float(folder_split[2])
        current_y = float(folder_split[3])
        current_z = float(folder_split[4][:-1])
        current = np.array([current_x, current_y, current_z])

        Bx_coil, By_coil, Bz_coil = utilities.amper2gauss_array(current_x, current_y, current_z)
        B_coil = np.linalg.norm([Bx_coil, By_coil, Bz_coil])


        print(rotation_axis, current)

        log_files = glob(f'{folder[:-1]}/*8.log')

        log_summary_files = glob(f'{folder[:-1]}/*summary.log')

        if len(log_files) == 1:
            log_file = log_files[0]
            print(log_file)
            log_summary_file = log_summary_files[0]

            B_dataframe = pd.read_csv(log_file, names=['Bx', 'By', 'Bz', 'B'], delimiter='\t')
            # print(B_dataframe)

            Bx_mean = B_dataframe['Bx'].mean()
            By_mean = B_dataframe['By'].mean()
            Bz_mean = B_dataframe['Bz'].mean()
            B_mean = B_dataframe['B'].mean()



            Bx_std = np.std(B_dataframe['Bx']) / np.sqrt(len(B_dataframe['Bx']) - 1)
            By_std = np.std(B_dataframe['By']) / np.sqrt(len(B_dataframe['By']) - 1)
            Bz_std = np.std(B_dataframe['Bz']) / np.sqrt(len(B_dataframe['Bz']) - 1)
            B_std = np.std(B_dataframe['B']) / np.sqrt(len(B_dataframe['B']) - 1)

            print(f'B = {B_mean:.2f}, std = {B_std:.2f}, rB = {np.abs(B_std / B_mean) * 100:.2f} %')
            print(f'Bx = {Bx_mean:.2f}, std = {Bx_std:.2f}, rBx = {np.abs(Bx_std / Bx_mean) * 100:.2f} %')
            print(f'B = {By_mean:.2f}, std = {By_std:.2f}, rBy = {np.abs(By_std / By_mean) * 100:.2f} %')
            print(f'B = {Bz_mean:.2f}, std = {Bz_std:.2f}, rBz = {np.abs(Bz_std / Bz_mean) * 100:.2f} %')

            f = open(log_summary_file, 'r')
            summary = f.read()
            f.close()

            print(summary)

            if len(B_dataframe) > 5:
                axis_list.append(rotation_axis)
                current_x_list.append(current_x)
                current_y_list.append(current_y)
                current_z_list.append(current_z)
                Bx_coil_list.append(Bx_coil)
                By_coil_list.append(By_coil)
                Bz_coil_list.append(Bz_coil)
                B_coil_list.append(B_coil)
                B_list.append(B_mean)
                Bx_list.append(Bx_mean)
                By_list.append(By_mean)
                Bz_list.append(Bz_mean)
                B_std_list.append(B_std)
                Bx_std_list.append(Bx_std)
                By_std_list.append(By_std)
                Bz_std_list.append(Bz_std)

    data = {'axis': axis_list,
            'current_x': current_x_list,
            'current_y': current_y_list,
            'current_z': current_z_list,
            'Bx_coil' : Bx_coil_list,
            'By_coil': By_coil_list,
            'Bz_coil': Bz_coil_list,
            'B_coil': B_coil_list,
            'Bx_measured': Bx_list,
            'By_measured': By_list,
            'Bz_measured': Bz_list,
            'B_measured': B_list,
            'Bx_measured_std': Bx_std_list,
            'By_measured_std': By_std_list,
            'Bz_measured_std': Bz_std_list,
            'B_measured_std': B_std_list}

    dataframe = pd.DataFrame(data)
    # print(dataframe)

    # rotate around x axis
    print('rotate around x axis')
    dataframe_x = dataframe[dataframe['axis'] == 'x']
    # print(dataframe_x)

    fig1 = plt.figure('x-axis')
    # plt.scatter(dataframe_x['current_y'], dataframe_x['current_z'])

    plt.title('Rotate around x axis')
    plt.scatter(dataframe_x['By_measured'], dataframe_x['Bz_measured'], label='measured')
    plt.scatter(dataframe_x['By_coil'], dataframe_x['Bz_coil'], label='coil')
    plt.xlabel('By, G')
    plt.ylabel('Bz, G')
    plt.axis('equal')
    plt.legend()

    # rotate around y axis
    print('rotate around y axis')
    dataframe_y = dataframe[dataframe['axis'] == 'y']
    # print(dataframe_x)

    plt.figure('y-axis')
    # plt.scatter(dataframe_x['current_y'], dataframe_x['current_z'])
    plt.title('Rotate around y axis')
    plt.scatter(dataframe_y['Bx_measured'], dataframe_y['Bz_measured'], label='measured')
    plt.scatter(dataframe_y['Bx_coil'], dataframe_y['Bz_coil'], label='coil')
    plt.xlabel('Bx, G')
    plt.ylabel('Bz, G')
    plt.axis('equal')
    plt.legend()

    # rotate around z axis
    print('rotate around z axis')
    dataframe_x = dataframe[dataframe['axis'] == 'z']
    # print(dataframe_x)

    plt.figure('z-axis')
    # plt.scatter(dataframe_x['current_y'], dataframe_x['current_z'])
    plt.title('Rotate around z axis')
    plt.scatter(dataframe_x['Bx_measured'], dataframe_x['By_measured'], label='measured')
    plt.scatter(dataframe_x['Bx_coil'], dataframe_x['By_coil'], label='coil')
    plt.xlabel('Bx, G')
    plt.ylabel('By, G')
    plt.axis('equal')
    plt.legend()

    # plt.figure()
    # plt.plot(dataframe_x['B_measured'], label='measured')
    # plt.plot(dataframe_x['B_coil'], label='coil')
    # plt.legend()
    plt.show()

