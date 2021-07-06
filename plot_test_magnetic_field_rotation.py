import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

import utilities

if __name__ == '__main__':


    foldernames = glob.glob('RQ1nc/Z/r2/*/')
    filename_coil = 'RQ1nc/Z/r2/RQ1_Z_R2_current.csv'
    print(foldernames)
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
    for foldername in foldernames:
        log_files = glob.glob(f'{foldername[:-1]}/*8.log')

        log_summary_files = glob.glob(f'{foldername[:-1]}/*summary.log')

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
                # axis_list.append(rotation_axis)
                # current_x_list.append(current_x)
                # current_y_list.append(current_y)
                # current_z_list.append(current_z)
                # Bx_coil_list.append(Bx_coil)
                # By_coil_list.append(By_coil)
                # Bz_coil_list.append(Bz_coil)
                # B_coil_list.append(B_coil)
                B_list.append(B_mean)
                Bx_list.append(Bx_mean)
                By_list.append(By_mean)
                Bz_list.append(Bz_mean)
                B_std_list.append(B_std)
                Bx_std_list.append(Bx_std)
                By_std_list.append(By_std)
                Bz_std_list.append(Bz_std)

    data = {
        # 'axis': axis_list,
            # 'current_x': current_x_list,
            # 'current_y': current_y_list,
            # 'current_z': current_z_list,
            # 'Bx_coil': Bx_coil_list,
            # 'By_coil': By_coil_list,
            # 'Bz_coil': Bz_coil_list,
            # 'B_coil': B_coil_list,
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


    # rotate around z axis
    print('rotate around z axis')
    # dataframe_x = dataframe[dataframe['axis'] == 'z']
    # print(dataframe_x)

    plt.figure('z-axis')
    # plt.scatter(dataframe_x['current_y'], dataframe_x['current_z'])
    plt.title('Rotate around z axis')
    plt.scatter(dataframe['Bx_measured'], dataframe['By_measured'], label='measured')
    # plt.scatter(dataframe['Bx_coil'], dataframe['By_coil'], label=

    plt.xlabel('Bx, G')
    plt.ylabel('By, G')
    plt.axis('equal')
    plt.legend()







    dataframe_coil = pd.read_csv(filename_coil, sep=';', index_col=0, names=['current_x', 'current_y'])
    dataframe_coil['current_z'] = 0
    dataframe_coil['Bx_coil'], dataframe_coil['By_coil'], dataframe_coil['Bz_coil'] = utilities.amper2gauss_array(dataframe_coil['current_x'],dataframe_coil['current_y'],dataframe_coil['current_z'])

    print(dataframe_coil)
    plt.scatter(dataframe_coil['Bx_coil'], dataframe_coil['By_coil'], label='coil')
    plt.legend()

    plt.figure()
    plt.scatter(dataframe['Bx_measured'], dataframe['Bz_measured'], label='x-z measured')
    plt.scatter(dataframe['By_measured'], dataframe['Bz_measured'], label='y-z measured')
    plt.xlabel('Bx (By), G')
    plt.ylabel('Bz, G')
    plt.legend()

    plt.figure()
    plt.plot(dataframe['B_measured'])

    # plt.figure()
    # plt.plot(dataframe_x['B_measured'], label='measured')
    # plt.plot(dataframe_x['B_coil'], label='coil')
    # plt.legend()
    plt.show()