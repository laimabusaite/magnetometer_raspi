import re

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

import utilities

if __name__ == '__main__':

    r_list = [1, 2, 3, 4]
    dir_name = 'RQ1nc'
    axis = 'Z'

    dataframe_B = pd.DataFrame()

    for r in r_list:
        foldernames = [f'{dir_name}/{axis}/r{r}/{i}/' for i in range(1, 18)] # glob.glob(f'RQ1nc/Z/r{r}/*/')
        filename_coil = f'RQ1nc/Z/r{r}/RQ1_Z_R{r}_current.csv'
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
        folder_index_list = []
        for foldername in foldernames:

            foldername_split = re.split("/", foldername)
            print(foldername_split)
            folder_index = int(foldername_split[3])

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
                    folder_index_list.append(folder_index)

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
                'B_measured_std': B_std_list
        }

        dataframe = pd.DataFrame(data, index=folder_index_list)
        print(dataframe)


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
        # plt.legend()




        # set current


        dataframe_coil = pd.read_csv(filename_coil, sep=';', index_col=0, names=['current_x', 'current_y'])
        dataframe_coil['current_z'] = 0
        dataframe_coil['Bx_coil'], dataframe_coil['By_coil'], dataframe_coil['Bz_coil'] = utilities.amper2gauss_array(
            dataframe_coil['current_x'],dataframe_coil['current_y'],dataframe_coil['current_z'])

        dataframe_coil['Ix err'] = 0.5
        dataframe_coil['Iy err'] = 0.5
        dataframe_coil['Iz err'] = 0.1
        dataframe_coil['Bx_coil_std'], dataframe_coil['By_coil_std'], dataframe_coil[
            'Bz_coil_std'] = utilities.amper2gauss_array(
            dataframe_coil['Ix err'], dataframe_coil['Iy err'], dataframe_coil['Iz err'])
        print(dataframe_coil)
        plt.scatter(dataframe_coil['Bx_coil'], dataframe_coil['By_coil'], label='coil')
        # plt.legend()

        plt.figure('xz, yz plane')
        plt.scatter(dataframe['Bx_measured'], dataframe['Bz_measured'], label='x-z measured')
        plt.scatter(dataframe['By_measured'], dataframe['Bz_measured'], label='y-z measured')
        plt.xlabel('Bx (By), G')
        plt.ylabel('Bz, G')
        # plt.legend()

        plt.figure('B mod')
        plt.plot(dataframe['B_measured'])

        # plt.figure()
        # plt.plot(dataframe_x['B_measured'], label='measured')
        # plt.plot(dataframe_x['B_coil'], label='coil')
        # plt.legend()

        dataframe_B[f'Bx_coil_r{r} (mT)'] = dataframe_coil['Bx_coil']
        dataframe_B[f'By_coil_r{r} (mT)'] = dataframe_coil['By_coil']
        dataframe_B[f'Bz_coil_r{r} (mT)'] = dataframe_coil['Bz_coil']
        dataframe_B[f'B_coil_r{r} (mT)'] = np.sqrt(dataframe_coil['Bx_coil']**2 +
                                                   dataframe_coil['By_coil']**2 +
                                                   dataframe_coil['Bz_coil']**2)

        dataframe_B[f'Bx_coil_std_r{r} (mT)'] = dataframe_coil['Bx_coil_std'].values
        dataframe_B[f'By_coil_std_r{r} (mT)'] = dataframe_coil['By_coil_std'].values
        dataframe_B[f'Bz_coil_std_r{r} (mT)'] = dataframe_coil['Bz_coil_std'].values
        dataframe_B[f'B_coil_std_r{r} (mT)'] = np.sqrt(
            dataframe_coil['Bx_coil_std'].values ** 2 +
            dataframe_coil['By_coil_std'].values ** 2 +
            dataframe_coil['Bz_coil_std'].values ** 2)

        dataframe_B[f'Bx_measured_r{r} (mT)'] = dataframe['Bx_measured']
        dataframe_B[f'By_measured_r{r} (mT)'] = dataframe['By_measured']
        dataframe_B[f'Bz_measured_r{r} (mT)'] = dataframe['Bz_measured']
        dataframe_B[f'B_measured_r{r} (mT)'] = dataframe['B_measured']

        dataframe_B[f'Bx_measured_std_r{r} (mT)'] = dataframe['Bx_measured_std']
        dataframe_B[f'By_measured_std_r{r} (mT)'] = dataframe['By_measured_std']
        dataframe_B[f'Bz_measured_std_r{r} (mT)'] = dataframe['Bz_measured_std']
        dataframe_B[f'B_measured_std_r{r} (mT)'] = dataframe['B_measured_std']

    dataframe_B['Bx_coil_mean (mT)'] = dataframe_B[[f'Bx_coil_r{r} (mT)' for r in r_list]].mean(axis=1)
    dataframe_B['By_coil_mean (mT)'] = dataframe_B[[f'By_coil_r{r} (mT)' for r in r_list]].mean(axis=1)
    dataframe_B['Bz_coil_mean (mT)'] = dataframe_B[[f'Bz_coil_r{r} (mT)' for r in r_list]].mean(axis=1)
    dataframe_B['B_coil_mean (mT)'] = dataframe_B[[f'B_coil_r{r} (mT)' for r in r_list]].mean(axis=1)

    dataframe_B['Bx_coil_std_mean (mT)'] = np.sqrt(dataframe_B['Bx_coil_std_r1 (mT)'] ** 2 +
                                                   dataframe_B['Bx_coil_std_r2 (mT)'] ** 2 +
                                                   dataframe_B['Bx_coil_std_r3 (mT)'] ** 2 +
                                                   dataframe_B['Bx_coil_std_r4 (mT)'] ** 2)
    dataframe_B['By_coil_std_mean (mT)'] = np.sqrt(dataframe_B['By_coil_std_r1 (mT)'] ** 2 +
                                                   dataframe_B['By_coil_std_r2 (mT)'] ** 2 +
                                                   dataframe_B['By_coil_std_r3 (mT)'] ** 2 +
                                                   dataframe_B['By_coil_std_r4 (mT)'] ** 2)
    dataframe_B['Bz_coil_std_mean (mT)'] = np.sqrt(dataframe_B['Bz_coil_std_r1 (mT)'] ** 2 +
                                                   dataframe_B['Bz_coil_std_r2 (mT)'] ** 2 +
                                                   dataframe_B['Bz_coil_std_r3 (mT)'] ** 2 +
                                                   dataframe_B['Bz_coil_std_r4 (mT)'] ** 2)
    dataframe_B['B_coil_std_mean (mT)'] = np.sqrt(dataframe_B['B_coil_std_r1 (mT)'] ** 2 +
                                                  dataframe_B['B_coil_std_r2 (mT)'] ** 2 +
                                                  dataframe_B['B_coil_std_r3 (mT)'] ** 2 +
                                                  dataframe_B['B_coil_std_r4 (mT)'] ** 2)

    dataframe_B['Bx_measured_mean (mT)'] = dataframe_B[[f'Bx_measured_r{r} (mT)' for r in r_list]].mean(axis=1)
    dataframe_B['By_measured_mean (mT)'] = dataframe_B[[f'By_measured_r{r} (mT)' for r in r_list]].mean(axis=1)
    dataframe_B['Bz_measured_mean (mT)'] = dataframe_B[[f'Bz_measured_r{r} (mT)' for r in r_list]].mean(axis=1)
    dataframe_B['B_measured_mean (mT)'] = dataframe_B[[f'B_measured_r{r} (mT)' for r in r_list]].mean(axis=1)

    dataframe_B['Bx_measured_std_mean (mT)'] = np.sqrt(dataframe_B['Bx_measured_std_r1 (mT)']**2 +
                                                  dataframe_B['Bx_measured_std_r2 (mT)']**2 +
                                                  dataframe_B['Bx_measured_std_r3 (mT)']**2 +
                                                  dataframe_B['Bx_measured_std_r4 (mT)']**2)
    dataframe_B['By_measured_std_mean (mT)'] = np.sqrt(dataframe_B['By_measured_std_r1 (mT)'] ** 2 +
                                                  dataframe_B['By_measured_std_r2 (mT)'] ** 2 +
                                                  dataframe_B['By_measured_std_r3 (mT)'] ** 2 +
                                                  dataframe_B['By_measured_std_r4 (mT)'] ** 2)
    dataframe_B['Bz_measured_std_mean (mT)'] = np.sqrt(dataframe_B['Bz_measured_std_r1 (mT)'] ** 2 +
                                                  dataframe_B['Bz_measured_std_r2 (mT)'] ** 2 +
                                                  dataframe_B['Bz_measured_std_r3 (mT)'] ** 2 +
                                                  dataframe_B['Bz_measured_std_r4 (mT)'] ** 2)
    dataframe_B['B_measured_std_mean (mT)'] = np.sqrt(dataframe_B['B_measured_std_r1 (mT)'] ** 2 +
                                                 dataframe_B['B_measured_std_r2 (mT)'] ** 2 +
                                                  dataframe_B['B_measured_std_r3 (mT)'] ** 2 +
                                                  dataframe_B['B_measured_std_r4 (mT)'] ** 2)

    # std
    # dataframe_B['Bx_measured_std_0 (mT)'] = 0.006928  # 0.084567
    # dataframe_B['By_measured_std_0 (mT)'] = 0.007752  # 0.094622
    # dataframe_B['Bz_measured_std_0 (mT)'] = 0.010947  # 0.133624
    # dataframe_B['B_measured_std_0 (mT)'] = 0.008400  # 0.102536

    # mean
    dataframe_B['Bx_measured_std_0 (mT)'] = 0.032879
    dataframe_B['By_measured_std_0 (mT)'] = 0.062017
    dataframe_B['Bz_measured_std_0 (mT)'] = 0.154423
    dataframe_B['B_measured_std_0 (mT)'] = 0.169628

    dataframe_B['diff_Bx (mT)'] = dataframe_B['Bx_coil_mean (mT)'] - dataframe_B['Bx_measured_mean (mT)']
    dataframe_B['diff_By (mT)'] = dataframe_B['By_coil_mean (mT)'] - dataframe_B['By_measured_mean (mT)']
    dataframe_B['diff_Bz (mT)'] = dataframe_B['Bz_coil_mean (mT)'] - dataframe_B['Bz_measured_mean (mT)']
    dataframe_B['diff_B (mT)'] = dataframe_B['B_coil_mean (mT)'] - dataframe_B['B_measured_mean (mT)']

    dataframe_B /= 10.

    print(dataframe_B)
    dataframe_B.to_csv(f'{dir_name}/dataframe_B_{axis}.csv')
    plt.legend()
    plt.show()