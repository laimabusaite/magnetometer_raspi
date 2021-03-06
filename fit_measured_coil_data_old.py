import re

import pandas as pd

from fit_odmr import *
import glob
import os





if __name__ == '__main__':

    axes_list = ['X', 'Y', 'Z']
    foldernames = [f'CURR_{axis}' for axis in axes_list]

    savedir = 'crystal_axis_calibration'
    if not os.path.exists(f'{savedir}'):
        os.makedirs(f'{savedir}')

    df = pd.DataFrame()
    idx_all = 0
    for idx_axis, foldername in enumerate(foldernames):
        # foldername = 'CURR_Z'
        axis_name = axes_list[idx_axis]

        filenames = sorted(glob.glob(f'{foldername}/full_scan*.dat'))

        print(filenames)

        B = 200
        theta = 80
        phi = 20
        Blab = CartesianVector(B, theta, phi)
        # Blab = get_initial_magnetic_field(x_data, y_data) #CartesianVector(B, theta, phi)
        print(Blab)

        B = np.linalg.norm(Blab)
        print(B)

        init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2],
                       'glor': 5, 'D': 2870, 'Mz1': 0,
                       'Mz2': 0, 'Mz3': 0, 'Mz4': 0}



        for idx, filename in enumerate(filenames):
            split_name = re.split('_', filename)
            print(split_name)
            current_value = float(split_name[5])
            print(current_value)

            dataframe = import_data(filename)
            print(dataframe.head())

            x_data = dataframe['MW']
            y_data = dataframe['ODMR']

            parameters = fit_full_odmr(x_data, y_data, init_params=init_params, debug=False,
                                       save=False)
            print(parameters)

            init_params = parameters

            parameters_temp = parameters
            parameters_temp[f'CURR_{axis_name}'] = current_value
            print(amper2gauss(current=current_value, axis=axis_name))
            parameters_temp[f'coil_B{axis_name}'] = amper2gauss(current=current_value, axis=axis_name)
            df_temp = pd.DataFrame(data=parameters_temp, index=[idx_all])
            df = df.append(df_temp)
            idx_all += 1
    print(df)
    df.to_csv(f'{savedir}/coil_axis_calibration.dat')
    # plt.show()