import glob

import numpy as np

from fit_odmr import *
import random
import os

if __name__ == '__main__':

    phi_tilted = 0
    theta_tilted = 0

    filedir0 = f'generated_data/phi_axis={phi_tilted}_theta_axis={theta_tilted}/'
    # filenames = glob.glob(f'{filedir}/*.dat')
    # print(len(filenames))

    savedir = f'fitted_B_angle/phi_axis={phi_tilted}_theta_axis={theta_tilted}'

    B_init = CartesianVector(210, 80, 20)
    init_params = {'B_labx': B_init[0], 'B_laby': B_init[1], 'B_labz': B_init[2],
                   'glor': 5, 'D': 2870, 'Mz1': 0,
                   'Mz2': 0, 'Mz3': 0, 'Mz4': 0}

    B_fitted_list_axis = []
    Mz_fitted_list_all = []
    D_fitted_list_all = []
    axis_list = ['x', 'y', 'z']
    for idx_axis, axis in enumerate(axis_list):
        filedir1 = f'axis_idx={idx_axis}'
        filenames = glob.glob(f'{filedir0}/{filedir1}/*.dat')
        # print(len(filenames))
        k = random.randint(10, 60)
        filenames_random = sorted(random.choices(filenames, k=k))
        print(f'{len(filenames_random)}/{len(filenames)}')
        B_fitted_list_angle = []
        Mz_fitted_list_angle = []
        D_fitted_angle = []

        B_fitted_array_angle = np.empty((len(filenames_random), 3))
        Mz_fitted_array_angle = np.empty((len(filenames_random), 4))
        D_fitted_array_angle = np.empty(len(filenames_random))
        for idx_angle, filename in enumerate(filenames_random):
            print(filename)
            dataframe = pd.read_csv(filename, names=['MW', 'ODMR'])
            omega = np.array(dataframe['MW'])
            odmr = np.array(dataframe['ODMR'])
            parameters = fit_full_odmr(omega, odmr, init_params=init_params, save=False, debug=False)
            B_lab_fitted = [parameters["B_labx"], parameters["B_laby"], parameters["B_labz"]]
            Mz_fitted_list = [parameters['Mz1'], parameters['Mz2'], parameters['Mz3'], parameters['Mz4']]
            D_fitted = parameters['D']

            B_fitted_array_angle[idx_angle] = np.array(B_lab_fitted)
            Mz_fitted_array_angle[idx_angle] = np.array(Mz_fitted_list)
            D_fitted_array_angle[idx_angle] = D_fitted

            B_fitted_list_angle.append(B_lab_fitted)
            Mz_fitted_list_all.append(Mz_fitted_list)
            D_fitted_list_all.append(D_fitted)

        print(Mz_fitted_array_angle.mean(axis=0))
        print(D_fitted_array_angle.mean())
        print(np.array(B_fitted_list_angle).shape)
        B_fitted_list_axis.append(np.array(B_fitted_list_angle))

        # B_fitted_array_angle = np.array(B_fitted_list_angle)
        df = pd.DataFrame(
            data = {'Bx': B_fitted_array_angle[:,0], 'By': B_fitted_array_angle[:,1], 'Bz': B_fitted_array_angle[:,2],
                    'Mz1':Mz_fitted_array_angle[:,0], 'Mz2':Mz_fitted_array_angle[:,1],
                    'Mz3':Mz_fitted_array_angle[:,2], 'Mz4':Mz_fitted_array_angle[:,3],
                    'D': D_fitted_array_angle[:]})
        if not os.path.exists(f'{savedir}'):
            os.makedirs(f'{savedir}')
        df.to_csv(f'{savedir}/{filedir1}.dat')




