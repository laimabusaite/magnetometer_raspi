import random

import numpy
import numpy as np
import NVcenter as nv
import matplotlib.pyplot as plt
from detect_peaks import *
from utilities import *
import json
from fit_odmr import *
from scipy import optimize


def generate_noisy_signal(omega, B_lab, D=2870, Mz_array=np.array([0, 0, 0, 0]), g_lor=5, noise_std=0.1):
    nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)

    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)

    noisy_odmr_signal = add_noise(odmr_signal, noise_std)

    return noisy_odmr_signal


def rotate(vec, theta=0, phi=0, alpha=0):
    theta = np.radians(theta)
    phi = np.radians(phi)
    alpha = np.radians(alpha)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])
    rotated_vec = np.dot(Rz, np.dot(Ry, np.dot(Rx, vec)))
    return rotated_vec


def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center c=(xc, yc) """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f_3(beta, x):
    """ implicit definition of the circle """
    return (x[0] - beta[0]) ** 2 + (x[1] - beta[1]) ** 2 - beta[2] ** 2


if __name__ == '__main__':

    # Extra magnetic field in the laboratory
    Bextra = np.array([0.5, 0.3, 0.4])
    print('Bextra:', Bextra, np.linalg.norm(Bextra))

    # Bias magnetic field
    B = 200
    theta = 80
    phi = 20
    Bbias = CartesianVector(B, theta, phi)
    print('Bbias:', Bbias)

    # initial magnetic field for fitting
    B_init = CartesianVector(210, 80, 20)
    init_params = {'B_labx': B_init[0], 'B_laby': B_init[1], 'B_labz': B_init[2],
                   'glor': 5, 'D': 2870, 'Mz1': 0,
                   'Mz2': 0, 'Mz3': 0, 'Mz4': 0}

    # frequency range
    omega = np.arange(2000, 3800, 0.4)
    all_angles = np.arange(0,360,5)
    # angle_list = sorted(random.choices(all_angles, k=360))
    axis_list = ['x', 'y', 'z']

    axis_box = np.eye(3)
    axis_diamond = rotate(axis_box, theta=0, phi=20, alpha=0)
    print('axis_diamond')
    print(axis_diamond)
    print(rotate(axis_diamond, theta=20, phi=0, alpha=0))


    # measurement process
    idx_axis = 0
    # while True:
    b_temp = []
    B_set_list =  []
    for axis in axis_list:
        theta = 0
        phi = 0
        alpha = 0
        try:
            print('Choose axis to rotate around: x, y, z')
            # axis = input()
            idx_angle = 0

            a_temp = np.array([])
            print(a_temp)
            k = random.randint(3, 20)
            angle_list = sorted(random.choices(all_angles, k=k))
            B_set_list_temp = np.array([])
            # while True:
            for a in angle_list:
                print('Enter angle:')
                my_input = a #input()
                if my_input == 'q':
                    break
                angle = float(my_input)
                print(angle)

                if axis == 'x':
                    alpha = angle
                elif axis == 'y':
                    theta = angle
                elif axis == 'z':
                    phi = angle
                else:
                    print('wrong axis')

                Bextra_rot = rotate(Bextra, theta=theta, phi=phi, alpha=alpha)
                B_bias_box = rotate(Bbias, 0, 0, 0)
                B_lab = B_bias_box + Bextra_rot
                print('B_lab:', B_lab)
                odmr = generate_noisy_signal(omega=omega, B_lab=B_lab, noise_std=0.03)
                # odmr_array[idx_axis, idx_angle, :] = odmr[:]
                a_temp = np.append(a_temp, odmr)
                a_temp = a_temp.reshape((-1, len(odmr)))

                B_set_list_temp = np.append(B_set_list_temp, B_lab)
                B_set_list_temp = B_set_list_temp.reshape(-1, len(B_lab))

                idx_angle += 1
                # print('a_temp', a_temp.shape)
                # print(a_temp)

            b_temp.append(a_temp)
            B_set_list.append(B_set_list_temp)

            # print('b_temp')
            # print(b_temp)
            idx_axis += 1
        except KeyboardInterrupt:
            break


    print(b_temp[0])
    print(b_temp[1])

    # Fit ODMR to get magnetic field values
    B_calibrated = np.zeros((len(axis_list), 3))
    for idx_axis, axis in enumerate(axis_list):
        print('axis')
        Blab_fitted_list = np.zeros((len(b_temp[idx_axis]), 3))
        Mz_fitted_list = np.zeros((len(b_temp[idx_axis]), 4))
        for idx, odmr in enumerate(b_temp[idx_axis]):
            # save_filename = f"ODMR_fit_parameters{idx+1}.json"
            parameters = fit_full_odmr(omega, odmr, init_params=init_params, save=False, debug=False)
            Blab_fitted_list[idx] = np.array([parameters["B_labx"], parameters["B_laby"], parameters["B_labz"]])
            Mz_fitted_list[idx] = np.array([parameters['Mz1'], parameters['Mz2'], parameters['Mz3'], parameters['Mz4']])

            # Blab_fitted_list[idx] = rotate(B_set_list[idx_axis][idx], theta=0, phi=0, alpha=0)


        #fit magnetic field values to circle
        # coordinates of the barycenter
        idxs_list = [0, 1, 2]
        idxs_list.remove(idx_axis)
        print(idxs_list)
        x = Blab_fitted_list[:, idxs_list[0]]
        y = Blab_fitted_list[:, idxs_list[1]]
        x_m = np.mean(x)
        y_m = np.mean(y)

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f_2(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        # initial guess for parameters
        R_m = calc_R(x_m, y_m).mean()
        beta0 = [x_m, y_m, R_m]

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        B_calibrated[idx_axis, idxs_list[0]] = xc_2
        B_calibrated[idx_axis, idxs_list[1]] = yc_2
        print(xc_2, yc_2)
        Ri_2 = calc_R(xc_2, yc_2)
        R_2 = Ri_2.mean()
        residu_2 = sum((Ri_2 - R_2) ** 2)
        residu2_2 = sum((Ri_2 ** 2 - R_2 ** 2) ** 2)

        theta_fit = np.linspace(-np.pi, np.pi, 180)
        x_fit2 = xc_2 + R_2 * np.cos(theta_fit)
        y_fit2 = yc_2 + R_2 * np.sin(theta_fit)



        plt.figure()
        plt.scatter(Blab_fitted_list[:, idxs_list[0]], Blab_fitted_list[:, idxs_list[1]])
        plt.axis('equal')
        plt.plot(x_fit2, y_fit2, 'k--', label='', lw=2)

    print('B_calibrated')
    print(B_calibrated)
    print('B_bias')
    print(Bbias)
    print('B_calibrated_rotated')
    B_calibrated[B_calibrated == 0] = np.nan
    print(B_calibrated)
    B_avg = np.nanmean(B_calibrated, axis=0)
    print('B_avg')
    print(B_avg)
    B_calibrated_rotated = rotate(B_avg, 0, 0, 0)
    print(B_calibrated_rotated)

    plt.show()


