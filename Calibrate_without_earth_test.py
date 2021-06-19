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
from skimage.measure import EllipseModel


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
    all_angles = np.arange(0, 360, 5)
    # angle_list = sorted(random.choices(all_angles, k=360))
    axis_list = ['x', 'y', 'z']
    #
    # axis_box = np.eye(3)
    # axis_diamond = rotate(axis_box, theta=0, phi=20, alpha=0)
    # print('axis_diamond')
    # print(axis_diamond)
    # print(rotate(axis_diamond, theta=20, phi=0, alpha=0))

    alpha_tilted = 0
    phi_tilted = 45
    theta_tilted = 0

    alpha_rotate_B = 0
    phi_rotate_B = 0
    theta_rotate_B = 0

    # measurement process
    idx_axis = 0
    # while True:
    b_temp = []
    B_set_list = []
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
            angle_list = all_angles  # sorted(random.choices(all_angles, k=k))
            B_set_list_temp = np.array([])
            # while True:
            for a in angle_list:
                print('Enter angle:')
                my_input = a  # input()
                if my_input == 'q':
                    break
                angle = float(my_input)
                print(angle)

                if axis == 'x':
                    alpha = angle
                    phi = phi_tilted
                    theta = theta_tilted
                elif axis == 'y':
                    theta = angle
                    alpha = alpha_tilted
                    phi = phi_tilted
                elif axis == 'z':
                    phi = angle
                    alpha = alpha_tilted
                    theta = theta_tilted
                else:
                    print('wrong axis')

                Bextra_rot = rotate(Bextra, theta=theta, phi=phi, alpha=alpha)
                B_bias_box = rotate(Bbias, theta=theta_tilted, phi=phi_tilted, alpha=alpha_tilted)
                B_lab = B_bias_box + Bextra_rot
                print('B_lab:', B_lab)
                odmr = generate_noisy_signal(omega=omega, B_lab=B_lab, noise_std=0.00)
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
            # parameters = fit_full_odmr(omega, odmr, init_params=init_params, save=False, debug=False)
            # Blab_fitted_list[idx] = np.array([parameters["B_labx"], parameters["B_laby"], parameters["B_labz"]])
            # Mz_fitted_list[idx] = np.array([parameters['Mz1'], parameters['Mz2'], parameters['Mz3'], parameters['Mz4']])

            Blab_fitted_list[idx] = rotate(B_set_list[idx_axis][idx], theta=theta_rotate_B, phi=phi_rotate_B,
                                           alpha=alpha_rotate_B)

        # fit magnetic field values to circle
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


        # theta = np.linspace(0, 360, 360)
        def calc_R_ellipse(xc, yc, a, b):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 / a ** 2 + (y - yc) ** 2 / b ** 2)
            # return a * b / np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


        def f_2_ellipse(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R_ellipse(*c)
            return Ri - 1  # Ri.mean()


        center_estimate_ellipse = x_m, y_m, max(x - x_m), max(y - y_m)
        center_2_ellipse, ier = optimize.leastsq(f_2_ellipse, center_estimate_ellipse)
        xc_2_ellipse, yc_2_ellipse, a_ellipse, b_ellipse = center_2_ellipse
        Ri_2_ellipse = calc_R_ellipse(xc_2_ellipse, yc_2_ellipse, a_ellipse, b_ellipse)
        R_2_ellipse = Ri_2_ellipse.mean()
        x_fit2_ellipse = xc_2_ellipse + a_ellipse * np.cos(theta_fit)
        y_fit2_ellipse = yc_2_ellipse + b_ellipse * np.sin(theta_fit)
        print('center_2_ellipse')
        print(center_2_ellipse, R_2_ellipse)

        plt.figure()
        plt.title(f'Rotate around {axis}-axis')
        plt.scatter(Blab_fitted_list[:, idxs_list[0]], Blab_fitted_list[:, idxs_list[1]])
        plt.axis('equal')
        plt.plot(x_fit2, y_fit2, 'k--', label='', lw=2)
        plt.scatter(xc_2, yc_2)
        plt.annotate(f'({xc_2:.2f},{yc_2:.2f})', center_2)
        plt.plot(x_fit2_ellipse, y_fit2_ellipse, 'r--',
                 label=f'{xc_2_ellipse:.2f},{yc_2_ellipse:.2f},{a_ellipse:.2f}, {b_ellipse:.2f}, {a_ellipse / b_ellipse:.2f}, {b_ellipse / a_ellipse:.2f}',
                 lw=2)
        plt.scatter(xc_2_ellipse, yc_2_ellipse)
        # plt.annotate(f'({xc_2_ellipse:.2f},{yc_2_ellipse:.2f},{a_ellipse:.2f},{b_ellipse:.2f})', (xc_2_ellipse, yc_2_ellipse))
        plt.xlabel(f'B{axis_list[idxs_list[0]]}, G')
        plt.ylabel(f'B{axis_list[idxs_list[1]]}, G')
        plt.legend()

        # second plane
        x = Blab_fitted_list[:, idxs_list[0]]
        y = Blab_fitted_list[:, idx_axis]


        # fit linear
        def calc_linear(slope, intercept):
            return slope * x + intercept


        def f_linear(c):
            y_2 = calc_linear(*c)
            return y - y_2


        linear_estimate = (max(y) - min(y)) / (max(x) - min(x)), 0
        linear_params, ier = optimize.leastsq(f_linear, linear_estimate)
        slope_lin, intercept_lin = linear_params
        slope = slope_lin.mean()
        intercept = intercept_lin.mean()
        x_fit_linear = x
        y_fit_linear = intercept + slope * x
        plt.figure()
        plt.title(f'Rotate around {axis}-axis')
        plt.scatter(Blab_fitted_list[:, idxs_list[0]], Blab_fitted_list[:, idx_axis])
        plt.plot(x_fit_linear, y_fit_linear, 'r--',
                 label=f'{intercept:.2f} + {slope:.2f} * x, {np.rad2deg(np.arctan(slope)):.2f}', lw=2)
        plt.axis('equal')
        plt.xlabel(f'B{axis_list[idxs_list[0]]}, G')
        plt.ylabel(f'B{axis_list[idx_axis]}, G')
        plt.legend()

        # third plane
        x = Blab_fitted_list[:, idxs_list[1]]
        y = Blab_fitted_list[:, idx_axis]
        x_m = np.mean(x)
        y_m = np.mean(y)


        def calc_R_ellipse(xc, yc, a, b, alfa):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            # return np.sqrt((x - xc) ** 2 / a ** 2 + (y - yc) ** 2 / b ** 2)
            # return a * b / np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            # alfa = np.deg2rad(alfa_deg)
            return (np.cos(alfa) ** 2 / a ** 2 + np.sin(alfa) ** 2 / b ** 2) * (x - xc) ** 2 + 2 * np.cos(
                alfa) * np.sin(alfa) * (1 / a ** 2 - 1 / b ** 2) * (x - xc) * (y - yc) + (
                           np.sin(alfa) ** 2 / a ** 2 + np.cos(alfa) ** 2 / b ** 2) * (y - yc) ** 2


        def f_2_ellipse(c):
            """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R_ellipse(*c)
            return Ri - 1  # Ri.mean()


        center_estimate_ellipse = x_m, y_m, max(x - x_m), max(y - y_m), 0
        center_2_ellipse, ier = optimize.leastsq(f_2_ellipse, center_estimate_ellipse)
        xc_2_ellipse, yc_2_ellipse, a_ellipse, b_ellipse, alfa_ellipse = center_2_ellipse
        Ri_2_ellipse = calc_R_ellipse(xc_2_ellipse, yc_2_ellipse, a_ellipse, b_ellipse, alfa_ellipse)
        R_2_ellipse = Ri_2_ellipse.mean()
        x_fit2_ellipse = xc_2_ellipse + a_ellipse * np.cos(alfa_ellipse) * np.cos(theta_fit) - b_ellipse * np.sin(
            alfa_ellipse) * np.sin(theta_fit)
        y_fit2_ellipse = yc_2_ellipse + a_ellipse * np.cos(alfa_ellipse) * np.sin(theta_fit) + b_ellipse * np.cos(
            alfa_ellipse) * np.sin(theta_fit)
        print('center_2_ellipse')
        print(center_2_ellipse, R_2_ellipse)
        plt.figure()
        plt.title(f'Rotate around {axis}-axis')
        plt.scatter(Blab_fitted_list[:, idxs_list[1]], Blab_fitted_list[:, idx_axis])
        plt.axis('equal')
        plt.plot(x_fit2_ellipse, y_fit2_ellipse, 'r--',
                 label=f'{xc_2_ellipse:.2f},{yc_2_ellipse:.2f},{a_ellipse:.2f}, {b_ellipse:.2f}, {a_ellipse / b_ellipse:.2f}, {b_ellipse / a_ellipse:.2f}, {np.rad2deg(alfa_ellipse):.2f}',
                 lw=2)
        plt.scatter(xc_2_ellipse, yc_2_ellipse)
        # plt.annotate(f'({xc_2_ellipse:.2f},{yc_2_ellipse:.2f},{a_ellipse:.2f},{b_ellipse:.2f})',
        # (xc_2_ellipse, yc_2_ellipse))
        plt.xlabel(f'B{axis_list[idxs_list[1]]}, G')
        plt.ylabel(f'B{axis_list[idx_axis]}, G')
        plt.legend()

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
    print('B_bias')
    print(Bbias)
    # B_calibrated_rotated = rotate(B_avg, 0, -45, 0)
    # print(B_calibrated_rotated)

    plt.show()
