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

    while True:
        theta = 0
        phi = 0
        alpha = 0
        try:
            print('Choose axis to rotate around: x, y, z')
            axis = input()
            while True:
                print('Enter angle:')
                my_input = input()
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
                B_lab = Bbias + Bextra_rot
                print('B_lab:', B_lab)
                odmr = generate_noisy_signal(omega=omega, B_lab=B_lab, noise_std=0.01)



        except KeyboardInterrupt:
            break





    #
    #
    # # B_init = np.array([190.30416375, 76.05192834, 37.91572616])
    # # frequencies0 = np.array([2610.1786052321295, 2902.430583691413, 3260.8360334550603, 3413.1714015015773])
    # # A_inv = np.array([[-0.09885543, -0.11814168, 0.20521331, 0.14720897],
    # #                   [-0.166928, 0.24067813, -0.11076892, 0.11649146],
    # #                   [0.2325539, 0.16654134, 0.05205758, 0.03018854]])
    #
    # # phi_list = np.arange(0,360,90)
    # phi_list = np.array([0, 15, 60, 150])
    # theta_list = numpy.zeros(len(phi_list))
    # fig = plt.figure()
    #
    # Blab_list = np.zeros((len(phi_list),3))
    # Blab_fitted_list = np.zeros((len(phi_list), 3))
    # Mz_fitted_list = np.zeros((len(phi_list), 4))
    # for idx, theta in enumerate(theta_list):
    #     phi = phi_list[idx]
    #     Bextra_rot = rotate(Bextra, theta=theta, phi=phi)
    #     print('Bextra_rot:', Bextra_rot, np.linalg.norm(Bextra_rot))
    #     B_lab = Bbias + Bextra_rot
    #     print('B_lab:', B_lab)
    #     Blab_list[idx] = B_lab
    #
    #     odmr = generate_noisy_signal(omega=omega, B_lab=B_lab, noise_std=0.01)
    #
    #     # peak_positions, peak_amplitudes = detect_peaks_simple(omega, odmr, height=0.4)
    #     # print('peak_positions:', peak_positions)
    #     # delta_fr = frequencies0 - peak_positions[1::2]
    #     # B_from_peak = B_init + deltaB_from_deltaFrequencies(A_inv, delta_fr)
    #     # print('B_from_peak:', B_from_peak)
    #     # init_params['B_labx'] = B_from_peak[0]
    #     # init_params['B_laby'] = B_from_peak[1]
    #     # init_params['B_laby'] = B_from_peak[2]
    #
    #     save_filename = f"ODMR_fit_parameters{idx+1}.json"
    #     parameters = fit_full_odmr(omega, odmr, init_params=init_params, save_filename=save_filename, debug=False)
    #     Blab_fitted_list[idx] = np.array([parameters["B_labx"], parameters["B_laby"], parameters[
    #     "B_labz"]])
    #     Mz_fitted_list[idx] = np.array([parameters['Mz1'], parameters['Mz2'], parameters['Mz3'], parameters['Mz4']])
    #
    # print('Bbias:', Bbias)
    # print('Bextra:', Bextra, np.linalg.norm(Bextra))
    # print((Blab_list[0] + Blab_list[1]) / 2.0)
    # print((Blab_list[0] + Blab_list[2]) / 2.0)
    #
    # print(np.mean(Blab_list))
    #
    # plt.scatter(Blab_list[:, 0], Blab_list[:, 1])
    # plt.scatter(Blab_fitted_list[:, 0], Blab_fitted_list[:, 1])
    #
    #
    # # coordinates of the barycenter
    # x = Blab_fitted_list[:, 0]
    # y = Blab_fitted_list[:, 1]
    # x_m = np.mean(x)
    # y_m = np.mean(y)
    #
    #
    #
    # def calc_R(xc, yc):
    #     """ calculate the distance of each 2D points from the center (xc, yc) """
    #     return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    #
    # def f_2(c):
    #     """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
    #     Ri = calc_R(*c)
    #     return Ri - Ri.mean()
    #
    # # initial guess for parameters
    # R_m = calc_R(x_m, y_m).mean()
    # beta0 = [x_m, y_m, R_m]
    #
    # center_estimate = x_m, y_m
    # center_2, ier = optimize.leastsq(f_2, center_estimate)
    #
    # xc_2, yc_2 = center_2
    # print(xc_2, yc_2)
    # Ri_2 = calc_R(xc_2, yc_2)
    # R_2 = Ri_2.mean()
    # residu_2 = sum((Ri_2 - R_2) ** 2)
    # residu2_2 = sum((Ri_2 ** 2 - R_2 ** 2) ** 2)
    # # ncalls_2 = f_2.ncalls
    #
    # # f = plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
    # plt.axis('equal')
    #
    # theta_fit = np.linspace(-np.pi, np.pi, 180)
    #
    #
    # x_fit2 = xc_2 + R_2 * np.cos(theta_fit)
    # y_fit2 = yc_2 + R_2 * np.sin(theta_fit)
    # plt.plot(x_fit2, y_fit2, 'k--', label='', lw=2)
    #
    # plt.show()
