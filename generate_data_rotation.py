import matplotlib.pyplot as plt
import numpy as np

from Calibrate_without_earth_test import *
from fit_ellipse import *
import fit_circle as cf
import os


def rotate_about_axis(vector, axis, angle):
    N_mat = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rot_mat = np.eye(3) + np.sin(angle) * N_mat + (1.0 - np.cos(angle)) * np.dot(N_mat, N_mat)
    rotated_vector = np.dot(rot_mat, vector)
    return rotated_vector


def generate_rotaion_data(Bbias, Bextra, alpha_tilted=0, phi_tilted=0, theta_tilted=0, noise_std=0.0, savedir='generated_data'):
    axis_diamond = np.eye(3)
    axis_box = rotate(axis_diamond, theta=theta_tilted, phi=phi_tilted, alpha=alpha_tilted).transpose()

    all_angles = np.arange(0, 360, 0.5)
    omega = np.arange(2000, 3800, 0.4)
    # initial magnetic field for fitting
    # B_init = CartesianVector(210, 80, 20)
    # init_params = {'B_labx': B_init[0], 'B_laby': B_init[1], 'B_labz': B_init[2],
    #                'glor': 5, 'D': 2870, 'Mz1': 2,
    #                'Mz2': 2, 'Mz3': 2, 'Mz4': 2}

    b_temp = []
    B_set_list = []
    for idx_axis, axis in enumerate(axis_box[:]):
        # print(idx_axis, axis)
        # k = random.randint(10, 60)
        # print('k =', k)
        angle_list = all_angles  #sorted(random.choices(all_angles, k=k)) #
        print(len(angle_list))
        B_set_list_temp = np.array([])
        Blab_fitted_list = np.array([])
        a_temp = np.array([])
        for angle in angle_list[:]:
            # print(k, idx_axis, angle)
            Bextra_rot = rotate_about_axis(Bextra, axis, angle)
            # print(angle, Bextra, Bextra_rot, np.linalg.norm(Bextra), np.linalg.norm(Bextra_rot))
            B_lab = Bbias + Bextra_rot
            # print('B_lab:', B_lab)
            odmr = generate_noisy_signal(omega=omega, B_lab=B_lab, noise_std=noise_std)
            a_temp = np.append(a_temp, odmr)
            a_temp = a_temp.reshape((-1, len(odmr)))
            B_set_list_temp = np.append(B_set_list_temp, B_lab)
            B_set_list_temp = B_set_list_temp.reshape(-1, len(B_lab))

            # parameters = fit_full_odmr(omega, odmr, init_params=init_params, save=False, debug=False)
            # B_lab_fitted = np.array([parameters["B_labx"], parameters["B_laby"], parameters["B_labz"]])
            # Blab_fitted_list = np.append(Blab_fitted_list, B_lab_fitted)
            # Blab_fitted_list = Blab_fitted_list.reshape(-1, len(B_lab_fitted))
            # Mz_fitted_list = np.array([parameters['Mz1'], parameters['Mz2'], parameters['Mz3'], parameters['Mz4']])


            if not os.path.exists(f'{savedir}/axis_idx={idx_axis}'):
                os.makedirs(f'{savedir}/axis_idx={idx_axis}')

            save_filename = f'{savedir}/axis_idx={idx_axis}/Bbias={Bbias[0]:.2f}-{Bbias[1]:.2f}-{Bbias[1]:.2f}_Bextra={Bextra[0]:.2f}-{Bextra[1]:.2f}-{Bextra[1]:.2f}_axis_idx={idx_axis}_axis={axis[0]:.2f}-{axis[1]:.2f}-{axis[2]:.2f}_angle={angle}_noise={noise_std}.dat'
            dataframe = pd.DataFrame(data={'MW':omega, 'ODMR':odmr})
            dataframe.to_csv(save_filename, index=False, header=False)

        b_temp.append(a_temp)
        B_set_list.append(B_set_list_temp)
        # B_set_list.append(Blab_fitted_list)
    print(len(B_set_list), B_set_list[0].shape, B_set_list[1].shape, B_set_list[2].shape)
    return B_set_list


if __name__ == '__main__':
    # Extra magnetic field in the laboratory
    Bextra = np.array([0.5, 0.3, 0.4])
    # print('Bextra:', Bextra, np.linalg.norm(Bextra))
    # Bias magnetic field
    B = 200
    theta = 80
    phi = 20
    Bbias = CartesianVector(B, theta, phi)
    alpha_tilted = 0
    phi_tilted = 0
    theta_tilted = 0

    noise_std = 0.01

    for phi_tilted in np.arange(0, 45, 5):
        # for theta_tilted in np.arange(0, 45, 5):
        savedir = f'generated_data/phi_axis={phi_tilted}_theta_axis={theta_tilted}'
        B_set_list = generate_rotaion_data(Bbias=Bbias, Bextra=Bextra, alpha_tilted=alpha_tilted, phi_tilted=phi_tilted,
                      theta_tilted=theta_tilted, noise_std=noise_std, savedir=savedir)
    #
    # axis_list = ['x', 'y', 'z']
    # for idx_axis, axis in enumerate(axis_list):
    #     Blab_fitted_list = B_set_list[idx_axis]
    #
    #
    #     for idx_plane in range(3):
    #         idxs_list = np.roll([0,1,2], -idx_plane)
    #         # idxs_list = [0, 1, 2]
    #         # idxs_list.remove(idx_axis)
    #         # if idx_axis != idxs_list[2]:
    #         #     continue
    #
    #         plt.figure()
    #         plt.title(f'Rotate around {axis}-axis')
    #         if idxs_list[1] == idx_axis:
    #             idx_x = idxs_list[0]
    #             idx_y = idxs_list[1]
    #         elif idxs_list[0] == idx_axis:
    #             idx_x = idxs_list[1]
    #             idx_y = idxs_list[0]
    #         else:
    #             idx_x = min(idxs_list[0],idxs_list[1])
    #             idx_y = max(idxs_list[0],idxs_list[1])
    #         plt.scatter(Blab_fitted_list[:, idx_x], Blab_fitted_list[:, idx_y])
    #         plt.axis('equal')
    #         plt.xlabel(f'B{axis_list[idx_x]}, G')
    #         plt.ylabel(f'B{axis_list[idx_y]}, G')
    #
    #         x = Blab_fitted_list[:, idx_x]
    #         y = Blab_fitted_list[:, idx_y]
    #         x_m = np.mean(x)
    #         y_m = np.mean(y)
    #         a = 0.58
    #         b = 0.58
    #         xy = np.array([x, y]).transpose()
    #         # print(xy)
    #
    #         # ellipse = EllipseModel()
    #         # # ellipse.params((x_m, y_m, a, b, 0))
    #         # ellipse.estimate(xy)
    #         # print(ellipse.params)
    #         #
    #         t = np.linspace(0, 2 * np.pi, 25)
    #         # try:
    #         #     xy_fitted = EllipseModel().predict_xy(t, params=ellipse.params).transpose()
    #         #     x_fitted = xy_fitted[0]# * np.cos(t)
    #         #     y_fitted = xy_fitted[1]# * np.sin(t)
    #         #     plt.plot(x_fitted, y_fitted)
    #         # except Exception as e:
    #         #     print(e)
    #         # # plt.legend()
    #
    #         try:
    #             center_ellipse, phi_ellipse, axes_ellipse = find_ellipse(x, y)
    #             print("center = ", center_ellipse)
    #             print("angle of rotation = ", np.rad2deg(phi_ellipse))
    #             print("axes = ", axes_ellipse)
    #             x_ellipse2 = center_ellipse[0] + axes_ellipse[0] * np.cos(phi_ellipse) * np.cos(t) - axes_ellipse[1] * np.sin(phi_ellipse) * np.sin(t)
    #             y_ellipse2 = center_ellipse[1] + axes_ellipse[0] * np.sin(phi_ellipse) * np.cos(t) + axes_ellipse[1] * np.cos(phi_ellipse) * np.sin(t)
    #
    #             plt.plot(x_ellipse2, y_ellipse2, label=f'c=({center_ellipse[0]:.2f},{center_ellipse[1]:.2f}), angle={np.rad2deg(phi_ellipse):.2f}, axes=({axes_ellipse[0]:.2f},{axes_ellipse[1]:.2f})')
    #             plt.legend()
    #         except Exception as e:
    #             print(e)
    #
    #         if idx_axis == idxs_list[2]:
    #             xc, yc, r, s = cf.hyper_fit(xy)
    #             x_circle = xc + r * np.cos(t)
    #             y_circle = yc + r * np.sin(t)
    #             plt.plot(x_circle, y_circle, label = f'c=({xc:.2f},{yc:.2f}), r={r:.2f}')
    #             plt.legend()


    plt.show()