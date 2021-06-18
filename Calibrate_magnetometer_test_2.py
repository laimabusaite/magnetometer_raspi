import numpy as np

from Calibrate_without_earth_test import *


def rotate_about_axis(vector, axis, angle):
    N_mat = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rot_mat = np.eye(3) + np.sin(angle) * N_mat + (1.0 - np.cos(angle)) * np.dot(N_mat, N_mat)
    rotated_vector = np.dot(rot_mat, vector)
    return rotated_vector


def generate_rotaion_data(Bbias, Bextra, alpha_tilted=0, phi_tilted=0, theta_tilted=0, noise_std=0.0):
    axis_diamond = np.eye(3)
    axis_box = rotate(axis_diamond, theta=theta_tilted, phi=phi_tilted, alpha=alpha_tilted).transpose()

    all_angles = np.arange(0, 360, 5)
    omega = np.arange(2000, 3800, 0.4)

    b_temp = []
    B_set_list = []
    for idx_axis, axis in enumerate(axis_box[:]):
        # print(idx_axis, axis)
        k = random.randint(3, 20)
        angle_list = all_angles  # sorted(random.choices(all_angles, k=k))
        B_set_list_temp = np.array([])
        a_temp = np.array([])
        for angle in angle_list[:]:
            Bextra_rot = rotate_about_axis(Bextra, axis, angle)
            # print(angle, Bextra, Bextra_rot, np.linalg.norm(Bextra), np.linalg.norm(Bextra_rot))
            B_lab = Bbias + Bextra_rot
            # print('B_lab:', B_lab)
            odmr = generate_noisy_signal(omega=omega, B_lab=B_lab, noise_std=noise_std)
            a_temp = np.append(a_temp, odmr)
            a_temp = a_temp.reshape((-1, len(odmr)))
            B_set_list_temp = np.append(B_set_list_temp, B_lab)
            B_set_list_temp = B_set_list_temp.reshape(-1, len(B_lab))

        b_temp.append(a_temp)
        B_set_list.append(B_set_list_temp)
    print(len(B_set_list), B_set_list[0].shape)
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
    phi_tilted = 45
    theta_tilted = 0

    noise_std = 0.

    B_set_list = generate_rotaion_data(Bbias=Bbias, Bextra=Bextra, alpha_tilted=alpha_tilted, phi_tilted=phi_tilted,
                          theta_tilted=theta_tilted, noise_std=noise_std)

    axis_list = ['x', 'y', 'z']
    for idx_axis, axis in enumerate(axis_list):
        Blab_fitted_list = B_set_list[idx_axis]


        for idx_plane in range(3):
            idxs_list = np.roll([0,1,2], -idx_plane)
            # idxs_list = [0, 1, 2]
            # idxs_list.remove(idx_axis)

            plt.figure()
            plt.title(f'Rotate around {axis}-axis')
            idx_x = min(idxs_list[0],idxs_list[1])
            idx_y = max(idxs_list[0],idxs_list[1])
            plt.scatter(Blab_fitted_list[:, idx_x], Blab_fitted_list[:, idx_y])
            plt.axis('equal')
            plt.xlabel(f'B{axis_list[idx_x]}, G')
            plt.ylabel(f'B{axis_list[idx_y]}, G')

            x = Blab_fitted_list[:, idx_x]
            y = Blab_fitted_list[:, idx_y]
            x_m = np.mean(x)
            y_m = np.mean(y)
            a = 0.58
            b = 0.58
            xy = np.array([x, y]).transpose()
            print(xy)

            ellipse = EllipseModel()
            # ellipse.params((x_m, y_m, a, b, 0))
            ellipse.estimate(xy)
            print(ellipse.params)

            try:
                t = np.linspace(0, 2 * np.pi, 25)
                xy_fitted = EllipseModel().predict_xy(t, ellipse.params).transpose()
                x_fitted = xy_fitted[0]# * np.cos(t)
                y_fitted = xy_fitted[1]# * np.sin(t)
                plt.plot(x_fitted, y_fitted)
            except Exception as e:
                print(e)
            # plt.legend()

    plt.show()