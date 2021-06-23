import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_odmr import *
import random
import os
import fit_circle as cf
from fit_ellipse import *

if __name__ == '__main__':

    phi_tilted = 20
    theta_tilted = 0
    filedir = f'fitted_B_angle_clean/phi_axis={phi_tilted}_theta_axis={theta_tilted}'

    # filename_list = [f'axis_idx={idx_axis}' for idx_axis in range(3)]
    # print(filename_list)

    filenames = glob.glob(f'{filedir}/*.dat')
    print(filenames)

    axis_list = ['x', 'y', 'z']
    # plane_list = ['xy', 'xz', 'yz']

    phi_fitted_list = []
    phi_fitted2_list = []
    phi_ellipse_list = []
    # plt.figure()
    # import fitted B data
    for idx_axis, filename in enumerate(filenames[:]):
        # plt.figure()
        dataframe = pd.read_csv(filename, index_col=0)
        print(dataframe.head())
        axis = axis_list[idx_axis]

        for idx_plane in range(3):
            idxs_list = np.roll([0, 1, 2], -idx_plane)

            plt.figure()
            plt.title(f'Rotate around {axis}-axis')
            # idx_x = idxs_list[0]
            # idx_y = idxs_list[1]
            idx_x = min(idxs_list[0], idxs_list[1])
            idx_y = max(idxs_list[0], idxs_list[1])
            if idxs_list[1] == idx_axis:
                idx_x = idxs_list[0]
                idx_y = idxs_list[1]
            elif idxs_list[0] == idx_axis:
                idx_x = idxs_list[1]
                idx_y = idxs_list[0]
            else:
                idx_x = min(idxs_list[0], idxs_list[1])
                idx_y = max(idxs_list[0], idxs_list[1])

            x = dataframe[f'B{axis_list[idx_x]}']
            y = dataframe[f'B{axis_list[idx_y]}']
            x -= x.mean()
            y -= y.mean()
            xy = np.array([x, y]).transpose()
            plt.scatter(dataframe[f'B{axis_list[idx_x]}'], dataframe[f'B{axis_list[idx_y]}'])
            plt.axis('equal')
            plt.xlabel(f'B{axis_list[idx_x]}, G')
            plt.ylabel(f'B{axis_list[idx_y]}, G')

            # fit ellipse and circle
            t = np.linspace(0, 2 * np.pi, 180)
            try:
                center_ellipse, phi_ellipse, axes_ellipse = find_ellipse(x, y)
                print("center = ", center_ellipse)
                print("angle of rotation = ", np.rad2deg(phi_ellipse))
                print("axes = ", axes_ellipse)
                x_ellipse2 = center_ellipse[0] + axes_ellipse[0] * np.cos(phi_ellipse) * np.cos(t) - axes_ellipse[
                    1] * np.sin(phi_ellipse) * np.sin(t)
                y_ellipse2 = center_ellipse[1] + axes_ellipse[0] * np.sin(phi_ellipse) * np.cos(t) + axes_ellipse[
                    1] * np.cos(phi_ellipse) * np.sin(t)

                a = max(axes_ellipse)
                b = min(axes_ellipse)
                print(a, b)
                e_ellipse = np.sqrt(1 - b ** 2 / a ** 2)
                angle_fit = np.rad2deg(np.arctan(b ** 2 / a ** 2))
                angle_fit2 = np.rad2deg(np.arctan(b / a))
                print('angle_fit=', angle_fit, angle_fit)

                plt.plot(x_ellipse2, y_ellipse2,
                         label=f'c=({center_ellipse[0]:.2f},{center_ellipse[1]:.2f}), angle={np.rad2deg(phi_ellipse):.2f}, axes=({axes_ellipse[0]:.2f},{axes_ellipse[1]:.2f}), e = {e_ellipse:.2f}')
                plt.legend()

                if idx_axis == 0:
                    if idx_axis == idxs_list[0]:
                        phi_ellipse_list.append(-np.rad2deg(phi_ellipse))
                    if idx_axis == idxs_list[2]:
                        y_proj = axes_ellipse[0]
                    if idx_axis == idxs_list[1]:
                        x_proj = axes_ellipse[1]
                elif idx_axis == 1:
                    if idx_plane == 0:
                        phi_ellipse_list.append(np.rad2deg(phi_ellipse))
                    if idx_plane == 1:
                        x_proj = axes_ellipse[1]
                    if idx_plane == 2:
                        y_proj = axes_ellipse[0]



                #
                # if idx_plane == 1:
                #     y_proj = axes_ellipse[0]
                # if idx_plane == 2:
                #     x_proj = axes_ellipse[1]

            except Exception as e:
                print(e)

            if idx_axis == idxs_list[2]:  # fit circle
                xc, yc, r, s = cf.hyper_fit(xy)
                x_circle = xc + r * np.cos(t)
                y_circle = yc + r * np.sin(t)
                plt.plot(x_circle, y_circle, label=f'c=({xc:.2f},{yc:.2f}), r={r:.2f}')
                plt.legend()


            else:  # fit line
                slope, intercept = cf.fit_linear(x, y)
                y_linear = slope * x + intercept
                slope_angle = np.rad2deg(np.arctan(slope))
                if idx_plane == 0:
                    if idx_axis == 0:
                        phi_fitted = -slope_angle
                    elif idx_axis == 1:
                        phi_fitted = slope_angle
                plt.plot(x, y_linear, label=f'{slope:.2} x + {intercept:.2}, {slope_angle:.2f}')
                plt.legend()
                # if idx_axis == idxs_list[0]:
                #     if idx_axis == 0:
                #         phi_fitted = np.rad2deg(np.arctan(-slope))
                #     elif idx_axis == 1:
                #         phi_fitted = np.rad2deg(np.arctan(slope))
                    # print('phi_fitted = ', phi_fitted)


        if (axis == 'x') or (axis=='y'):
            phi_fitted_2 = np.rad2deg(np.arctan(x_proj/y_proj))
            print('y_proj', 'x_proj')
            print(y_proj, x_proj)
            print('phi_fitted', phi_fitted, phi_fitted_2)
            phi_fitted_list.append(phi_fitted)
            phi_fitted2_list.append(phi_fitted_2)

    print(phi_fitted_list, phi_fitted2_list)
    phi_fitted_list_all = np.append(phi_fitted_list, phi_fitted2_list)
    print(phi_fitted_list_all)
    print(f'phi_fitted_mean = {np.mean(phi_fitted_list)}')
    print(f'phi_fitted2_mean = {np.mean(phi_fitted2_list)}')
    print(f'phi_mean = {phi_fitted_list_all.mean()}')
    print('phi_ellipse_list')
    print(phi_ellipse_list)
    print('phi_ellipse mean = ', np.mean(phi_ellipse_list))
    # plt.legend()
    plt.show()
