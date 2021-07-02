import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_odmr import *
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
from Calibrate_without_earth_test import rotate
from generate_data_rotation import rotate_about_axis

if __name__ == '__main__':

    filename = 'crystal_axis_calibration/coil_axis_calibration.dat'

    df = pd.read_csv(filename, index_col=0)

    print(df)
    curr_list = ['CURR_X', 'CURR_Y', 'CURR_Z']

    axes_coordinates = np.zeros((3,3))
    # axes_coordinates_rotated = np.zeros((3, 3))

    axes_box = np.eye(3)
    angles_list = np.zeros(3)

    ax = Axes3D(plt.figure())

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    for idx, curr in enumerate(curr_list):

        df_temp = df[df[curr].notnull()].dropna(axis=1)

        Bx0 = df_temp[df_temp[curr] == 0]['B_labx'].values[0]
        By0 = df_temp[df_temp[curr] == 0]['B_laby'].values[0]
        Bz0 = df_temp[df_temp[curr] == 0]['B_labz'].values[0]
        print(Bx0, By0, Bz0)
        # print(df_x)
        df_temp['delta_Bx'] = df_temp['B_labx'] - Bx0
        df_temp['delta_By'] = df_temp['B_laby'] - By0
        df_temp['delta_Bz'] = df_temp['B_labz'] - Bz0
        print(df_temp)
        x = df_temp['delta_Bx'].values
        y = df_temp['delta_By'].values
        z = df_temp['delta_Bz'].values

        data = np.concatenate((x[:, np.newaxis],
                               y[:, np.newaxis],
                               z[:, np.newaxis]),
                              axis=1)


        print(data)

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data.mean(axis=0)
        print(datamean)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data)

        print(uu)
        print(dd)
        print(vv)


        # Now vv[0] contains the first principal component, i.e. the direction
        # vector of the 'best fit' line in the least squares sense.

        # Now generate some points along this best fit line, for plotting.

        # I use -7, 7 since the spread of the data is roughly 14
        # and we want it to have mean 0 (like the points we did
        # the svd on). Also, it's a straight line, so we only need 2 points.
        linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

        axis_crystal = vv[0]
        axes_coordinates[idx] = axis_crystal

        # shift by the mean to get the line in the right place
        linepts += datamean

        # Verify that everything looks right.
        ax.scatter3D(*data.T, c=f'C{idx}')
        ax.plot3D(*linepts.T, c=f'C{idx}')

    print('axes_coordinates')
    print(axes_coordinates)

    #cross product - test if orthogonal
    print('test orthogonal')
    cross_xy = np.cross(axes_coordinates[0], axes_coordinates[1])
    cross_yz = np.cross(axes_coordinates[1], axes_coordinates[2])
    cross_zx = np.cross(axes_coordinates[2], axes_coordinates[0])
    print(cross_xy, axes_coordinates[2], cross_xy-axes_coordinates[2], np.dot(cross_xy, axes_coordinates[2]))
    print(cross_yz, axes_coordinates[0], cross_yz-axes_coordinates[0], np.dot(cross_yz, axes_coordinates[0]))
    print(cross_zx, axes_coordinates[1], cross_zx-axes_coordinates[1], np.dot(cross_zx, axes_coordinates[1]))


    # angle between Z and z axis
    # axis_Z = np.array([0, 0, 1])
    # dotproduct = np.dot(axis_Z, axes_coordinates[2])
    # norm = np.linalg.norm(axis_Z) * np.linalg.norm(axes_coordinates[2])
    # theta_rad = np.arccos(dotproduct / norm)
    # theta_deg = np.rad2deg(theta_rad)
    # cross = np.cross(axis_Z, axes_coordinates[2])
    # axis_theta = cross / np.linalg.norm(cross)
    # print('theta_deg = ', theta_deg, 'axis =', axis_theta)

    # x-z projection - determine theta
    axis_Z = np.array([0, 1])
    m, b = np.polyfit(z, x, 1)
    # z_fit = m * x + b
    x_fit = m * z + b
    print(m, b)
    axis_z1 = np.array([1, 1. / m])
    axis_z1 /= np.linalg.norm(axis_z1)
    ax1.scatter(x, z)
    ax1.plot(x_fit, z)
    dotproduct = np.dot(axis_z1, axis_Z)
    angle_rad = np.arccos(dotproduct)
    theta_deg = np.rad2deg(angle_rad)
    print('theta_deg =', theta_deg)

    # y-z projection - determine alpha
    axis_Z = np.array([0, 1])
    m, b = np.polyfit(y, z, 1)
    z_fit = m * y + b
    print(m, b)
    axis_z2 = np.array([1, m])
    axis_z2 /= np.linalg.norm(axis_z2)
    ax1.scatter(y, z)
    ax1.plot(y, z_fit)
    dotproduct = np.dot(axis_z2, axis_Z)
    angle_rad = np.arccos(dotproduct)
    alpha_deg = np.rad2deg(angle_rad)
    print('alpha_deg =', alpha_deg)

    #rotate around axis X and Y
    axes_coordinates_rotated1 = np.zeros((3,3))
    for idx, line in enumerate(axes_coordinates):
        coord_rot = rotate(line, phi=0, alpha=alpha_deg, theta=theta_deg)
        print(coord_rot)
        axes_coordinates_rotated1[idx] = coord_rot
        linepts_rot1 = coord_rot * np.mgrid[-7:7:2j][:, np.newaxis]
        ax.plot3D(*linepts_rot1.T, c=f'C{idx}')

    # data_rot = np.zeros_like(data)
    # for idx, vec in enumerate(data):
    #     data_rot[idx] = rotate(vec, phi=0, alpha=alpha_deg, theta=theta_deg)


    #x-y projection of rotated axis (z') - determine phi
    curr = 'CURR_Y'
    df_temp = df[df[curr].notnull()].dropna(axis=1)

    Bx0 = df_temp[df_temp[curr] == 0]['B_labx'].values[0]
    By0 = df_temp[df_temp[curr] == 0]['B_laby'].values[0]
    Bz0 = df_temp[df_temp[curr] == 0]['B_labz'].values[0]
    print(Bx0, By0, Bz0)
    # print(df_x)
    df_temp['delta_Bx'] = df_temp['B_labx'] - Bx0
    df_temp['delta_By'] = df_temp['B_laby'] - By0
    df_temp['delta_Bz'] = df_temp['B_labz'] - Bz0
    # print(df_temp)
    x = df_temp['delta_Bx'].values
    y = df_temp['delta_By'].values
    z = df_temp['delta_Bz'].values

    axis_X = np.array([1, 0])
    m, b = np.polyfit(x, y, 1)
    y_fit = m * x + b
    axis_x1 = np.array([1, m])
    axis_x1 /= np.linalg.norm(axis_x1)
    ax1.scatter(x, y)
    ax1.plot(x, y_fit)
    dotproduct = np.dot(axis_x1, axis_X)
    angle_rad = np.arccos(dotproduct)
    phi_deg = np.rad2deg(angle_rad)
    print('phi_deg =', phi_deg)

    # rotate around axis Z (z')
    axes_coordinates_rotated = np.zeros((3, 3))
    for idx, line in enumerate(axes_coordinates_rotated1):
        coord_rot = rotate(line, phi=phi_deg, alpha=0, theta=0)
        print(coord_rot)
        axes_coordinates_rotated[idx] = coord_rot
        linepts_rot2 = coord_rot * np.mgrid[-7:7:2j][:, np.newaxis]
        ax.plot3D(*linepts_rot2.T)

    filename_parameters = 'rotation_angles.json'
    rotation_parameters = {'alpha': alpha_deg, 'phi': phi_deg, 'theta': theta_deg}
    a_file = open(filename_parameters, "w")
    json.dump(rotation_parameters, a_file)
    a_file.close()




    for idx1, curr in enumerate(curr_list):
        df_temp = df[df[curr].notnull()].dropna(axis=1)
        Bx0 = df_temp[df_temp[curr] == 0]['B_labx'].values[0]
        By0 = df_temp[df_temp[curr] == 0]['B_laby'].values[0]
        Bz0 = df_temp[df_temp[curr] == 0]['B_labz'].values[0]
        print(Bx0, By0, Bz0)
        # print(df_x)
        df_temp['delta_Bx'] = df_temp['B_labx'] - Bx0
        df_temp['delta_By'] = df_temp['B_laby'] - By0
        df_temp['delta_Bz'] = df_temp['B_labz'] - Bz0
        print(df_temp)
        x = df_temp['delta_Bx'].values
        y = df_temp['delta_By'].values
        z = df_temp['delta_Bz'].values
        data = np.concatenate((x[:, np.newaxis],
                               y[:, np.newaxis],
                               z[:, np.newaxis]),
                              axis=1)
        data_rot = np.zeros_like(data)
        for idx, magn in enumerate(data):
            data_rot[idx] = rotate(magn, alpha=alpha_deg, phi=phi_deg, theta=theta_deg)
        print(data_rot)

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data_rot.mean(axis=0)
        print(datamean)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data_rot)

        print(uu)
        print(dd)
        print(vv)


        # Now vv[0] contains the first principal component, i.e. the direction
        # vector of the 'best fit' line in the least squares sense.

        # Now generate some points along this best fit line, for plotting.

        # I use -7, 7 since the spread of the data is roughly 14
        # and we want it to have mean 0 (like the points we did
        # the svd on). Also, it's a straight line, so we only need 2 points.
        linepts_rot = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

        # shift by the mean to get the line in the right place
        linepts_rot += datamean

        # Verify that everything looks right.
        ax.scatter3D(*data_rot.T, c=f'C{idx1}')
        ax.plot3D(*linepts_rot.T, c=f'C{idx1}')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



    #
    # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    #
    # ax = Axes3D(fig)
    # ax.scatter(df_x.delta_Bx, df_x.delta_By, df_x.delta_Bz)


    #
    # plt.show()