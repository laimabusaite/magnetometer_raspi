# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

from fit_odmr import *
# import glob
# import os
from mpl_toolkits.mplot3d import Axes3D
from utilities import rotate
# from generate_data_rotation import rotate_about_axis

if __name__ == '__main__':

    # filename = 'crystal_axis_calibration/coil_axis_calibration_old.dat'
    filename = '../not_used_files/new_calibration/coil_axis_calibration.dat'

    df = pd.read_csv(filename, index_col=0)

    # print(df)
    curr_list = ['CURR_X', 'CURR_Y', 'CURR_Z']

    axes_coordinates = np.zeros((3,3))
    # axes_coordinates_rotated = np.zeros((3, 3))

    axes_box = np.eye(3)
    angles_list = np.zeros(3)

    colors1 = ['r', 'g', 'b']
    colors2 = ['tab:red', 'tab:green', 'tab:blue']
    colors3 = ['magenta', 'yellow', 'cyan']
    # colors3 = colors2

    fig = plt.figure()
    ax = Axes3D(fig)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    for idx, curr in enumerate(curr_list):

        # df_temp = df[df[curr].notnull()].dropna(axis=1)
        # df_temp = df[df[curr] != 0].dropna(axis=1)
        axis_name = curr[-1]
        df_temp = df[df['axis'] == axis_name]

        Bx0 = df_temp[df_temp[curr] == 0]['B_labx'].values[0]
        By0 = df_temp[df_temp[curr] == 0]['B_laby'].values[0]
        Bz0 = df_temp[df_temp[curr] == 0]['B_labz'].values[0]
        print(Bx0, By0, Bz0)
        Bxm0 = df_temp[df_temp[curr] == -0.0]['B_labx'].values[0]
        Bym0 = df_temp[df_temp[curr] == -0.0]['B_laby'].values[0]
        Bzm0 = df_temp[df_temp[curr] == -0.0]['B_labz'].values[0]
        print(Bxm0, Bym0, Bzm0)
        index0 = df_temp[df_temp[curr] == 0.0].index[0]
        print(index0)

        # print(df_x)
        df_temp['delta_Bx'] = (df_temp.loc[:,'B_labx'] - Bx0).values
        df_temp['delta_By'] = (df_temp.loc[:,'B_laby'] - By0).values
        df_temp['delta_Bz'] = (df_temp.loc[:,'B_labz'] - Bz0).values
        print('df_temp', idx, curr)
        df_temp.sort_values(by=curr, inplace=True)
        # print(df_temp)
        x = df_temp.loc[:,'delta_Bx'].values
        y = df_temp.loc[:,'delta_By'].values
        z = df_temp.loc[:,'delta_Bz'].values

        data = np.concatenate((x[:, np.newaxis],
                               y[:, np.newaxis],
                               z[:, np.newaxis]),
                              axis=1)


        # print(data)

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data.mean(axis=0)
        print(datamean)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data)

        # print(uu)
        # print(dd)
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
        ax.scatter3D(*data.T, c=colors1[idx])
        ax.plot3D(*linepts.T, c=colors1[idx])

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


    # x-z projection - determine theta
    # print(x, z)
    axis_Z = np.array([0, 1])
    # m, b = np.polyfit(z, x, 1)
    m, b = np.polyfit(x, z, 1)
    z_fit = m * x + b
    # x_fit = m * z + b
    print(m, b)
    # axis_z1 = np.array([1, 1. / m])
    axis_z1 = np.array([1, m])
    axis_z1 /= np.linalg.norm(axis_z1)
    ax1.scatter(x, z)
    # ax1.plot(x_fit, z)
    ax1.plot(x, z_fit)
    ax1.set_xlabel('Bx fit')
    ax1.set_ylabel('Bz fit')
    dotproduct = np.dot(axis_z1, axis_Z)
    angle_rad = np.arccos(dotproduct)
    theta_deg = np.rad2deg(angle_rad)
    print('theta_deg =', theta_deg)

    # rotate around axis Y
    axes_coordinates_rotated1 = np.zeros((3, 3))
    for idx, line in enumerate(axes_coordinates):
        coord_rot = rotate(line, phi=0, alpha=0, theta=theta_deg)
        print(coord_rot)
        axes_coordinates_rotated1[idx] = coord_rot
        linepts_rot1 = coord_rot * np.mgrid[-7:7:2j][:, np.newaxis]
        ax.plot3D(*linepts_rot1.T, c='k', ls='dashed')

    #rotate data
    data_rot = np.zeros_like(data)
    for idx2, magn in enumerate(data):
        data_rot[idx2] = rotate(magn, alpha=0, phi=0, theta=theta_deg)
    # print(data_rot)

    # y-z projection - determine alpha
    x = data_rot[:, 0]
    y = data_rot[:, 1]
    z = data_rot[:, 2]
    axis_Z = np.array([0, 1])
    m, b = np.polyfit(y, z, 1)
    z_fit = m * y + b
    print(m, b)
    axis_z2 = np.array([1, m])
    axis_z2 /= np.linalg.norm(axis_z2)
    ax2.scatter(y, z)
    ax2.plot(y, z_fit)
    ax2.set_xlabel('By fit')
    ax2.set_ylabel('Bz fit')
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
        ax.plot3D(*linepts_rot1.T, c='k')

    # data_rot = np.zeros_like(data)
    # for idx, vec in enumerate(data):
    #     data_rot[idx] = rotate(vec, phi=0, alpha=alpha_deg, theta=theta_deg)


    #x-y projection of rotated axis (z') - determine phi
    curr = 'CURR_Y'
    # curr = 'CURR_X'
    axis_name = curr[-1]
    df_temp = df[df['axis'] == axis_name].copy()
    df_temp.sort_values(by=curr, inplace=True)
    Bx0 = df_temp[df_temp[curr] == 0]['B_labx'].values[0]
    By0 = df_temp[df_temp[curr] == 0]['B_laby'].values[0]
    Bz0 = df_temp[df_temp[curr] == 0]['B_labz'].values[0]
    print(Bx0, By0, Bz0)
    # print(df_x)
    df_temp.loc[:,'delta_Bx'] = df_temp.loc[:,'B_labx'] - Bx0
    df_temp.loc[:,'delta_By'] = df_temp.loc[:,'B_laby'] - By0
    df_temp.loc[:,'delta_Bz'] = df_temp.loc[:,'B_labz'] - Bz0
    # print(df_temp)
    x = df_temp.loc[:, 'delta_Bx'].values
    y = df_temp.loc[:, 'delta_By'].values
    z = df_temp.loc[:, 'delta_Bz'].values

    data = np.concatenate((x[:, np.newaxis],
                           y[:, np.newaxis],
                           z[:, np.newaxis]),
                          axis=1)

    # rotate data
    data_rot = np.zeros_like(data)
    for idx2, magn in enumerate(data):
        data_rot[idx2] = rotate(magn, alpha=0, phi=0, theta=theta_deg)
    # print(data_rot)
    x = data_rot[:, 0]
    y = data_rot[:, 1]
    z = data_rot[:, 2]

    axis_X = np.array([1, 0])
    m, b = np.polyfit(x, y, 1)
    y_fit = m * x + b
    axis_x1 = np.array([1, m])
    axis_x1 /= np.linalg.norm(axis_x1)
    ax3.scatter(x, y)
    ax3.plot(x, y_fit)
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
        ax.plot3D(*linepts_rot2.T, c=colors1[idx], ls='dashed')

    filename_parameters = '../not_used_files/new_calibration/rotation_angles.json'
    rotation_parameters = {'alpha': alpha_deg, 'phi': phi_deg, 'theta': theta_deg}
    a_file = open(filename_parameters, "w")
    json.dump(rotation_parameters, a_file)
    a_file.close()




    for idx1, curr in enumerate(curr_list):
        df_temp = df[df['axis'] == axis_name]
        Bx0 = df_temp[df_temp[curr] == 0]['B_labx'].values[0]
        By0 = df_temp[df_temp[curr] == 0]['B_laby'].values[0]
        Bz0 = df_temp[df_temp[curr] == 0]['B_labz'].values[0]
        print(Bx0, By0, Bz0)
        # print(df_x)
        df_temp.loc[:,'delta_Bx'] = df_temp.loc[:,'B_labx'] - Bx0
        df_temp.loc[:,'delta_By'] = df_temp.loc[:,'B_laby'] - By0
        df_temp.loc[:,'delta_Bz'] = df_temp.loc[:,'B_labz'] - Bz0
        # print(df_temp)
        x = df_temp['delta_Bx'].values
        y = df_temp['delta_By'].values
        z = df_temp['delta_Bz'].values
        data = np.concatenate((x[:, np.newaxis],
                               y[:, np.newaxis],
                               z[:, np.newaxis]),
                              axis=1)
        data_rot = np.zeros_like(data)
        for idx2, magn in enumerate(data):
            data_rot[idx2] = rotate(magn, alpha=alpha_deg, phi=phi_deg, theta=theta_deg)
        # print(data_rot)

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data_rot.mean(axis=0)
        print(datamean)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data_rot)

        # print(uu)
        # print(dd)
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
        print(idx1, colors3[idx1])
        ax.scatter3D(*data_rot.T, c=colors3[idx1])
        ax.plot3D(*linepts_rot.T, c=colors3[idx1])


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