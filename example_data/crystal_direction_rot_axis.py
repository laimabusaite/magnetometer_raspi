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
    axes_coordinates2 = np.zeros((3, 3))
    axes_coordinates3 = np.zeros((3, 3))
    # axes_coordinates_rotated = np.zeros((3, 3))

    axes_box = np.eye(3)
    angles_list = np.zeros(3)

    colors1 = ['r', 'g', 'b']
    colors2 = ['tab:red', 'tab:green', 'tab:blue']
    colors3 = ['magenta', 'yellow', 'cyan']
    # colors3 = colors2

    fig = plt.figure()
    ax = Axes3D(fig)

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

        axis_crystal = vv[0] / np.linalg.norm(vv[0])
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


    # rotate z axis around new rotation axis
    axis_Z = np.array([0, 0, 1])
    axis_z = axes_coordinates[2]

    dotproduct = np.dot(axis_z, axis_Z)
    print(dotproduct)
    angle_rad = np.arccos(dotproduct)
    print(angle_rad)
    angle_deg = np.rad2deg(angle_rad)
    print(angle_deg)

    cross = np.cross(axis_z, axis_Z)
    print(cross)
    rot_axis = cross / np.linalg.norm(cross)
    print(rot_axis)
    linepts_rot_axis = rot_axis * np.mgrid[-7:7:2j][:, np.newaxis]
    ax.plot3D(*linepts_rot_axis.T, c='k')

    data_rot_list = []
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

        # rotate data
        data_rot = np.zeros_like(data)
        for idx2, magn in enumerate(data):
            data_rot[idx2] = rotate_about_axis(magn, rot_axis, angle_rad)

        data_rot_list.append(data_rot)# Calculate the mean of the points, i.e. the 'center' of the cloud

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
        linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

        # shift by the mean to get the line in the right place
        linepts += datamean
        axis_crystal = vv[0] / np.linalg.norm(vv[0])
        axes_coordinates2[idx] = axis_crystal

        # Verify that everything looks right.
        ax.scatter3D(*data_rot.T, c=colors2[idx])
        ax.plot3D(*linepts.T, c=colors2[idx])

    print('axes_coordinates2')
    print(axes_coordinates2)

    #rotate x axis around z-axis
    axis_X = np.array([1, 0, 0])
    axis_x = axes_coordinates2[0]

    dotproduct = np.dot(axis_x, axis_X)
    print(dotproduct)
    phi_rad = np.arccos(dotproduct)
    print('phi_rad', phi_rad)
    phi_deg = np.rad2deg(phi_rad)
    print('phi_deg', phi_deg)

    cross = np.cross(axis_x, axis_X)
    print('cross', cross)
    rot_axis_z = cross / np.linalg.norm(cross)
    print('rot_axis_z', rot_axis_z)
    linepts_rot_axis_z = rot_axis_z * np.mgrid[-7:7:2j][:, np.newaxis]
    ax.plot3D(*linepts_rot_axis_z.T, c='k', ls='dashed')

    data_rot_list2 = []
    for idx, curr in enumerate(curr_list):

        data = data_rot_list[idx]

        # rotate data
        data_rot = np.zeros_like(data)
        for idx2, magn in enumerate(data):
            data_rot[idx2] = rotate_about_axis(magn, axis_Z, phi_rad)

        data_rot_list2.append(data_rot)  # Calculate the mean of the points, i.e. the 'center' of the cloud

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
        linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

        # shift by the mean to get the line in the right place
        linepts += datamean
        axis_crystal = vv[0] / np.linalg.norm(vv[0])
        axes_coordinates3[idx] = axis_crystal

        # Verify that everything looks right.
        ax.scatter3D(*data_rot.T, c=colors3[idx])
        ax.plot3D(*linepts.T, c=colors3[idx])

    print('axes_coordinates3')
    print(axes_coordinates3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    filename_parameters = '../not_used_files/new_calibration/rotation_angles_axis.json'
    rotation_parameters = {'axis_rot1_0': rot_axis[0],
                           'axis_rot1_1': rot_axis[1],
                           'axis_rot1_2': rot_axis[2],
                           'angle': angle_rad,
                           'phi': phi_rad}
    a_file = open(filename_parameters, "w")
    json.dump(rotation_parameters, a_file)
    a_file.close()

    plt.show()



    #
    # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    #
    # ax = Axes3D(fig)
    # ax.scatter(df_x.delta_Bx, df_x.delta_By, df_x.delta_Bz)


    #
    # plt.show()