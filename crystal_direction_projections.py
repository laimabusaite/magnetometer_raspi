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
    filename = 'crystal_axis_calibration/coil_axis_calibration_old.dat'

    df = pd.read_csv(filename, index_col=0)


    curr = 'CURR_Z'
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

    # x-z projection
    axis_Z = np.array([0, 1])
    m, b = np.polyfit(z, x, 1)
    # z_fit = m * x + b
    x_fit = m * z + b
    print(m, b)
    axis_z1 = np.array([1, 1. / m])
    axis_z1 /= np.linalg.norm(axis_z1)
    plt.scatter(x, z)
    plt.plot(x_fit, z)
    dotproduct = np.dot(axis_z1, axis_Z)
    angle_rad = np.arccos(dotproduct)
    angle_deg = np.rad2deg(angle_rad)
    print('z angle_deg =', angle_deg)

    # y-z projection
    axis_Z = np.array([0, 1])
    m, b = np.polyfit(y, z, 1)
    z_fit = m * y + b
    print(m, b)
    axis_z2 = np.array([1, m])
    axis_z2 /= np.linalg.norm(axis_z2)
    plt.scatter(y, z)
    plt.plot(y, z_fit)
    dotproduct = np.dot(axis_z2, axis_Z)
    angle_rad = np.arccos(dotproduct)
    angle_deg = np.rad2deg(angle_rad)
    print('z angle_deg =', angle_deg)

    curr = 'CURR_X'
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



    plt.show()