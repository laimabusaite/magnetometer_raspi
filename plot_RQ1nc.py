import re

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utilities

if __name__ == '__main__':

    # rot_axis = 'Y'
    rot_axis = '45deg'
    dir_name = 'RQ1nc'
    filename = f'{dir_name}/dataframe_B_{rot_axis}.csv'
    print(filename)
    r_list = [1, 2, 3, 4]

    dataframe_B = pd.read_csv(filename)
    print(dataframe_B)
    print(dataframe_B['Unnamed: 0'])

    fig_name = 'X-axis'
    plt.figure(fig_name)
    for r in r_list:
        plt.scatter(dataframe_B[f'By_measured_r{r} (mT)'], dataframe_B[f'Bz_measured_r{r} (mT)'], c='C0', alpha=0.1, label='')
        plt.scatter(dataframe_B[f'By_coil_r{r} (mT)'], dataframe_B[f'Bz_coil_r{r} (mT)'], c='C1', alpha=0.1, label='')
    plt.scatter(dataframe_B['By_measured_mean (mT)'], dataframe_B['Bz_measured_mean (mT)'], s = 4, c='b', label='measured')
    plt.scatter(dataframe_B['By_coil_mean (mT)'], dataframe_B['Bz_coil_mean (mT)'], s = 4,  c='r', label='coil')
    for i in dataframe_B.index:
        plt.text(dataframe_B.loc[i,'By_coil_mean (mT)'], dataframe_B.loc[i,'Bz_coil_mean (mT)'], f'{i + 1}')
    plt.xlabel('By, mT')
    plt.ylabel('Bz, mT')
    if rot_axis in fig_name:
        plt.axis('equal')
    plt.legend()
    plt.savefig(f'{dir_name}/{rot_axis}/x-axis.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/x-axis.png', bbox_inches='tight')

    fig_name = 'Y-axis'
    plt.figure(fig_name)
    for r in r_list:
        plt.scatter(dataframe_B[f'Bx_measured_r{r} (mT)'], dataframe_B[f'Bz_measured_r{r} (mT)'], c='C0', alpha=0.1,
                    label='')
        plt.scatter(dataframe_B[f'Bx_coil_r{r} (mT)'], dataframe_B[f'Bz_coil_r{r} (mT)'], c='C1', alpha=0.1, label='')
    plt.scatter(dataframe_B['Bx_measured_mean (mT)'], dataframe_B['Bz_measured_mean (mT)'], s=4, c='b', label='measured')
    plt.scatter(dataframe_B['Bx_coil_mean (mT)'], dataframe_B['Bz_coil_mean (mT)'], s=4, c='r', label='coil')
    for i in dataframe_B.index:
        plt.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'], dataframe_B.loc[i, 'Bz_coil_mean (mT)'], f'{i + 1}')
    plt.xlabel('Bx, mT')
    plt.ylabel('Bz, mT')
    if rot_axis in fig_name:
        plt.axis('equal')
    plt.legend()
    plt.savefig(f'{dir_name}/{rot_axis}/y-axis.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/y-axis.png', bbox_inches='tight')

    fig_name = 'Z-axis'
    plt.figure(fig_name)
    for r in r_list:
        plt.scatter(dataframe_B[f'Bx_measured_r{r} (mT)'], dataframe_B[f'By_measured_r{r} (mT)'], c='C0', alpha=0.1,
                    label='')
        plt.scatter(dataframe_B[f'Bx_coil_r{r} (mT)'], dataframe_B[f'By_coil_r{r} (mT)'], c='C1', alpha=0.1, label='')
    plt.scatter(dataframe_B['Bx_measured_mean (mT)'], dataframe_B['By_measured_mean (mT)'], s=4, c='b', label='measured')
    plt.scatter(dataframe_B['Bx_coil_mean (mT)'], dataframe_B['By_coil_mean (mT)'], s=4, c='r', label='coil')
    for i in dataframe_B.index:
        plt.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'], dataframe_B.loc[i, 'By_coil_mean (mT)'], f'{i + 1}')
    plt.xlabel('Bx, mT')
    plt.ylabel('By, mT')
    if rot_axis in fig_name:
        plt.axis('equal')
    plt.legend()
    plt.savefig(f'{dir_name}/{rot_axis}/z-axis.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/z-axis.png', bbox_inches='tight')

    plt.figure('diff_B')
    plt.plot(np.abs(dataframe_B['diff_Bx (mT)']), label=r'$\Delta B_x$')
    plt.plot(np.abs(dataframe_B['diff_By (mT)']), label=r'$\Delta B_y$')
    plt.plot(np.abs(dataframe_B['diff_Bz (mT)']), label=r'$\Delta B_z$')
    plt.plot(np.abs(dataframe_B['diff_B (mT)']), label=r'$\Delta Bmod$')
    plt.xlabel('index')
    plt.ylabel(r'$\Delta B$, mT')
    plt.legend()
    plt.savefig('diffB.pdf', bbox_inches='tight')
    plt.savefig('diffB.png', bbox_inches='tight')

    plt.figure('diff_B rel')
    plt.plot(dataframe_B['Bx_measured_mean (mT)'],  np.abs(dataframe_B['diff_Bx (mT)']), label=r'$\Delta B_x$')
    plt.plot(dataframe_B['By_measured_mean (mT)'], np.abs(dataframe_B['diff_By (mT)']), label=r'$\Delta B_y$')
    plt.plot(dataframe_B['Bz_measured_mean (mT)'], np.abs(dataframe_B['diff_Bz (mT)']), label=r'$\Delta B_z$')
    # plt.plot(dataframe_B['B_measured_mean (mT)'], np.abs(dataframe_B['diff_B (mT)']), label=r'$\Delta B$')
    plt.xlabel('B measured')
    plt.ylabel(r'$\Delta B$, mT')
    plt.legend()
    plt.savefig(f'{dir_name}/{rot_axis}/diffB_vs_B.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/diffB_vs_B.png', bbox_inches='tight')

    plt.figure('B mod')
    plt.plot(dataframe_B['B_measured_mean (mT)'], marker='o', c='C0', label='measured')
    plt.plot(dataframe_B['B_coil_mean (mT)'], marker='o', c='C1', label='coil')
    # plt.plot(dataframe_B['B_measured_mean (mT)'], np.abs(dataframe_B['diff_B (mT)']), label=r'$\Delta B$')
    plt.xlabel('index')
    plt.ylabel('B mod, mT')
    plt.legend()
    plt.savefig(f'{dir_name}/{rot_axis}/Bmod.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/Bmod.png', bbox_inches='tight')

    fig = plt.figure()
    ax = Axes3D(fig)
    try:
        ax.plot_trisurf(dataframe_B['Bx_coil_mean (mT)'],
                 dataframe_B['By_coil_mean (mT)'],
                 dataframe_B['Bz_coil_mean (mT)'], color='C1', alpha=0.1)
    except Exception as e:
        print(e)

    ax.scatter3D(dataframe_B['Bx_measured_mean (mT)'],
                 dataframe_B['By_measured_mean (mT)'],
                 dataframe_B['Bz_measured_mean (mT)'], c='b', s=4)
    for r in r_list:
        ax.scatter3D(dataframe_B[f'Bx_measured_r{r} (mT)'],
                     dataframe_B[f'By_measured_r{r} (mT)'],
                     dataframe_B[f'Bz_measured_r{r} (mT)'], c='C0', alpha=0.1)
    ax.scatter3D(dataframe_B['Bx_coil_mean (mT)'],
                 dataframe_B['By_coil_mean (mT)'],
                 dataframe_B['Bz_coil_mean (mT)'], c='r', s=4)
    for r in r_list:
        ax.scatter3D(dataframe_B[f'Bx_coil_r{r} (mT)'],
                     dataframe_B[f'By_coil_r{r} (mT)'],
                     dataframe_B[f'Bz_coil_r{r} (mT)'], c='C1', alpha=0.1)
    ax.set_xlabel('Bx, mT')
    ax.set_ylabel('By, mT')
    ax.set_zlabel('Bz, mT')
    plt.savefig(f'{dir_name}/{rot_axis}/3D_plot.pdf')
    plt.savefig(f'{dir_name}/{rot_axis}/3D_plot.png')


    plt.show()
