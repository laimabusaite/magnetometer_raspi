import re

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Patch

import utilities


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='None', alpha=0.5, ms=4, markercolor='k', label=None):
    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.scatter(xdata, ydata, s=ms, c=markercolor)
    # ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
    #                       fmt='None', ecolor='k')

    handle = Rectangle((0, 0), 1, 1, facecolor=facecolor, alpha=alpha,
                       edgecolor=edgecolor)

    return (artists, handle), label


if __name__ == '__main__':

    # rot_axis = 'Z'
    rot_axis = '45deg'
    dir_name = 'RQ1nc'
    filename = f'{dir_name}/dataframe_B_{rot_axis}.csv'
    print(filename)
    r_list = [1, 2, 3, 4]

    dataframe_B = pd.read_csv(filename)
    print(dataframe_B)
    print(dataframe_B['Unnamed: 0'])

    fig_name = 'X-axis'
    # plt.figure(fig_name)
    fig, ax = plt.subplots(1)
    plt.title(fig_name)

    xerr = np.sqrt(dataframe_B['By_measured_std_mean (mT)'] ** 2 + dataframe_B['By_measured_std_0 (mT)'] ** 2)
    yerr = np.sqrt(dataframe_B['Bz_measured_std_mean (mT)'] ** 2 + dataframe_B['Bz_measured_std_0 (mT)'] ** 2)
    handle_coil, label_coil = make_error_boxes(ax, dataframe_B['By_coil_mean (mT)'].values,
                                               dataframe_B['Bz_coil_mean (mT)'].values,
                                               xerror=np.array([dataframe_B['By_coil_std_mean (mT)'],
                                                                dataframe_B['By_coil_std_mean (mT)']]),
                                               yerror=np.array([dataframe_B['Bz_coil_std_mean (mT)'],
                                                                dataframe_B['Bz_coil_std_mean (mT)']]),
                                               facecolor='C1', edgecolor='None', alpha=0.2, ms=4, markercolor='r',
                                               label='coil'
                                               )
    handle_measured, label_measured = make_error_boxes(ax, dataframe_B['By_measured_mean (mT)'].values,
                                                       dataframe_B['Bz_measured_mean (mT)'].values,
                                                       xerror=np.array([xerr,
                                                                        xerr]),
                                                       yerror=np.array([yerr,
                                                                        yerr]),
                                                       facecolor='C0', edgecolor='None', alpha=0.2, ms=4,
                                                       markercolor='b',
                                                       label='measured'
                                                       )
    # print(pc)
    for i in dataframe_B.index:
        if i == 0:
            ax.text(dataframe_B.loc[i, 'By_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'Bz_coil_mean (mT)'] + 0.02,
                    f'{i + 1}')
        elif i == 16:
            ax.text(dataframe_B.loc[i, 'By_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'Bz_coil_mean (mT)'] - 0.02,
                    f'{i + 1}')
        else:
            ax.text(dataframe_B.loc[i, 'By_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'Bz_coil_mean (mT)'], f'{i + 1}')
    plt.xlabel('By, mT')
    plt.ylabel('Bz, mT')
    if rot_axis in fig_name:
        plt.axis('equal')
    plt.legend(handles=[handle_coil, handle_measured], labels=['coil', 'measured'])
    plt.savefig(f'{dir_name}/{rot_axis}/x-axis.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/x-axis.png', bbox_inches='tight')

    fig_name = 'Y-axis'
    # plt.figure(fig_name)
    fig, ax = plt.subplots(1)
    plt.title(fig_name)
    xerr = np.sqrt(dataframe_B['Bx_measured_std_mean (mT)'] ** 2 + dataframe_B['Bx_measured_std_0 (mT)'] ** 2)
    yerr = np.sqrt(dataframe_B['Bz_measured_std_mean (mT)'] ** 2 + dataframe_B['Bz_measured_std_0 (mT)'] ** 2)
    handle_coil, label_coil = make_error_boxes(ax, dataframe_B['Bx_coil_mean (mT)'].values,
                                               dataframe_B['Bz_coil_mean (mT)'].values,
                                               xerror=np.array([dataframe_B['Bx_coil_std_mean (mT)'].values,
                                                                dataframe_B['Bx_coil_std_mean (mT)'].values]),
                                               yerror=np.array([dataframe_B['Bz_coil_std_mean (mT)'].values,
                                                                dataframe_B['Bz_coil_std_mean (mT)'].values]),
                                               facecolor='C1', edgecolor='None', alpha=0.2, ms=4, markercolor='r',
                                               label='coil'
                                               )
    handle_measured, label_measured = make_error_boxes(ax, dataframe_B['Bx_measured_mean (mT)'].values,
                                                       dataframe_B['Bz_measured_mean (mT)'].values,
                                                       xerror=np.array([dataframe_B['Bx_measured_std_mean (mT)'].values,
                                                                        dataframe_B[
                                                                            'Bx_measured_std_mean (mT)'].values]),
                                                       yerror=np.array([dataframe_B['Bz_measured_std_mean (mT)'].values,
                                                                        dataframe_B[
                                                                            'Bz_measured_std_mean (mT)'].values]),
                                                       facecolor='C0', edgecolor='None', alpha=0.2, ms=4,
                                                       markercolor='b',
                                                       label='measured'
                                                       )
    for i in dataframe_B.index:
        if i == 0:
            ax.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'Bz_coil_mean (mT)'] + 0.02,
                    f'{i + 1}')
        elif i == 16:
            ax.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'Bz_coil_mean (mT)'] - 0.02,
                    f'{i + 1}')
        else:
            ax.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'Bz_coil_mean (mT)'], f'{i + 1}')
    plt.xlabel('Bx, mT')
    plt.ylabel('Bz, mT')
    if rot_axis in fig_name:
        plt.axis('equal')
    plt.legend(handles=[handle_coil, handle_measured], labels=['coil', 'measured'])
    plt.savefig(f'{dir_name}/{rot_axis}/y-axis.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/y-axis.png', bbox_inches='tight')

    fig_name = 'Z-axis'
    # plt.figure(fig_name)
    fig, ax = plt.subplots(1)
    plt.title(fig_name)
    xerr = np.sqrt(dataframe_B['Bx_measured_std_mean (mT)'] ** 2 + dataframe_B['Bx_measured_std_0 (mT)'] ** 2)
    yerr = np.sqrt(dataframe_B['By_measured_std_mean (mT)'] ** 2 + dataframe_B['By_measured_std_0 (mT)'] ** 2)
    handle_coil, label_coil = make_error_boxes(ax, dataframe_B['Bx_coil_mean (mT)'].values,
                                               dataframe_B['By_coil_mean (mT)'].values,
                                               xerror=np.array([dataframe_B['Bx_coil_std_mean (mT)'].values,
                                                                dataframe_B['Bx_coil_std_mean (mT)'].values]),
                                               yerror=np.array([dataframe_B['By_coil_std_mean (mT)'].values,
                                                                dataframe_B['By_coil_std_mean (mT)'].values]),
                                               facecolor='C1', edgecolor='None', alpha=0.2, ms=4, markercolor='r',
                                               label='coil'
                                               )
    handle_measured, label_measured = make_error_boxes(ax, dataframe_B['Bx_measured_mean (mT)'].values,
                                                       dataframe_B['By_measured_mean (mT)'].values,
                                                       xerror=np.array([xerr,
                                                                        xerr]),
                                                       yerror=np.array([yerr,
                                                                        yerr]),
                                                       facecolor='C0', edgecolor='None', alpha=0.2, ms=4,
                                                       markercolor='b',
                                                       label='measured'
                                                       )
    for i in dataframe_B.index:
        if i == 0:
            ax.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'By_coil_mean (mT)'] + 0.02,
                    f'{i + 1}')
        elif i == 16:
            ax.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'By_coil_mean (mT)'] - 0.02,
                    f'{i + 1}')
        else:
            ax.text(dataframe_B.loc[i, 'Bx_coil_mean (mT)'] + 0.02, dataframe_B.loc[i, 'By_coil_mean (mT)'], f'{i + 1}')
    plt.xlabel('Bx, mT')
    plt.ylabel('By, mT')
    if rot_axis in fig_name:
        plt.axis('equal')
    plt.legend(handles=[handle_coil, handle_measured], labels=['coil', 'measured'])
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
    plt.plot(dataframe_B['Bx_measured_mean (mT)'], np.abs(dataframe_B['diff_Bx (mT)']), label=r'$\Delta B_x$')
    plt.plot(dataframe_B['By_measured_mean (mT)'], np.abs(dataframe_B['diff_By (mT)']), label=r'$\Delta B_y$')
    plt.plot(dataframe_B['Bz_measured_mean (mT)'], np.abs(dataframe_B['diff_Bz (mT)']), label=r'$\Delta B_z$')
    # plt.plot(dataframe_B['B_measured_mean (mT)'], np.abs(dataframe_B['diff_B (mT)']), label=r'$\Delta B$')
    plt.xlabel('B measured')
    plt.ylabel(r'$\Delta B$, mT')
    plt.legend()
    plt.savefig(f'{dir_name}/{rot_axis}/diffB_vs_B.pdf', bbox_inches='tight')
    plt.savefig(f'{dir_name}/{rot_axis}/diffB_vs_B.png', bbox_inches='tight')

    print(dataframe_B['B_measured_mean (mT)'] - dataframe_B['B_measured_std_mean (mT)'])
    err_mod = np.sqrt(dataframe_B['B_measured_std_mean (mT)'] ** 2 + dataframe_B['B_measured_std_0 (mT)'] ** 2)
    plt.figure('B mod')
    l1, = plt.plot(dataframe_B.index + 1, dataframe_B['B_measured_mean (mT)'], marker='o', c='C0', label='measured')
    fill1 = plt.fill_between(dataframe_B.index + 1,
                             dataframe_B['B_measured_mean (mT)'] - err_mod,
                             dataframe_B['B_measured_mean (mT)'] + err_mod, alpha=0.2)
    l2, = plt.plot(dataframe_B.index + 1, dataframe_B['B_coil_mean (mT)'], marker='o', c='C1', label='coil')
    fill2 = plt.fill_between(dataframe_B.index + 1,
                             dataframe_B['B_coil_mean (mT)'] - dataframe_B['B_coil_std_mean (mT)'],
                             dataframe_B['B_coil_mean (mT)'] + dataframe_B['B_coil_std_mean (mT)'], alpha=0.2)
    # plt.plot(dataframe_B['B_measured_mean (mT)'], np.abs(dataframe_B['diff_B (mT)']), label=r'$\Delta B$')
    plt.xlim(min(dataframe_B.index + 1), max(dataframe_B.index + 1))
    plt.xlabel('index')
    plt.ylabel('B mod, mT')
    plt.legend(handles=[(l1, fill1), (l2, fill2)], labels=['measured', 'coil'])
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
                     dataframe_B[f'Bz_measured_r{r} (mT)'], c='C0', alpha=0.1, edgecolors=None)
    ax.scatter3D(dataframe_B['Bx_coil_mean (mT)'],
                 dataframe_B['By_coil_mean (mT)'],
                 dataframe_B['Bz_coil_mean (mT)'], c='r', s=4)
    for r in r_list:
        ax.scatter3D(dataframe_B[f'Bx_coil_r{r} (mT)'],
                     dataframe_B[f'By_coil_r{r} (mT)'],
                     dataframe_B[f'Bz_coil_r{r} (mT)'], c='C1', alpha=0.1, edgecolors=None)
    ax.set_xlabel('Bx, mT')
    ax.set_ylabel('By, mT')
    ax.set_zlabel('Bz, mT')
    plt.savefig(f'{dir_name}/{rot_axis}/3D_plot.pdf')
    plt.savefig(f'{dir_name}/{rot_axis}/3D_plot.png')

    plt.show()
