import glob
import detect_peaks as dp
import matplotlib.pyplot as plt

if __name__ == '__main__':

    filedir = 'test_data_rp'
    filenames = sorted(glob.glob(f'{filedir}/test_full_scan*.dat'))
    fullscan_filename = filenames[-1]
    print(fullscan_filename)
    #import full scan
    dataframe_full = dp.import_data(fullscan_filename)
    print(dataframe_full.head())

    x_data = dataframe_full['MW']
    y_data = dataframe_full['ODMR']
    plt.plot(x_data, y_data, lw=2, c='k', label='fullscan')

    filedir = 'test_width_contrast_data'
    filenames = sorted(glob.glob(f'{filedir}/*.dat'))
    print(filenames)
    for filename in filenames:
        dataframe = dp.import_data(filename)
        print(dataframe.head())

        x_data = dataframe['MW']
        y_data = dataframe['ODMR']
        plt.plot(x_data, y_data)


    plt.legend()
    plt.show()