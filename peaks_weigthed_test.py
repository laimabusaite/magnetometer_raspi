import numpy as np

from detect_peaks import *
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':


    # filename = 'test_data/test_dev20.0_peak2611.0.dat'

    # filenames = sorted(glob.glob("test_data/*.dat"))
    filenames = sorted(glob.glob("data/*.dat"))
    print(filenames)
    idx = 0
    order_min_list = []
    for filename in filenames[2:]:
        print(filename)
        dataframe = import_data(filename)
        print(dataframe.head())


        plt.figure(1)
        peaks, amplitudes = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=True)
        print(peaks)

        weight_max = 2.0
        weight_order = 2
        peak_weighted_2 = detect_peaks_weighted(dataframe['MW'], dataframe['ODMR'], weight_max = weight_max, weight_order = weight_order)
        print(peak_weighted_2)
        plt.axvline(x=peak_weighted_2, color = 'r', lw=2)

        # weight_max_list = np.arange(1.0,10.0, 0.1)
        # for weight_max in weight_max_list:
        #     peak_weighted = detect_peaks_weighted(dataframe['MW'], dataframe['ODMR'], weight_max=weight_max,
        #                                           weight_order=weight_order)
        #     print(weight_max, peak_weighted)

        weight_order_list = np.arange(1, 10, 0.001)
        peak_weighted_list = []
        for weight_order in weight_order_list:
            peak_weighted = detect_peaks_weighted(dataframe['MW'], dataframe['ODMR'], weight_max=weight_max,
                                                  weight_order=weight_order)
            # print(weight_order, peak_weighted - peaks)
            peak_weighted_list.append(peak_weighted)

        peak_weighted_list = np.array(peak_weighted_list)
        peak_diff_list = peak_weighted_list - peaks
        idx_min_diff = np.argmin(np.abs(peak_diff_list))
        order_min_list.append(weight_order_list[idx_min_diff])
        print(idx_min_diff, weight_order_list[idx_min_diff], peak_weighted_list[idx_min_diff], peak_diff_list[idx_min_diff])
        plt.figure(2)
        plt.axhline(y=0, color='gray')
        p = plt.plot(weight_order_list, peak_diff_list)
        col = p[0].get_color()
        plt.axvline(x=weight_order_list[idx_min_diff], color=col, lw=2,
                    label=f'order={weight_order_list[idx_min_diff]:.2f}')
        plt.legend()


        plt.figure(1)
        plt.axvline(x=peak_weighted_list[idx_min_diff], color=col, lw=2, label=f'order={weight_order_list[idx_min_diff]:.2f}')
        plt.legend()

        idx += 1

    mean_order = np.mean(order_min_list)
    print(f'mean_order = {mean_order:.2f}')

    plt.figure(3)
    bins = np.arange(1, 8, 0.5)
    plt.hist(order_min_list, bins=bins)

    plt.show()
