# from terminalplot import plot, get_terminal_size
import  terminalplot as tplt
from detect_peaks import *
from bashplotlib.scatterplot import plot_scatter
import time

x = range(100)
y = [i ** 2 for i in x]
tplt.plot(x, y)
print(tplt.get_terminal_size())

filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
dataframe = import_data(filename)
time0 = time.time()
tplt.plot(list(dataframe['MW']), list(dataframe['ODMR']))
time1 = time.time()
print(time1 - time0)
# plot_scatter(filename)