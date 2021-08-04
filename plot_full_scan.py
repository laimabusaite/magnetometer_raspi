from terminalplot import plot

import glob
import numpy as np
import time
import os

#filenames = sorted(glob.glob("test_data_rp/*.dat"))
#filenames.sort(key=os.path.getmtime)
#filenames = filenames[-3*16::1]

filenames = sorted(glob.glob("test_data_rp/test_full_scan_*.dat"))
filenames.sort(key=os.path.getmtime)
#filenames = filenames[-1::1]
#filename = filenames[0]

#f = filenames[0]

#print(f)

for f in filenames:
    print(f, end= " ")
    x, y = np.loadtxt(fname = f, delimiter='\t', usecols=(0, 1), unpack = True)
    #x = x[0:10000:1]
    #y = y[0:10000:1]
    print(len(x), len(y))

    y=y/max(y)
    x = x.tolist()
    y = y.tolist()

    plot(x, y)
    time.sleep(1)
