from terminalplot import plot

import glob
import numpy as np
import time
import os

filenames = sorted(glob.glob("RQnc/temp/fan_on_23.5_degrees/fan_on_dev20_avg8.log"))
filenames.sort(key=os.path.getmtime)
#filenames = filenames[-4*10::1]

#filenames = sorted(glob.glob("test_data_rp/test_full_scan_*.dat"))
#filenames.sort(key=os.path.getmtime)
#filenames = filenames[-1::1]
#filename = filenames[0]

#f = filenames[0]

#print(f)

for f in filenames:
    print(f, end= " ")
    x = np.loadtxt(fname = f, delimiter='\t', usecols=(3), unpack = True)
    x = x[0:400:1]
    #y = y[0:10000:1]
    print(len(x))

    #y=y/max(y)
    x = x.tolist()
    #y = y.tolist()

    plot(range(len(x)),x)
    time.sleep(1)
