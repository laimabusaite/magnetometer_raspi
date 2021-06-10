from terminalplot import plot

import glob
import numpy as np
import time
import os

filenames = sorted(glob.glob("full_scan_rp_1.dat"))
filenames.sort(key=os.path.getmtime)
filenames = filenames[-4::1]

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
