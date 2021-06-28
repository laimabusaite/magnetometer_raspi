import glob
import numpy as np
import matplotlib.pyplot as plt

filenames = sorted(glob.glob('*.dat'))
filenames = filenames[::1]

plt.axes(frameon=True)

i=0
for f in filenames:
    #print(f)
    x, y = np.loadtxt(fname=f, delimiter='\t', usecols=(0,1), unpack=True)
    xmin=10
    imin=-1
    iminpos=0
    for x1 in x[:]:#x[:1]
        imin+=1
        #print(imin,x1)
        if x1<xmin:
            xmin=x1
            iminpos=imin

    xmax=0
    imax=-1
    imaxpos=0
    for x2 in x[:]:#x[-1:]
        imax+=1
        #print(imax,x2)
        if x2>xmax:
            xmax=x2
            imaxpos=imax

    imaxpos=len(x)-(1-imaxpos)
    print(iminpos,"\t",xmin,"\t",imaxpos,"\t",xmax)
    #imin = np.where(x == max(x))
    #imax = np.where(x == min(x))
    #print(imin.item(1))
    #print(min(x),imin,"\n",max(x),imax)
    delta=0
    x=x[iminpos+delta:imaxpos-delta]
    y=y[iminpos+delta:imaxpos-delta]
    y=y/max(y)
    #plt.plot(x,y, color="black", markersize=10.1, markerfacecolor='red', marker='.', linewidth=0.1)
    plt.plot(x, y, markersize=10.4, marker='.', linewidth=1.4, label=f[14:-4])
    #print(x,y)
    i+=1

print("Files found {0:d}".format(i))
fs=24
#plt.legend(loc='upper left', fontsize=fs/2, markerscale=fs/10)
plt.title('')
plt.xlabel('Microwave frequency, MHz',fontdict={'size':fs,})
plt.ylabel('ODMR / Laser',fontdict={'size':fs,})
#plt.xlim(48,54)
#plt.ylim(-100,100)
plt.tick_params(direction='in', length=5, width=1, labelsize=fs)
plt.grid(True)
plt.show()
