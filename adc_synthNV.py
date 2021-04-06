import serial
import time
import os
import math

import numpy as np

from scipy.signal import find_peaks

import spidev
import signal
import sys
import RPi.GPIO as GPIO

import NVcenter as nv
from detect_peaks import *
from utilities import *
import glob
import json

os.system("clear")

# read parameters
filename = "ODMR_fit_parameters.json"
a_file = open(filename, "r")
parameters = json.load(a_file)
# parameters = dict(parameters)
# print(parameters)

D = parameters["D"] #2867.61273
Mz_array = np.array([parameters["Mz1"], parameters["Mz2"], parameters["Mz3"], parameters["Mz4"]]) #np.array([1.85907453, 2.16259814, 1.66604227, 2.04334145]) #np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])
B_lab = np.array([parameters["B_labx"], parameters["B_laby"], parameters["B_labz"]]) #np.array([169.121512, 87.7180839, 40.3986877]) #np.array([191.945068, 100.386360, 45.6577322])

print('D =', D)
print('Mz_array =', Mz_array)
print('B_lab =', B_lab)

# NV center orientation in laboratory frame
# (100)
nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
nv_center_set.setMagnetic(B_lab=B_lab)
# print(nv_center_set.B_lab)

frequencies0 = nv_center_set.four_frequencies(np.array([2000, 3500]), nv_center_set.B_lab)
# print(frequencies0)

A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)
# print(A_inv)

spi = spidev.SpiDev()
spi.open(1,2)

GPIO.setmode(GPIO.BCM)


def signal_handler(sig, frame):
    #GPIO.cleanup()
    print("\nexit")
    spi.close()
    sys.exit(0)


def read_values(channel):
    spi.max_speed_hz = 115200
    adc = spi.xfer2([1,(8+channel)<<4,0])
    data = ((adc[1]&3) << 8) + adc[2]
    return data


signal.signal(signal.SIGINT, signal_handler)

sleeptime=0.4

ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyACM0'
ser.timeout = 4
print("\nSERIAL PORT")
print(ser)
time.sleep(sleeptime)
print("Port\t\t\t",ser.name)
time.sleep(sleeptime)
print("Open port\t\t",ser.open())
time.sleep(sleeptime)
print("Port is open\t\t",ser.is_open)

amp=15.0

#print()
#print("Model\t", end="")
#print(ser.write(b"+"))
#temp_error=ser.readline()
#temp_error=temp_error.decode('UTF-8')[1:]
#print("{0:s}\t".format(temp_error))

# Set channel 0 power and frequency
#synth[0].power = amp

send_amp="W{0:.1f}".format(amp)
print("\n{0:s}\t".format(send_amp), end="")
print(ser.write(send_amp.encode(encoding='UTF-8',errors='strict')))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))

print("RF ON \t", end="")
print(ser.write(b"h1"))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))

print("RF ON \t", end="")
print(ser.write(b"E1"))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))


def write_file(x,y,s):
    filename1="test_data/test_{0:s}.dat".format(s)

    #print(filename1)

    f1 = open(filename1, 'w+')
    #print("FILE OPEN",f1)
    for i2 in range(len(x)):
        f1.write("%10.9f\t%10.9f\n" % (x[i2], y[i2]))
    f1.close()

    return


def scan_peak(f0, dev, step_size, avg1, avg2, level, noise):
    #print("Peak scan, {0:.1f}, {1:.1f}".format(f0, dev))
    t0=time.time()
    points=int((2*dev)/step_size)

    averages1=avg1
    averages2=avg2

    frequency_chan=[(f0-dev+i1*2*dev/points) for i1 in range(int(points))]
    average_chan=[[0 for i1 in range(int(points))] for j1 in range(averages1)]
    for k in range(averages1):
        #print("k={0:d} ".format(k), end="", flush=True)
        for i in range(int(points)):
            vmean=[0 for i1 in range(averages2)]
            #print(" i={0:d}".format(i), end="", flush=True)
            f=(f0-dev+i*step_size)

            send_frequency="f{0:.4f}".format(f)
            #print("{0:s}\t".format(send_frequency), end="")
            send_freq=ser.write(send_frequency.encode(encoding='UTF-8',errors='strict'))
            #print(send_freq)
            #temp_error=ser.readline()
            #temp_error=temp_error.decode('UTF-8')[1:]
            #print("{0:s}\t".format(temp_error))

            time.sleep(350/1000000)
            vmean_sum1=0
            #print(" j=", end="", flush=True)
            for j in range(averages2):
                while True:
                    #print("{0:d}".format(j), end="", flush=True)
                    odmr = (read_values(0))*4096/1000
                    laser = (read_values(1))*4096/1000
                    #print(odmr, laser)
                    vmean[j]=odmr/laser
                    #print("{0:d}".format(j+1), end=" ", flush=True)
                    if (vmean[j]<level*(100+noise)/100) and (vmean[j]>level*(100-noise)/100):
                        vmean_sum1+=vmean[j]
                        #print(" ok ", end="", flush=True)
                        break
                    else:
                        print("no ok")


            #print("{0:f}".format(vmean_sum/averages))
            #chan[1][i]=f
            average_chan[k][i]=vmean_sum1/averages2

        #print("")

    average_chan1=[0 for i1 in range(int(points))]
    for i in range(int(points)):
        vmean_sum2=0
        for k in range(averages1):
            vmean_sum2+=average_chan[k][i]
            average_chan1[i]=vmean_sum2/averages1

    write_file(frequency_chan,average_chan1,"dev{0:.1f}_peak{1:.1f}".format(dev,f0))
    #peaks_chan=[0 for i1 in range(int(points))]
    #for i in range(int(points)):
    #    peaks_chan[i]=(-1)*average_chan1[i]
    #peaks_chan=np.array(peaks_chan)
    #peaks, _ = find_peaks(peaks_chan, distance=20)
    #plt.plot(peaks_chan)
    #plt.plot(peaks, peaks_chan[peaks], "x")
    #plt.show(block=False)
    #time.sleep(1)
    #plt.close()
    #frequency_chan=np.array(frequency_chan)
    #print("{0:.1f} s".format(time.time()-t0))
    #return frequency_chan[peaks]


def get_baseline(f0,points):
    #print("Peak scan, {0:.1f}, {1:.1f}".format(f0, dev))
    #t0=time.time()

    send_frequency="f{0:.4f}".format(f0)
    #print("{0:s}\t".format(send_frequency), end="")
    send_freq=ser.write(send_frequency.encode(encoding='UTF-8',errors='strict'))
    #print(send_freq)
    #temp_error=ser.readline()
    #temp_error=temp_error.decode('UTF-8')[1:]
    #print("{0:s}\t".format(temp_error))

    time.sleep(350/1000000)
    vmean_sum1=0
    vmean=[0 for i1 in range(points)]
    for i in range(points):
        #print("{0:d}".format(j), end="", flush=True)
        odmr = (read_values(0))*4096/1000
        laser = (read_values(1))*4096/1000
        #print(odmr, laser)
        vmean[i]=odmr/laser
        #print("{0:d}".format(j+1), end=" ", flush=True)
        vmean_sum1+=vmean[i]

    return vmean_sum1/points


dev0=20
step=0.4
a1=4
a2=1
noise=2
t0=time.time()

for i in range(1):
    #print(i)
    #scan_peak(2417,dev0,step,a1,a2)
    level = get_baseline(2611-10,10)
    scan_peak(2611,dev0,step,a1,a2,level,noise)
    #scan_peak(2805,dev0,step,a1,a2)
    level = get_baseline(2938-10,10)
    scan_peak(2938,dev0,step,a1,a2,level,noise)
    #scan_peak(3113,dev0,step,a1,a2)
    level = get_baseline(3204-10,10)
    scan_peak(3204,dev0,step,a1,a2,level,noise)
    #scan_peak(3323,dev0,step,a1,a2)
    level = get_baseline(3389-10,10)
    scan_peak(3389,dev0,step,a1,a2,level,noise)
    level = get_baseline(2905,10)
    scan_peak(2905,600,0.2,32,4,level,10)

    #os.system("python3 plot1.py")
    filenames = sorted(glob.glob("test_data/*.dat"))
    peaks_list = []
    #print()
    for filename in filenames:
        #print(filename)
        dataframe = import_data(filename)
        # print(dataframe)
        peaks, amplitudes = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=False)
        #print(peaks)
        peaks_list.append(peaks)

    peaks_list = np.array(peaks_list).flatten()
    # print(peaks_list)

    delta_frequencies = frequencies0 - peaks_list #peaks[1::2]
    #print("\ndelta F =",delta_frequencies)

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
    print("\nBxyz = ({0:.2f}, {1:.2f}, {2:.2f}) G\t\t|B| = {3:.2f} G".format(Bsens[0],Bsens[1],Bsens[2],np.sqrt(Bsens[0]*Bsens[0]+Bsens[1]*Bsens[1]+Bsens[2]*Bsens[2])))
    #time.sleep(2)


t=time.time()
print("\nTotal time {0:.1f} s".format(t-t0))

print("\nSHUT DOWN")

print("RF OFF \t", end="")
print(ser.write(b"h0"))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))

print("RF OFF \t", end="")
print(ser.write(b"E0"))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))

print("Close port\t\t",ser.close())
print("Port is open\t\t",ser.is_open)
