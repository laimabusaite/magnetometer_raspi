import serial
import time
import os
import math

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

import spidev
import signal
import sys
import RPi.GPIO as GPIO

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

os.system("cls")

sleeptime=0.4

ser = serial.Serial()
ser.baudrate = 115200
ser.port = 'COM12'
ser.timeout = 4
print("SERIAL PORT")
print(ser)
time.sleep(sleeptime)
print("Port\t\t\t",ser.name)
time.sleep(sleeptime)
print("Open port\t\t",ser.open())
time.sleep(sleeptime)
print("Port is open\t\t",ser.is_open)

amp=15.0

#synth = SynthHD('COM12')
#synth.init()

print()
print("Model\t", end="")
print(ser.write(b"+"))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))

# Set channel 0 power and frequency
#synth[0].power = amp

send_amp="W{0:.1f}".format(amp)
print("{0:s}\t".format(send_amp), end="")
print(ser.write(send_amp.encode(encoding='UTF-8',errors='strict')))
temp_error=ser.readline()
temp_error=temp_error.decode('UTF-8')[1:]
print("{0:s}\t".format(temp_error))

# Enable channel 0
#synth[0].enable = True

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
    filename1="D:\\ESA\\test_windfreak_{0:s}.dat".format(s)

    print(filename1)

    f1 = open(filename1, 'w+')
    #print("FILE OPEN",f1)
    for i2 in range(len(x)):
        f1.write("%10.9f\t%10.9f\n" % (x[i2], y[i2]))
    f1.close()

    return


def full_scan(f0,dev,step_size):
    print("Full scan")
    t0=time.time()
    points=(2*dev)/step_size

    averages=4

    frequency_chan=[(f0-dev+i1*2*dev/points) for i1 in range(int(points))]
    average_chan=[0 for i1 in range(int(points))]

    progress_bar=0

    for i in range(int(points)):
        vmean=[0 for i1 in range(averages)]
        #print(" i={0:d}".format(i), end="", flush=True)
        f=(f0-dev+i*step_size)
        #if progress_bar==1:
        #    print("#", end="")
        #    progress_bar=0

        #if (((i*step_size/(2*dev))*100)/4).is_integer():
        #    progress_bar=1
        #synth[0].frequency = f

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
        for j in range(averages):
            #print("{0:d}".format(j), end="", flush=True)
            vmean[j]=(read_values(0))*4096/1000
            #print("{0:d}".format(j+1), end=" ", flush=True)
            vmean_sum1+=vmean[j]


        #print("{0:f}".format(vmean_sum/averages))
        #chan[1][i]=f
        average_chan[i]=vmean_sum1/averages

    write_file(frequency_chan,average_chan,"all_peaks")
    peaks_chan=[0 for i1 in range(int(points))]
    for i in range(int(points)):
        peaks_chan[i]=(-1)*average_chan[i]
    peaks_chan=np.array(peaks_chan)
    peaks, _ = find_peaks(peaks_chan, distance=50)
    plt.plot(peaks_chan)
    plt.plot(peaks, peaks_chan[peaks], "x")
    plt.show()
    frequency_chan=np.array(frequency_chan)
    print("{0:.1f} s".format(time.time()-t0))
    return frequency_chan[peaks]


def scan_peak(f0,dev,step_size):
    print("Peak scan")
    t0=time.time()
    points=(2*dev)/step_size

    averages1=1
    averages2=4

    frequency_chan=[(f0-dev+i1*2*dev/points) for i1 in range(int(points))]
    average_chan=[[0 for i1 in range(int(points))] for j1 in range(averages1)]
    for k in range(averages1):
        print("k={0:d}".format(k), end="", flush=True)
        for i in range(int(points)):
            vmean=[0 for i1 in range(averages2)]
            #print(" i={0:d}".format(i), end="", flush=True)
            f=(f0-dev+i*step_size)
            #synth[0].frequency = f

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
                #print("{0:d}".format(j), end="", flush=True)
                vmean[j]=(read_values(0))*4096/1000
                #print("{0:d}".format(j+1), end=" ", flush=True)
                vmean_sum1+=vmean[j]


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
    peaks_chan=[0 for i1 in range(int(points))]
    for i in range(int(points)):
        peaks_chan[i]=(-1)*average_chan1[i]
    peaks_chan=np.array(peaks_chan)
    peaks, _ = find_peaks(peaks_chan, distance=20)
    plt.plot(peaks_chan)
    plt.plot(peaks, peaks_chan[peaks], "x")
    plt.show(block=False)
    time.sleep(1)
    plt.close()
    frequency_chan=np.array(frequency_chan)
    print("{0:.1f} s".format(time.time()-t0))
    return frequency_chan[peaks]


#while True:
#    output1 = (read_values(0))*4096/1000
#    output2 = (read_values(0))*4096/1000
#    output3 = (read_values(0))*4096/1000
#    output4 = (read_values(0))*4096/1000
#    avg = (output1+output2+output3+output4)/4
#    print("{0:.4f} mV".format(avg))


dev0=30
t0=time.time()
# peaks=scan_peak(2415,dev0,0.5)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(2607,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(2805,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(2940,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(3113,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(3213,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(3323,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
# peaks=scan_peak(3400,dev0,2)
# peaks=scan_peak(peaks[0],5,0.5)
# peaks=scan_peak(peaks[0],2.5,0.25)
scan_peak(2905,520,0.5)
t=time.time()

print("{0:.1f} s".format(t-t0))

#peaks=full_scan(2880,100,2)
#print(peaks)

#t0=time.time()
#for i in range(len(peaks)):
#    scan_peak(peaks[i],5,0.5)

print("\nSHUT DOWN")
print("CLOSE RTB\t\t",virtb.close())
time.sleep(0.5)

#synth[0].enable = False
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
time.sleep(0.5)
print("Port is open\t\t",ser.is_open)