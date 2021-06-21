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

import subprocess

import datetime
import string

import fit_odmr

def exit_application():
    # Turning off the microwave generator and closing communication serial ports
    print("\nSHUT DOWN")

    print("\nRF OFF \t", end="")
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
    print()

    print("Close UART port\t\t",ser_uart.close())
    print("UART port is open\t\t",ser_uart.is_open)
    print()

    #GPIO.cleanup()
    spi.close()
    sys.exit(0)

    print("\nExit")
    # ----


def signal_handler(sig, frame):
    '''
    Handle ctr+c exit from the program.
    '''
    exit_application()


def read_values(channel):
    '''
    Read analog values from channel = 1 or 2 using the Pi 3 Click Shield.
    Outputs:
        Meassured analog value as a float number.
    '''
    adc = spi.xfer2([1,(8+channel)<<4,0])
    data = ((adc[1]&3) << 8) + adc[2]
    return data


def read_values_rp():
    '''
    Read serial data from RedPitaya STEMlab 125-14 using UART.
    Outputs:
        Received UART data as a float number.
    '''
    get_data = 1
    while get_data:
        received_data = ''
        endcharacter = ''
        while (endcharacter != b'\n'):
            endcharacter = ser_uart.read()
            try:
                received_data += str(endcharacter.decode('UTF-8'))
            except:
                print("Data decode error.\n", flush = True)

        try:
            received_data = float(received_data)
            if ((received_data > 0.8) and (received_data < 1.6)):
                get_data = 0
            else:
                print(".", flush = True) #Wrong UART data

        except:
            print("String to float error.\n", flush = True)

    return received_data


def write_file(x,y,s):
    '''
    Write meassured individual ODMR peak data to file with a timestamp.
    Outputs:
        ODMR signal data in two columns: ODMR signal intensity and microwave frequency.
    '''
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(":","-")
    timestamp = timestamp.replace(" ","_")
    filename1="test_data_rp/test_{0:s}_{1:s}.dat".format(s,timestamp)

    f1 = open(filename1, 'w+')
    for i2 in range(len(x)):
        f1.write("%10.9f\t%10.9f\n" % (x[i2], y[i2]))
    f1.close()

    return


def scan_peak(f0, dev, step_size, avg1, avg2, level, noise, write):
    '''
    Scan one ODMR peak:
        f0 - scan central frequency
        dev - defines the width of the scan: scan start frequency = f0 - dev, scan end frequency = f0 + dev
        step_size - scan step size in MHz
        avg1 - number of averaged ODMR scans
        level - expected / acceptable ODMR signal base level
        noise - maximum noise level for meassured data
        write - write to file yes or no (1 / 0)
    Outputs:
        One ODMR peak data: ODMR signal intensity as a function of microwaves.
    '''
    points=int((2*dev)/step_size)

    averages1=avg1
    averages2=avg2

    frequency_chan=[(f0-dev+i1*2*dev/points) for i1 in range(int(points))]
    average_chan=[[0 for i1 in range(int(points))] for j1 in range(averages1)]
    for k in range(averages1):
        for i in range(int(points)):
            vmean=[0 for i1 in range(averages2)]
            f=(f0-dev+i*step_size)
            send_frequency="f{0:.4f}".format(f)
            send_freq=ser.write(send_frequency.encode(encoding='UTF-8',errors='strict'))
            #print(send_freq)
            #temp_error=ser.readline()
            #temp_error=temp_error.decode('UTF-8')[1:]
            #print("{0:s}\t".format(temp_error))

            #time.sleep(350/1000000)
            vmean_sum1=0
            for j in range(averages2):
                while True:
                    vmean[j] = read_values_rp()
                    if (vmean[j]<level*(100+noise)/100) and (vmean[j]>level*(100-noise)/100):
                        vmean_sum1+=vmean[j]
                        break
                    else:
                        print("Noisy data, choosing next value")


            average_chan[k][i]=vmean_sum1/averages2


    average_chan1=[0 for i1 in range(int(points))]
    for i in range(int(points)):
        vmean_sum2=0
        for k in range(averages1):
            vmean_sum2+=average_chan[k][i]
            average_chan1[i]=vmean_sum2/averages1

    if write:
        write_file(frequency_chan[2:-2],average_chan1[2:-2],"dev{0:.1f}_peak{1:.1f}".format(dev,f0))

    return frequency_chan[2:-2], average_chan1[2:-2]


def get_baseline(f0,points):
    '''
    Meassure the ODMR signal intensity base level.
    Outputs:
        ODMR signal intensity base level as a float number.
    '''
    send_frequency="f{0:.4f}".format(f0)
    send_freq=ser.write(send_frequency.encode(encoding='UTF-8',errors='strict'))
    #print(send_freq)
    #temp_error=ser.readline()
    #temp_error=temp_error.decode('UTF-8')[1:]
    #print("{0:s}\t".format(temp_error))

    #time.sleep(350/1000000)
    vmean_sum1=0
    vmean=[0 for i1 in range(points)]
    for i in range(points):
        vmean[i] = read_values_rp()
        vmean_sum1+=vmean[i]

    return vmean_sum1/points


# Start of the program
os.system("clear")

signal.signal(signal.SIGINT, signal_handler)

# SPI communication initialization for meassuring analog values using the Pi 3 Click Board
spi = spidev.SpiDev()
spi.open(1,2)
spi.max_speed_hz = 115200
# ----

ser_uart = serial.Serial("/dev/ttyS0", 115200) # Setting up the UART connection to receive data from RedPitaya STEMlab 125-14

GPIO.setmode(GPIO.BCM) # do we need this ?

sleeptime=0.1

# Setting up the connection with the SynthNV PRO microwave generator
ser = serial.Serial()
ser.baudrate = 115200
ser.port = '/dev/ttyACM0'
ser.timeout = 1
print("\nSERIAL PORT")
time.sleep(sleeptime)
print("Port\t\t\t",ser.name)
time.sleep(sleeptime)
print("Open port\t\t",ser.open())
time.sleep(sleeptime)
print("Port is open\t\t",ser.is_open)
# ----

# Setting the microwave power in dBm
amp=15.0

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
# ----

# Menu
B1 = {}
i1 = 0
warmup = 32
t0=time.time()

while True:
    print("Select:")
    print("1 - Run calibration scan, setting all external magnetic field to zero")
    print("2 - Run calibration scan, maintaining all external magnetic field (get bias magnetic field of the device)")
    print("3 - Run N continious meassurements of magnetic field")
    print("0 - Exit")
    menu1 = input("->")
    if menu1 == "1":
        print("\nMicrowave generator warmup.")
        print(".",end="",flush=True)
        for _ in range(warmup-2):
            print(" ",end="",flush=True)

        print(".")
        for _ in range(warmup):
            level = get_baseline(2900,10)
            scan_peak(2600,20,2,8,1,level,10,0)
            scan_peak(2800,20,2,8,1,level,10,0)
            scan_peak(3300,20,2,8,1,level,10,0)
            scan_peak(3400,20,2,8,1,level,10,0)
            print("#", end="", flush=True)

        print("`",end="",flush=True)
        for _ in range(warmup-2):
            print(" ",end="",flush=True)

        print("`")
        print("Warmup done.\n")
        print("Running full ODMR peak scan.\n")
        level = get_baseline(2900,10)
        full_scan_mw, full_scan_odmr = scan_peak(2905,600,1,64,1,level,10,0)
        write_file(full_scan_mw, full_scan_odmr, "full_scan")
        B = 210
        theta = 80
        phi = 20
        Blab = CartesianVector(B, theta, phi)
        print(Blab)
        init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2], 'glor': 4.44, 'D': 2867.61, 'Mz1': 0, 'Mz2': 0, 'Mz3': 0, 'Mz4': 0}
        save_filename = "ODMR_fit_parameters.json"
        fit_odmr.fit_full_odmr(full_scan_mw, full_scan_odmr, init_params=init_params, save_filename=save_filename, debug=False)
        print("\nCalibration done.\n")
    elif menu1 == "2":
        print("3d funkcija")
    elif menu1 == "3":
        while True:
            print("Select:")
            print("1 - Use calibration file with external magnetic field set to zero")
            print("2 - Use calibration file with external magnetic field maintained")
            menu2 = input("->")
            print("Loading latest calibration file data, callculating initial system, initializing Ainv matrix ...")
            if menu2 == "1":
                frequencies0, A_inv = nv.calculateInitalSystem(filename = "ODMR_fit_parameters.json")
                break
            elif menu2 == "2":
                frequencies0, A_inv = nv.calculateInitalSystem(filename = "ODMR_fit_parameters_apliishi.json")
                break
            else:
                print("Undefined menu entry.")

        f2 = frequencies0[0]
        f4 = frequencies0[1]
        f6 = frequencies0[2]
        f8 = frequencies0[3]
        print("Done.\n")
        # Set meassuring parameters
        dev0=30 # Microwave scan width
        step=2 # Microwave scan step size in MHz
        a1 = int(input("Input number of ODMR scan averages: "))
        a2=1
        noise=10 # Maximum acceptable noise level
        sets = int(input("Input number of magnetic field meassurements, N = "))
        # ----

        print("\nMicrowave generator warmup.\n")
        print(".",end="",flush=True)
        for _ in range(warmup-2):
            print(" ",end="",flush=True)

        print(".")
        for _ in range(warmup):
            level = get_baseline(2900,10)
            scan_peak(2600,20,2,8,1,level,10,0)
            scan_peak(2800,20,2,8,1,level,10,0)
            scan_peak(3300,20,2,8,1,level,10,0)
            scan_peak(3400,20,2,8,1,level,10,0)
            print("#", end="", flush=True)

        print("`",end="",flush=True)
        for _ in range(warmup-2):
            print(" ",end="",flush=True)

        print("`")
        print("Warmup done.\n")

        print("\n###########################################################")
        for i in range(sets):
            time.sleep(0.05)
            print("\n {0:d}. ".format(i+1), end = "", flush = True)
            level = get_baseline(2900,10)

            peak2_MW, peak2_ODMR = scan_peak(f2,dev0,step,a1,a2,level,noise,1)
            peak4_MW, peak4_ODMR = scan_peak(f4,dev0,step,a1,a2,level,noise,1)
            peak6_MW, peak6_ODMR = scan_peak(f6,dev0,step,a1,a2,level,noise,1)
            peak8_MW, peak8_ODMR = scan_peak(f8,dev0,step,a1,a2,level,noise,1)

            peaks_list = []

            #peaks, amplitudes = detect_peaks(peak2_MW, peak2_ODMR, debug=False)
            #peaks_list.append(peaks)
            #peaks, amplitudes = detect_peaks(peak4_MW, peak4_ODMR, debug=False)
            #peaks_list.append(peaks)
            #peaks, amplitudes = detect_peaks(peak6_MW, peak6_ODMR, debug=False)
            #peaks_list.append(peaks)
            #peaks, amplitudes = detect_peaks(peak8_MW, peak8_ODMR, debug=False)
            #peaks_list.append(peaks)

            c1 = 0.002 # minimum contrast
            peaks = detect_peaks_weighted(peak2_MW, peak2_ODMR, min_contrast = c1)
            peaks_list.append(peaks)
            peaks = detect_peaks_weighted(peak4_MW, peak4_ODMR, min_contrast = c1)
            peaks_list.append(peaks)
            peaks = detect_peaks_weighted(peak6_MW, peak6_ODMR, min_contrast = c1)
            peaks_list.append(peaks)
            peaks = detect_peaks_weighted(peak8_MW, peak8_ODMR, min_contrast = c1)
            peaks_list.append(peaks)

            peaks_list = np.array(peaks_list).flatten()
            if (peaks_list == 0).any():
                print("No peak found.", flush = True)
                continue

            delta_frequencies = frequencies0 - peaks_list

            Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
            print("Bxyz = ({1:.2f}, {2:.2f}, {3:.2f}) G\t\t|B| = {4:.2f} G".format(i1+1,Bsens[0],Bsens[1],Bsens[2],np.sqrt(Bsens[0]*Bsens[0]+Bsens[1]*Bsens[1]+Bsens[2]*Bsens[2])), flush = True)
            B1[i1] = np.sqrt(Bsens[0]*Bsens[0]+Bsens[1]*Bsens[1]+Bsens[2]*Bsens[2])
            i1 += 1 # add +1 if 4 peaks found successfully
            f2 = peaks_list[0]
            f4 = peaks_list[1]
            f6 = peaks_list[2]
            f8 = peaks_list[3]
            subprocess.Popen(["./cplot", "{0:f}".format(Bsens[0]), "{0:f}".format(Bsens[1]), "{0:f}".format(Bsens[2])]) # run the magnetic field visualization application as a background process


        t=time.time()
        # Display the results of the magnetic field meassurements
        print("\n-----------------------------------------------------------")
        print("\nTotal time {0:.1f} s    Measurements {1:d}/{2:d} = {3:.0f} %    Rate {4:.2f} Hz".format(t-t0,i1,sets,100.0*i1/sets,sets/(t-t0)))

        Bmean = np.mean([B1[i] for i in range(i1)])
        Bstd = np.std([B1[i] for i in range(i1)])

        print("\nB = ({0:.2f} \u00B1 {1:.2f}) G\trB = {2:.2f} %".format(Bmean, Bstd, (Bstd/Bmean)*100))

        print("\n###########################################################\n")
        # ----
    elif menu1 == "0":
        exit_application()
    else:
        print("Undefined menu entry.")

# ----
