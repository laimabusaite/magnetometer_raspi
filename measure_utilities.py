# import serial
# import time
import os
# import math
#
# import numpy as np
#
# from scipy.signal import find_peaks
#
# import spidev
# import signal
import sys
# import RPi.GPIO as GPIO
#
# import NVcenter as nv
# from detect_peaks import *
# from utilities import *
# import glob
#
# import subprocess

import datetime
# import string
#
# import fit_odmr
#
# import socket


def exit_application(ser, ser_uart, s, spi):
    '''
    Parameters
    ----------
    ser: Setting up the connection with the SynthNV PRO microwave generator
    ser_uart: Setting up the UART connection to receive data from RedPitaya STEMlab 125-14
    s: socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    spi: SPI communication initialization for meassuring analog values using the Pi 3 Click Board
    '''
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

    s.close()

    #GPIO.cleanup()
    spi.close()
    sys.exit(0)

    print("\nExit")
    # ----


# def signal_handler(sig, frame):
#     '''
#     Handle ctr+c exit from the program.
#     '''
#     exit_application(ser, ser_uart, s, spi)


def read_values(channel, spi):
    '''
    Read analog values from channel = 1 or 2 using the Pi 3 Click Shield.
    Outputs:
        Meassured analog value as a float number.
    '''
    adc = spi.xfer2([1,(8+channel)<<4,0])
    data = ((adc[1]&3) << 8) + adc[2]
    return data


def read_values_rp(ser_uart):
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


def receive_udp_data(s):
    #print("{0:s}".format(data))
    get_data = 1
    while get_data:
        try:
            data, addr = s.recvfrom(16)
            data = data.decode("UTF-8")
            data = float(data)
            if ((data > 0.8) and (data < 1.6)):
                get_data = 0
            else:
                print(".", flush = True) #Wrong UDP data

        except:
            print("String to float error.\n", flush = True)

    return data


def write_file(x,y,folder_name,s1,s2=""):
    '''
    Write measured individual ODMR peak data to file with a timestamp.
    Outputs:
        ODMR signal data in two columns: ODMR signal intensity and microwave frequency.
    '''
    check_directory = os.path.isdir(folder_name)
    if not check_directory:
        os.makedirs(folder_name)
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(":","-")
    timestamp = timestamp.replace(" ","_")
    filename1="{0:s}/{1:s}_{2:s}_{3:s}.dat".format(folder_name,s1,s2,timestamp)

    f1 = open(filename1, 'w+')
    for i2 in range(len(x)):
        f1.write("%10.9f\t%10.9f\n" % (x[i2], y[i2]))
    f1.close()

    return


def write_file_log(foldername,s1,s2):
    '''
    Write log messages during the execution of the program
    Outputs:
        Log message file
    '''
    filename1="{0:s}/{1:s}.log".format(foldername,s1)

    f1 = open(filename1, 'a')
    #for i2 in range(len(x)):
    f1.write("%s\n" % (s2))
    f1.close()

    return


def scan_peak(f0, dev, step_size, avg1, avg2, level, noise, write, data_name,folder_name, ser, s):
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
                    vmean[j] = receive_udp_data(s) #read_values_rp()
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
        write_file(frequency_chan[2:-2],average_chan1[2:-2],folder_name,data_name,"dev{0:.1f}_peak{1:.1f}".format(dev,f0))

    return frequency_chan[2:-2], average_chan1[2:-2]


def get_baseline(f0,points, ser, s):
    '''
    Measure the ODMR signal intensity base level.
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
        vmean[i] = receive_udp_data(s) #read_values_rp()
        vmean_sum1+=vmean[i]

    return vmean_sum1/points


def microwave_generator_warmup(warmup, ser, s):
    print("\nMicrowave generator warmup.")
    print(".",end="",flush=True)
    for _ in range(warmup-2):
        print(" ",end="",flush=True)

    print(".")
    for _ in range(warmup):
        level = get_baseline(2900,10, ser, s)
        scan_peak(2600,20,2,8,1,level,10,0,"","", ser, s)
        scan_peak(2800,20,2,8,1,level,10,0,"","", ser, s)
        scan_peak(3300,20,2,8,1,level,10,0,"","", ser, s)
        scan_peak(3400,20,2,8,1,level,10,0,"","", ser, s)
        print("#", end="", flush=True)

    print("\n`",end="",flush=True)
    for _ in range(warmup-2):
        print(" ",end="",flush=True)

    print("`")
    print("Warmup done.\n")

