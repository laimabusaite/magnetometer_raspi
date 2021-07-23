import serial
import time

import spidev
import signal
# import sys
import RPi.GPIO as GPIO
import NVcenter as nv
from detect_peaks import *
from utilities import *
from measure_utilities import *
import subprocess
import datetime
import fit_odmr
import socket

def signal_handler(sig, frame):
    '''
    Handle ctr+c exit from the program.
    '''
    exit_application(ser, ser_uart, s, spi)



# Start of the program
os.system("clear")

signal.signal(signal.SIGINT, signal_handler)

# SPI communication initialization for measuring analog values using the Pi 3 Click Board
spi = spidev.SpiDev()
spi.open(1, 2)
spi.max_speed_hz = 115200
# ----

ser_uart = serial.Serial("/dev/ttyS0", 115200) # Setting up the UART connection to receive data from RedPitaya STEMlab 125-14

GPIO.setmode(GPIO.BCM)

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

# Set up the UDP data transfer settings
host = "192.168.0.102" # "169.254.77.253" # "169.254.191.69"
port = 4005

#server = ("169.254.191.69", 4000)
server = ("192.168.0.100", 8080)

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((host, port))

print()
message = input("Send identification address over UDP, ENTER")
s.sendto(message.encode("UTF-8"), server)
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
warmup = 32

while True:
    print("Select:")
    print("1 - Run calibration scan, setting all external magnetic field to zero")
    print("2 - Run calibration scan, maintaining all external magnetic field (get bias magnetic field of the device)")
    print("3 - Run N continuous measurements of magnetic field")
    print("4 - Run full scan")
    print("0 - Exit")
    menu1 = input("->")
    if menu1 == "1":
        #microwave_generator_warmup(warmup, ser, s)
        foldername1 = str(input("Input directory name: "))
        log_file_name = str(input("Input log file name for this measurement: "))
        step= float(input("Input microwave scan step: ")) # Microwave scan step size in MHz
        #timestamp = str(datetime.datetime.now())
        #timestamp = timestamp.replace(":","-")
        #timestamp = timestamp.replace(" ","_")
        #log_file_name1 = log_file_name+"_"+timestamp
        print("Running full ODMR peak scan.\n")
        level = get_baseline(2900,10, ser, s)
        full_scan_mw, full_scan_odmr = scan_peak(2905,600,step,64,1,level,10,1,log_file_name,foldername1, ser, s)
        #write_file(full_scan_mw, full_scan_odmr, "full_scan")
        B = 200
        theta = 80
        phi = 20
        Blab = CartesianVector(B, theta, phi)
        print(Blab)
        init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2], 'glor': 10., 'D': 2870.00, 'Mz1': 1.67, 'Mz2': 1.77, 'Mz3': 1.83, 'Mz4': 2.04}
        save_filename = "ODMR_fit_parameters.json"
        fit_odmr.fit_full_odmr(full_scan_mw, full_scan_odmr, init_params=init_params, save_filename=save_filename, debug=False)
        print("\nCalibration done.\n")
    elif menu1 == "2":
        print("3d function")
    elif menu1 == "3":
        while True:
            print("Select:")
            print("1 - Use calibration file with external magnetic field set to zero")
            print("2 - Use calibration file with external magnetic field maintained")
            menu2 = input("->")
            print("Loading latest calibration file data, calculating initial system, initializing Ainv matrix ...")
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
        # Set measuring parameters
        foldername1 = str(input("Input directory name: "))
        log_file_name = str(input("Input log file name for this measurement: "))
        timestamp = str(datetime.datetime.now())
        timestamp = timestamp.replace(":","-")
        timestamp = timestamp.replace(" ","_")
        dev0= int(input("Input microwave scan DEV: ")) # Microwave scan width
        step= float(input("Input microwave scan step: ")) # Microwave scan step size in MHz
        a1 = int(input("Input number of ODMR scan averages: "))
        a2=1
        noise=10 # Maximum acceptable noise level
        sets = int(input("Input number of magnetic field measurements, N = "))
        # ----

        log_file_name += "_dev{0:d}_avg{1:d}".format(dev0,a1)
        log_file_name1 = log_file_name+"_"+timestamp

        B1 = {}
        B1x = {}
        B1y = {}
        B1z = {}
        i1 = 0

        scan_direction = 0

        #microwave_generator_warmup(warmup, ser, s)

        t0=time.time()

        print("\n###########################################################")
        for i in range(sets):
            #time.sleep(0.05)
            print("\n {0:d}. ".format(i+1), end = "", flush = True)
            level = get_baseline(2900,10, ser, s)

            if scan_direction == 0:
                # scan peaks f2, f4, f6, f8
                peak2_MW, peak2_ODMR = scan_peak(f2,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)
                peak4_MW, peak4_ODMR = scan_peak(f4,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)
                peak6_MW, peak6_ODMR = scan_peak(f6,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)
                peak8_MW, peak8_ODMR = scan_peak(f8,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)

            else:
                # scan peaks f8, f6, f4, f2
                peak8_MW, peak8_ODMR = scan_peak(f8,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)
                peak6_MW, peak6_ODMR = scan_peak(f6,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)
                peak4_MW, peak4_ODMR = scan_peak(f4,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)
                peak2_MW, peak2_ODMR = scan_peak(f2,dev0,step,a1,a2,level,noise,1,log_file_name,foldername1, ser, s)

            scan_direction += 1
            if scan_direction > 1:
                scan_direction = 0

            peaks_list = []

            peaks = detect_peaks(peak2_MW, peak2_ODMR, debug=False)
            peaks_list.append(peaks)
            peaks = detect_peaks(peak4_MW, peak4_ODMR, debug=False)
            peaks_list.append(peaks)
            peaks = detect_peaks(peak6_MW, peak6_ODMR, debug=False)
            peaks_list.append(peaks)
            peaks = detect_peaks(peak8_MW, peak8_ODMR, debug=False)
            peaks_list.append(peaks)

            #c1 = 0.002 # minimum contrast
            #peaks = detect_peaks_weighted(peak2_MW, peak2_ODMR, min_contrast = c1)
            #peaks_list.append(peaks)
            #peaks = detect_peaks_weighted(peak4_MW, peak4_ODMR, min_contrast = c1)
            #peaks_list.append(peaks)
            #peaks = detect_peaks_weighted(peak6_MW, peak6_ODMR, min_contrast = c1)
            #peaks_list.append(peaks)
            #peaks = detect_peaks_weighted(peak8_MW, peak8_ODMR, min_contrast = c1)
            #peaks_list.append(peaks)

            peaks_list = np.array(peaks_list).flatten()
            if (peaks_list == 0).any():
                print("No peak found.", flush = True)
                continue

            delta_frequencies = frequencies0 - peaks_list

            Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
            Bsens = Bsens*(-1)
            print("Bxyz = ({1:.2f}, {2:.2f}, {3:.2f}) G\t\t|B| = {4:.2f} G".format(i1+1,Bsens[0],Bsens[1],Bsens[2],np.sqrt(Bsens[0]*Bsens[0]+Bsens[1]*Bsens[1]+Bsens[2]*Bsens[2])), flush = True)
            result1 = "{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}".format(Bsens[0],Bsens[1],Bsens[2],np.sqrt(Bsens[0]*Bsens[0]+Bsens[1]*Bsens[1]+Bsens[2]*Bsens[2]))
            write_file_log(foldername1,log_file_name, result1)
            B1[i1] = np.sqrt(Bsens[0]*Bsens[0]+Bsens[1]*Bsens[1]+Bsens[2]*Bsens[2])
            B1x[i1] = Bsens[0]
            B1y[i1] = Bsens[1]
            B1z[i1] = Bsens[2]
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
        rate1 = "Total time {0:.1f} s    Measurements {1:d}/{2:d} = {3:.0f} %    Rate {4:.2f} Hz".format(t-t0,i1,sets,100.0*i1/sets,sets/(t-t0))
        write_file_log(foldername1,log_file_name+"_summary",rate1)

        Bmean = np.mean([B1[i] for i in range(i1)])
        Bstd = np.std([B1[i] for i in range(i1)])/np.sqrt(i1-1)
        Bxmean = np.mean([B1x[i] for i in range(i1)])
        Bxstd = np.std([B1x[i] for i in range(i1)])/np.sqrt(i1-1)
        Bymean = np.mean([B1y[i] for i in range(i1)])
        Bystd = np.std([B1y[i] for i in range(i1)])/np.sqrt(i1-1)
        Bzmean = np.mean([B1z[i] for i in range(i1)])
        Bzstd = np.std([B1z[i] for i in range(i1)])/np.sqrt(i1-1)

        print("\nB = ({0:.2f} \u00B1 {1:.2f}) G\trB = {2:.2f} %".format(Bmean, Bstd, (Bstd/Bmean)*100))
        std_error1 = "B = ({0:.2f} \u00B1 {1:.2f}) G\trB = {2:.2f} %".format(Bmean, Bstd, (Bstd/Bmean)*100)
        print("\nBx = ({0:.2f} \u00B1 {1:.2f}) G\trBx = {2:.2f} %".format(Bxmean, Bxstd, np.abs((Bxstd/Bxmean)*100)))
        std_error1x = "Bx = ({0:.2f} \u00B1 {1:.2f}) G\trBx = {2:.2f} %".format(Bxmean, Bxstd, np.abs((Bxstd/Bxmean)*100))
        print("\nBy = ({0:.2f} \u00B1 {1:.2f}) G\trBy = {2:.2f} %".format(Bymean, Bystd, np.abs((Bystd/Bymean)*100)))
        std_error1y = "By = ({0:.2f} \u00B1 {1:.2f}) G\trBy = {2:.2f} %".format(Bymean, Bystd, np.abs((Bystd/Bymean)*100))
        print("\nBz = ({0:.2f} \u00B1 {1:.2f}) G\trBz = {2:.2f} %".format(Bzmean, Bzstd, np.abs((Bzstd/Bzmean)*100)))
        std_error1z = "Bz = ({0:.2f} \u00B1 {1:.2f}) G\trBz = {2:.2f} %".format(Bzmean, Bzstd, np.abs((Bzstd/Bzmean)*100))
        write_file_log(foldername1,log_file_name+"_summary",std_error1)
        write_file_log(foldername1,log_file_name+"_summary",std_error1x)
        write_file_log(foldername1,log_file_name+"_summary",std_error1y)
        write_file_log(foldername1,log_file_name+"_summary",std_error1z)

        print("\n###########################################################\n")
        # ----
    elif menu1 == "4":
        foldername1 = str(input("Input directory name: "))
        log_file_name = str(input("Input log file name for this measurement: "))
        #microwave_generator_warmup(warmup, ser, s)
        level = get_baseline(2900,10, ser, s)
        scan_peak(2905,600,1,64,1,level,10,1,log_file_name,foldername1, ser, s)
    elif menu1 == "0":
        exit_application(ser, ser_uart, s, spi)
    else:
        print("Undefined menu entry.")

# ----
