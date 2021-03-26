import spidev
import time
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

while True:
    output1 = (read_values(0))*4096/1000
    output2 = (read_values(0))*4096/1000
    output3 = (read_values(0))*4096/1000
    output4 = (read_values(0))*4096/1000
    avg = (output1+output2+output3+output4)/4
    print("{0:.4f} mV".format(avg))
