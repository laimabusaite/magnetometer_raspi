import signal
import sys
import RPi.GPIO as GPIO
import spidev
import struct
import time

GPIO.setmode(GPIO.BCM)

pinDRL = 4
pinPRE = 5
pinBSY = 6
pinCS = 8
pinSDI = 10
pinMCK = 18

GPIO.setup(pinDRL, GPIO.IN)
GPIO.setup(pinPRE, GPIO.OUT)
GPIO.setup(pinBSY, GPIO.IN)
GPIO.setup(pinCS, GPIO.OUT)
GPIO.setup(pinMCK, GPIO.OUT)
GPIO.setup(pinSDI, GPIO.OUT)

GPIO.output(pinCS, GPIO.LOW)
GPIO.output(pinSDI, GPIO.HIGH) # if I set this and
GPIO.output(pinPRE, GPIO.HIGH) # this to high (averaging filter with digital gain compression / expansion off)


def signal_handler(sig, frame):
    GPIO.cleanup()
    print("\nexit")
    spi.close()
    sys.exit(0)


def read_data(channel):
    #pin_status = GPIO.input(pinDRL)
    #print("DRL {0:d}".format(pin_status))
    data = spi.readbytes(4)
    #data = struct.unpack("<I", bytearray(data))
    #print("pin {0:d}".format(channel), end=" ")
    print("data {0:s}\n".format(str(data)))
    time.sleep(0.1)


spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1000000
#spi.mode = 0b00
#spi.lsbfirst = False

#print("send data")
#input()

#send = [0xb57] # 32 averages
send = [0x27] # 4 averages
#send = [0x07, 0x02] # averaging, 4 averages
for _ in range(1):
    #rez = spi.xfer(send)
    #rez = spi.writebytes(send) # and don't send any SPI config, then I get [0, 0, 0, 0] result
    #print("response {0:s}".format(str(rez)))
    time.sleep(0.1)

#input()

pwm = GPIO.PWM(pinMCK, 10000)
pwm.start(0.1)

GPIO.add_event_detect(pinDRL, GPIO.RISING, callback=read_data)

signal.signal(signal.SIGINT, signal_handler)

while True:
    #pin_status = GPIO.input(pinDRL)
    #print("DRL {0:d}".format(pin_status))
    time.sleep(0.1)
