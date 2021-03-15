import signal
import sys
import RPi.GPIO as GPIO
import spidev
import struct
import time

import pigpio

pi = pigpio.pi()

GPIO.setmode(GPIO.BCM)

pinDRL = 4
pinPRE = 5
pinBSY = 6
pinCS = 8
pinSCK = 11
pinSDI = 10
pinSDO = 9
pinMCK = 18

GPIO.setup(pinDRL, GPIO.IN)
GPIO.setup(pinPRE, GPIO.OUT)
GPIO.setup(pinBSY, GPIO.IN)
#GPIO.setup(pinMCK, GPIO.OUT)
pi.set_mode(pinMCK, pigpio.OUTPUT)

# SPI pins
GPIO.setup(pinCS, GPIO.OUT)
GPIO.setup(pinSCK, GPIO.OUT)
GPIO.setup(pinSDI, GPIO.OUT)
GPIO.setup(pinSDO, GPIO.IN)

GPIO.output(pinPRE, GPIO.HIGH) # if I set this and
GPIO.output(pinSDI, GPIO.HIGH) # this to high (averaging filter with digital gain compression / expansion off)


def signal_handler(sig, frame):
    pi.set_PWM_dutycycle(pinMCK, 0)
    GPIO.cleanup()
    print("\nexit")
    spi.close()
    sys.exit(0)


def read_data(channel):
    GPIO.output(pinCS, GPIO.LOW)
    data = spi.readbytes(4)
    print(data)
    print(data[0])
    print(data[1])
    print(data[2])
    print(data[3])
    #print("pin {0:d}".format(channel), end=" ")
    #print("data {0:s}".format(str(data)))
    data = struct.unpack("<I", bytearray(data))
    print(data)
    print()
    time.sleep(1)


def busy_pin(channel):
    data = spi.readbytes(4)
    print(data)
    print()
    time.sleep(1)


#frequency = 100
#pulse_time = 1000*40.0/1000000000
#duty_cycle = 100*frequency*pulse_time
#pwm = GPIO.PWM(pinMCK, 100)
#pwm.start(10)

pi.set_PWM_frequency(pinMCK, 1000)
pi.set_PWM_dutycycle(pinMCK, 1)
print(pi.get_PWM_frequency(pinMCK))
#pi.set_PWM_range(pinMCK, 40000)

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1000000
#spi.mode = 0b00
#spi.lsbfirst = False
#spi.no_cs = True

#print("send data")
#input()

#send = [0xb57] # 32 averages
#send = [0x07, 0x02, 0x00] # 4 averages
#send = [0x00, 0x02, 0x07] # 4 averages
#send = [0x07, 0x02] # averaging, 4 averages
send = [0x02]
#GPIO.output(pinCS, GPIO.LOW)
for _ in range(10):
    #rez = spi.xfer(send)
    #rez = spi.writebytes(send)
    #print("response {0:s}".format(str(rez)))
    #input()
    time.sleep(0.1)

#input()

GPIO.add_event_detect(pinDRL, GPIO.FALLING, callback=read_data)
#GPIO.add_event_detect(pinBSY, GPIO.FALLING, callback=busy_pin)

signal.signal(signal.SIGINT, signal_handler)

while True:
    #pin_status = GPIO.input(pinDRL)
    #print("DRL {0:d}".format(pin_status))
    #GPIO.output(pinMCK, GPIO.HIGH)
    time.sleep(0.001)
    #GPIO.output(pinMCK, GPIO.LOW)
    #time.sleep(0.009)
    #i += 1
    #if i == 64:
    #    data = spi.readbytes(4)
    #    print(data)
    #    i = 0
