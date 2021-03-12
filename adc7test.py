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
pinSCK = 11
pinSDI = 10
pinSDO = 9
pinMCK = 18

GPIO.setup(pinDRL, GPIO.IN)
GPIO.setup(pinPRE, GPIO.OUT)
GPIO.setup(pinBSY, GPIO.IN)
GPIO.setup(pinMCK, GPIO.OUT)

# SPI pins
GPIO.setup(pinCS, GPIO.OUT)
GPIO.setup(pinSCK, GPIO.OUT)
GPIO.setup(pinSDI, GPIO.OUT)
GPIO.setup(pinSDO, GPIO.IN)

GPIO.output(pinPRE, GPIO.HIGH) # if I set this and
GPIO.output(pinSDI, GPIO.HIGH) # this to high (averaging filter with digital gain compression / expansion off)


def signal_handler(sig, frame):
    GPIO.cleanup()
    print("\nexit")
    spi.close()
    sys.exit(0)


def read_data(channel):
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


spi = spidev.SpiDev()
spi.open(0,0)
#spi.max_speed_hz = 1000000
#spi.mode = 0b00
#spi.lsbfirst = False
spi.no_cs = True

#print("send data")
#input()

#send = [0xb57] # 32 averages
#send = [0x07, 0x02, 0x00] # 4 averages
#send = [0x00, 0x02, 0x07] # 4 averages
#send = [0x07, 0x02] # averaging, 4 averages
send = [0x2]
for _ in range(1):
    #rez = spi.xfer(send)
    GPIO.output(pinCS, GPIO.LOW)
#    rez = spi.writebytes(send) # and don't send any SPI config, then I get [0, 0, 0, 0] result
#    print("response {0:s}".format(str(rez)))
    time.sleep(0.1)

#input()

frequency = 100
pulse_time = 1000*40.0/1000000000
duty_cycle = 100*frequency*pulse_time
pwm = GPIO.PWM(pinMCK, frequency)
pwm.start(duty_cycle)

GPIO.add_event_detect(pinDRL, GPIO.BOTH, callback=read_data)
#GPIO.add_event_detect(pinBSY, GPIO.FALLING, callback=busy_pin)

signal.signal(signal.SIGINT, signal_handler)

i=0
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
