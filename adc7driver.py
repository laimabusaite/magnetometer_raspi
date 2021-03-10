# This program will test interactions with the ADC 7 Click board
# from a Raspberry Pi.
#
# First, we need to load the necessary modules. 

import signal               # module to allow us to interrupt program
import sys                  # system module (to exit program)
import RPi.GPIO as GPIO     # RPi GPIO module
import spidev               # SPI module
import struct               # change bytes to integers
import time

# We need to put our data somewhere. We will be lazy for now
# and make a global variable for the file handle.

myDataFileName='datafile.txt'
myDataFile=open(myDataFileName,"w")

# Then we will define a callback that allows us to interrupt
# the program by pressing <CTRL-C>

def signal_handler(sig, frame):
    global myDataFile
    myDataFile.close()      # close the data file
    GPIO.cleanup()          # do some cleanup
    spi.close()
    sys.exit(0)             # exit program

# Next we will define a callback function to read the 
# data from the ADC 7 Click when the DRL signal indicates
# that data is ready.

def readADC7_callback(channel):
    # This is an edge event callback function!
    # It is run in a different thread to the main program.
    # 
    # We will be lazy and put the data in a global variable
    global myDataFile
    data=spi.readbytes(4)
    dataword=struct.unpack("<I", bytearray(data))
    myDataFile.write('%d\n',dataword)
    print('Writing data: %d\n',dataword)

# Now we are ready to start the main program.

if __name__ == '__main__':
# First, we will configure the GPIO pins for working with the
# ADC 7 Click installed on position 1 of the Click Shield.
    pinRST=5
    pinINT=6
    pinMCK=18
    pinDRL=4
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pinRST,GPIO.OUT)  # RST. Used for Filter Preset Enable (PRE)
    GPIO.setup(pinINT,GPIO.IN)   # INT. Busy Indicator (BSY)
    GPIO.setup(pinMCK,GPIO.OUT)  # PWM. Sampling Trigger (MCK)

# We leave pin GPIO13 as is for now. We need to write to it, in order
# to load a new control word into the LTC2500-32 chip, but we need 
# to read from it in order to know when Data is ready.

# Next, we configure the SPI interface.

    spi = spidev.SpiDev()        # Create instance of list
    spi.open(0,0)
    spi.max_speed_hz = 1000000      # SPI clock speed 10 MHz
    spi.lsbfirst = False          # Send MSB as first bit
    #spi.open(0,0)                 # Open Connection bus 0, device 0
# The first 0 is the bus number. We have only one SPI bus on the 
# RPi, so it must always be 0. The second parameter is the device
# number/chip select os CS(CS0 or CS1). RPI supports two SPI
# devices. CS0 controls the shield connectors. CS1 controls the
# additional 12-bit ADC on the click shield. 


# Now we will send a control word. 
    GPIO.setup(pinDRL,GPIO.OUT)       # We DRL to start communication
    GPIO.output(pinMCK,GPIO.LOW)      # The MCK should be 0.
    GPIO.output(pinDRL,GPIO.LOW)      # To start sending data, set DRL to 0.
# We will select the Averaging filter (0111) with downsampling
# factor DF32 (0101). 
    #send = [0x0b,0x57]            # Data word: 1011 0101 0111
    send = [0xb57]
    for i in range(1):
        print(i)
        time.sleep(0.1)
        print(spi.xfer(send))                # Send our word to LTC2500.

    GPIO.setup(pinDRL,GPIO.IN)
# Now we need to change DRL back to read, so it can tell serve 
# as Data Ready Indicator and tell us when data is ready. 

# Next we setup the PWM for pulse width modulation
# Let's start with a trigger rate of 250 kHz. I think it should 
# be possible to increase to 1 MHz, but I'm not sure.
# The important thing is that there is enough time to send the 
# data before the next event is ready.
# For 250 Khz, that time is 4 us. 
# With an SPI clock at 10 MHz, it takes 32*100ns=3.2us to send
# 32 bytes, so it's close. Might be safer to try slower 
# ADC trigger or faster SPI. Need to experiment.

    pwm = GPIO.PWM(pinMCK, 100)    # Set PWM 250 kHz.
    print(pwm.start(0.1))                # Start with duty cycle 10%.
# I believe the minimum pulse width is 4 ns, so 10% duty 
# cycle should be conservative.

# Next we add an edge detection function to call our callback
# function when DRL switches from 0 to 1.
    print("get to here")
    #channel = 4                  # DRL pin for click shield 1
    #GPIO.setup(pinDRL,GPIO.IN)
    GPIO.add_event_detect(pinDRL, GPIO.RISING, callback=readADC7_callback)  # add rising edge detection on a channel
    print("asd")
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        time.sleep(0.1)
    #signal.pause()
