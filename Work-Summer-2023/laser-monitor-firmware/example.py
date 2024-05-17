import struct
import serial

# constants
TERMINATION_SEQUENCE = b'\xff\xff\xff\xff'
AVERAGE_OVER = 1 # units of 0.1 seconds (or firmware delay)

# open a serial connection
s = serial.Serial('/dev/tty.usbmodem1103', baudrate=9600)

# flush until termination sequence is found
s.flush()
all_read = b''
while all_read[-len(TERMINATION_SEQUENCE):] != TERMINATION_SEQUENCE:
    all_read += s.read(1)

# main loop to read data
while True:
    # read data from serial port
    data = s.read(16)
    # decode the floating point numbers
    pdm = struct.unpack('<f', data[0:4])[0]
    ldc = struct.unpack('<f', data[4:8])[0]
    temp = struct.unpack('<f', data[8:12])[0]
    # assert the termination sequence
    assert data[12:16] == TERMINATION_SEQUENCE
    # print the collected data
    print('-----')
    print(f'PDM: {pdm}\nLDC: {ldc}\nTemp: {temp}')
    print('-----')
