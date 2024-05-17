from serial import Serial
from lab_framework import ElliptecMotor
import time

if __name__ == '__main__':
    # open the serial port
    s = Serial('COM7', timeout=10)
    
    # initialize the motor object
    m = ElliptecMotor('BCQWP', s, 'B')
    
    # send the optimize motor instruction
    m._send_instruction(b'om')
    
    # constantly print anything coming from the motor
    output = ''
    if 'GS00' in output or 'gs00' in output:
        print('Nominal status code detected. Exiting')
    elif m.com_port.in_waiting:
        out = str(m.com_port.readall())[2:-1]
        output += out
        print(out, end='')
    else:
        time.sleep(0.1)

