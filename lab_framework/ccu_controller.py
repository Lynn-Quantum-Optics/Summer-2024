''' ccu_controller.py
This file contains the FPGACCUController class, which is used to interface with the CCU for the experiment. It's most sophisticated method collects count rates for the detectors over some specified time period (with no uncertainties).

FPGA CCU Documentation: http://people.whitman.edu/~beckmk/QM/circuit/circuit.html

authors:
Alec Roberson (aroberson@hmc.edu)
Kye W. Shi (kwshi@hmc.edu)
'''
import serial as ser
import functools as ft
import numpy as np

class FPGACCUController:
    ''' Main controller for the Altera DE2 FPGA CCU.

    Parameters
    ----------
    port : str
        Serial port to use (e.g. 'COM1' on windows, or
        '/dev/ttyUSB1' on unix).
    baud : int
        Serial communication baud rate (19200 for the Altera DE2 FPGA CC).
    '''

    # +++ CLASS VARIABLES +++

    UPDATE_PERIOD = 0.1 # from device documentation
    # CHANNEL_KEYS = ['C0 (A)', 'C1 (B)', "C2 (A')", "C3 (B')", 'C4', 'C5', 'C6', 'C7'] # for our setup
    CHANNEL_KEYS = ['A', 'B', 'A\'', 'B\'', 'C4', 'C5', 'C6', 'C7'] # for our setup
    TERMINATION_BYTE = 0xff # from device documentation
    
    # +++ BASIC METHODS +++

    def __init__(self, port:str, baud:int) -> None:
        # set local vars
        self.port = port
        self.baud = baud
        # open connection
        self.connection = ser.Serial(self.port, self.baud)
    
    def __del__(self) -> None:
        # automatically close connection on cleanup
        self.connection.close()

    def __next__(self) -> np.ndarray:
        ''' Reads the next data packet in the CCU buffer. '''
        return self._read_packet()

    # +++ INTERNAL METHODS +++

    def _read_packet(self) -> np.ndarray:
        ''' Reads a packet of data from the FPGA CCU.

        Returns
        -------
        np.ndarray
            8-counter measurements from the FPGA CCU.
        '''

        out = np.zeros(8)

        # read 8-counter measurements
        for i in range(8):
            # read in 5 bytes from the port
            packet = self.connection.read(size=5)
            # reduce to single unsigned integer
            # only bitshift by 7 to cut off the zero bit in each byte
            out[i] = ft.reduce(lambda v, b: (v << 7) + b,
                                  reversed(packet))

        # read the termination character
        assert self.connection.read()[0] == self.TERMINATION_BYTE, 'misplaced termination character'

        return out
    
    def _flush(self) -> None:
        ''' Flushes the input buffer from the CCU. '''
        # TODO: is this the right way to do this?

        self.connection.reset_input_buffer()

        # skip leftover data, in case the buffer is flushed mid-read,
        # until next termination
        while self.connection.read()[0] != self.TERMINATION_BYTE:
            pass
    
    def _read(self, size=1):
        ''' Read the next size packets from the CCU.
        
        Parameters
        ----------
        size : int
            Number of packets to read.
        
        Returns
        -------
        np.ndarray
            Array of packets from the CCU.
        '''
        # clear the buffer
        self._flush()
        # start reading
        return np.row_stack(next(self) for _ in range(size))

    # +++ PUBLIC METHODS +++

    def get_count_rates(self, period:float) -> np.ndarray:
        ''' Acquires the coincidence count rates from the CCU over the specified period.

        Parameters
        ----------
        period : float
            Time to collect data for, in seconds.
        
        Returns
        -------
        np.ndarray of size (8,)
            Coincidence count RATES from the CCU. Each element corresponds to a detector/coincidence ['A', 'B', 'A\'', 'B\'', 'C4', 'C5', 'C6', 'C7'] (in order).
        '''
        # calculate number of samples to collect
        samples = max(int(period / self.UPDATE_PERIOD), 1)
        # flush the buffer
        self._flush()
        # read data and calculate rate
        data = self._read(samples)
        # accumulate data and convert to rate
        return np.sum(data, axis=0) / period

# cont = FPGACCUController('COM4', 19200)
