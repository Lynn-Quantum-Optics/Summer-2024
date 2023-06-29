''' ccu.py

This file contains the FPGACCUController class, which is used to interface with the CCU for the experiment. It's most sophisticated method collects count rates for the detectors over some specified time period (with no uncertainties).

FPGA CCU Documentation: http://people.whitman.edu/~beckmk/QM/circuit/circuit.html

Author(s)
- Alec Roberson (aroberson@hmc.edu) 2023
- Kye W. Shi (kwshi@hmc.edu) 2018
'''

# python imports
from typing import Union, List
import functools as ft
import struct
import serial
import multiprocessing as mp

# package imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# base class for hardware monitoring interface

class SerialMonitor:
    ''' Base class for all serial monitors.

    Parameters
    ----------
    port : str
        Serial port to use (e.g. 'COM1' on windows, or
        '/dev/ttyUSB1' on unix).
    baud : int
        Serial communication baud rate (19200 for the Altera DE2 FPGA CC, 9600 for Nucleo-32).
    update_period : float
        How often the hardware will be sending new data (in seconds).
    channel_keys : list[str]
        List of channel keys associated with the data.
    termination_seq : bytes
        The byte termination sequence for each packet.
    plot_xlim : float
        The x limit for the plot (in seconds).
    plot_smoothing : float
        The smoothing factor for the plot (between 0 and 1).
    ignore : list[str]
        Any channels that should be ignored.
    rate_data : bool
        If true, then data returned by SerialMonitor.acquire_data will be rate data (i.e. measurements/second) and if false, then it will be a simple average over the sample period.
    '''

    # +++ BASIC METHODS +++

    def __init__(self, port:str, baud:int, update_period:float, channel_keys:List[str], termination_seq:bytes, plot_xlim:float, plot_smoothing:float, ignore:List[str], rate_data:bool) -> None:
        # set local vars
        self._port = port
        self._baud = baud
        self._update_period = update_period
        self._channel_keys = channel_keys
        self._num_chan = len(self._channel_keys)
        self._term_seq = termination_seq
        self._ignore = ignore
        self._rate_data = rate_data

        # put plot vars in terms of update period
        self._plot_smoothing = max(int(plot_smoothing/self._update_period), 1)
        # make sure xlim / smoothing is an integer
        self._num_plot_xlim = int(plot_xlim / self._update_period / self._plot_smoothing)
        self._num_plot_xlim -= (self._num_plot_xlim % self._plot_smoothing)
        self._plot_xlim = self._num_plot_xlim * self._plot_smoothing * self._update_period

        # open a pipeline for the listening process
        self._main_pipe, self._listener_pipe = mp.Pipe()

        # initialize a bunch of variables that only get set inside the listening process
        self._conn = None
        self._request = None
        self._plot_data = None
        self._data_buffer = None

        # initialize the subprocess
        self._listener = mp.Process(target=self._listening_subprocess)
        self._listener.start()

    def __repr__(self) -> str:
        return 'SerialMonitor'
    
    def __str__(self) -> str:
        return self.__repr__()

    # +++ CLEANUP +++
    
    def shutdown(self) -> None:
        ''' Terminates the listening process, closes the serial connections, and closes the plot. '''
        # send close notice
        self._main_pipe.send('close')
        # close the pipe
        self._main_pipe.close()
        # wait for the process to close
        self._listener.join()
        # close the plot
        plt.close()

    # +++ SERIAL INTERFACING +++

    def _flush(self, one=False) -> None:
        ''' Flushes the serial input buffer.
        
        Parameters
        ----------
        one : bool, optional
            If True, flushes current data until the next termination byte. Defaults to false, flushing all data.
        '''
        # check that there is a connection
        if self._conn is None:
            raise RuntimeError('Attempted to flush a connection that was never initialized! Did you call this method outside of the listening process?')
        # flush the whole buffer if one is false
        if not one:
            self._conn.reset_input_buffer()
        # skip all data until next termination
        all_read = b''
        while all_read[-len(self._term_seq):] != self._term_seq:
            all_read += self._conn.read()

    def _grab_data(self) -> None:
        ''' Reads all of the packets from the serial connection and updates the data buffer. '''
        # check that there is a connection
        if self._conn is None:
            raise RuntimeError('Attempted to grab data from non-existent connection. Did you call this method outside of the listening process?')
        # grab all packets in waiting
        while self._conn.in_waiting:
            # read next packet
            pkt = self._read_packet()
            # check valididty
            if pkt is None:
                # invalid packet, flush the buffer
                print(f'Invalid packet detected from {self}! Flushing buffer...')
                self._flush(one=True)
            else:
                # valid, add to buffer
                self._data_buffer.append(pkt)
                # if there is an active request, send it
                if self._request > 0:
                    self._listener_pipe.send(pkt)
                    self._request -= 1
        return
    
    def _listening_subprocess(self) -> None:
        # initialize the stuff for this process alone
        self._conn = serial.Serial(self._port, self._baud, timeout=2)
        self._request = 0
        self._plot_data = np.zeros((self._num_plot_xlim, self._num_chan))
        self._data_buffer = []

        # initialize plots
        self._init_plots()

        # flush the buffer once before jumping into the main loop
        self._flush()

        # main loop!
        while True:
            # grab all the data that is in waiting, sending packets if there is an active request
            self._grab_data()

            # take any requests
            if self._request == 0 and self._listener_pipe.poll():
                # get request
                self._request = self._listener_pipe.recv()
                
                # close if requested
                if self._request == 'close':
                    break
            
            # smoothing the data
            new_datas = []
            while len(self._data_buffer) > self._plot_smoothing:
                # get smoothed data
                new_datas.append(np.mean(self._data_buffer[:self._plot_smoothing], axis=0).reshape(1, self._num_chan))
                # trim the data buffer
                self._data_buffer = self._data_buffer[self._plot_smoothing:]
            self._plot_data = np.concatenate([self._plot_data] + new_datas, axis=0)

            # trim the plot data
            self._plot_data = self._plot_data[-self._num_plot_xlim:]

            # update the plots with the new data
            self._update_plots()

        # outside of the loop, close the connection to CCU and pipe
        self._conn.close()
        self._listener_pipe.close()

        return

    # +++ METHODS TO BE OVERRIDDEN +++ 
    
    def _read_packet(self) -> np.ndarray:
        ''' Method to read a packet of data from the serial connection to be overridden by subclasses.

        Returns
        -------
        np.ndarray
            The array containing the data for each channel from this packet.
        '''
        raise NotImplementedError()
    
    def _init_plots(self) -> None:
        raise NotImplementedError()

    def _update_plots(self) -> None:
        raise NotImplementedError()

    # +++ PUBLIC METHODS +++

    def acquire_data(self, num_samp:int, samp_period:float) -> np.ndarray:
        ''' Acquires the data from this SerialMonitor's connection.

        Parameters
        ----------
        num_samp : int
            The number of samples to take from the data.
        samp_period : float
            How long to sample the data per sample. Note that this will be rounded to the nearest multiple of the update period.

        Returns
        -------
        np.ndarray (num_chan,)
            The data from the SerialMonitor's connection.
        '''
        # sample period # of data points
        samp_period = max(int(samp_period / self._update_period), 1)
        # total number of data points needed
        num_data = num_samp * samp_period
        # send a request for all the data
        self._main_pipe.send(num_data)

        # wait for data
        data_out = []
        while len(data_out) < num_data:
            data_out.append(self._main_pipe.recv())
        
        # reshape and accumulate the data
        data_out = np.array(data_out).reshape(num_samp, samp_period, self._num_chan)
        
        if self._rate_data:
            # convert to rate data
            data_out /= self._update_period
        
        # average across each sample
        data_out = np.mean(data_out, axis=1)

        # take means and sems across samples
        data_avgs = np.mean(data_out, axis=0)
        data_sems = stats.sem(data_out, axis=0)
        
        # return the means and SEMs
        return data_avgs, data_sems

# ccu class

class CCU(SerialMonitor):
    ''' Interface for the Altera DE2 FPGA CCU.

    Parameters
    ----------
    port : str
        Serial port to use (e.g. 'COM1' on windows, or
        '/dev/ttyUSB1' on unix).
    baud : int
        Serial communication baud rate (19200 for the Altera DE2 FPGA CC, 9600 for Nucleo-32).
    plot_xlim : float
        The x limit for the plot (in seconds).
    plot_smoothing : float
        The smoothing factor for the plot (between 0 and 1).
    ignore : list[str]
        Any channels that should be ignored.
    '''
    
    # +++ BASIC METHODS +++

    def __init__(self, port:str, baud:int, plot_xlim:float, plot_smoothing:float, ignore:List[str]) -> None:
        super(CCU, self).__init__(
            port=port,
            baud=baud,
            update_period=0.1, # from device documentation
            channel_keys=['A', 'B', 'A\'', 'B\'', 'C4', 'C5', 'C6', 'C7'],
            termination_seq=b'\xff',
            plot_xlim=plot_xlim,
            plot_smoothing=plot_smoothing,
            ignore=ignore,
            rate_data=True)

    # +++ OVERRIDE METHODS +++

    def _read_packet(self) -> Union[np.ndarray,None]:
        ''' Read a packet from the serial connection to the CCU.
        
        Returns
        -------
        np.ndarray or None
            The array containing the data for each channel from this packet, or None if tha attempt to read a packet failed.
        '''
        # initialize output array
        out = np.zeros(self._num_chan)

        # read 8-counter measurements
        for i in range(8):
            # read in 5 bytes from the port
            packet = self._conn.read(size=5)
            # reduce to single unsigned integer
            # only bitshift by 7 to cut off the zero bit in each byte
            out[i] = ft.reduce(lambda v, b: (v << 7) + b, reversed(packet))
        
        # read the termination character
        if self._conn.read(len(self._term_seq)) != self._term_seq:
            return None
        else:
            return out

    def _init_plots(self) -> None:
        ''' Initialize the live plots that go with this. '''
        # initialize the figure with the four subplots
        self._fig = plt.figure(figsize=(8,8))
        axes = self._fig.subplots(2,2)

        # setup titles and such
        self._fig.canvas.manager.set_window_title('CCU Monitor')

        # setup all of the plot titles/axes

        self._axes = {
            'count_lines': axes[0][0],
            'coin_lines': axes[0][1],
            'count_bars': axes[1][0],
            'coin_bars': axes[1][1]}
        
        self._axes['count_lines'].set_title('Count Rates by Channel')
        self._axes['count_lines'].set_ylabel('Count Rates (#/s)')
        self._axes['count_lines'].set_xlabel('Time (s)')

        self._axes['coin_lines'].set_title('Coincidence Count Rates')
        self._axes['coin_lines'].set_ylabel('Coincidence Count Rates')
        self._axes['coin_lines'].set_xlabel('Time (s)')

        self._axes['count_bars'].set_ylabel('Coincidence Count Rates (#/s)')
        self._axes['coin_bars'].set_ylabel('Coincidence Count Rates (#/s)')

        # set limits on the x axes
        self._axes['count_lines'].set_xlim(-self._plot_xlim, 0)
        self._axes['coin_lines'].set_xlim(-self._plot_xlim, 0)

        # turn interactions on and show the plot without blocking
        self._fig.tight_layout()
        plt.ion()
        plt.show(False)

        # initialize all of the lines and such
        self._x_values = np.linspace(-self._plot_xlim, 0, self._num_plot_xlim)
        self._artists = {
            'count_lines': {},
            'coin_lines': {},
            'count_bars': None,
            'coin_bars': None}
        
        # count lines
        for k in self._channel_keys[:4]:
            if not k in self._ignore:
                self._artists['count_lines'][k], = self._axes['count_lines'].plot(self._x_values, np.zeros_like(self._x_values), label=k)

        # coin lines
        for k in self._channel_keys[4:]:
            if not k in self._ignore:
                self._artists['coin_lines'][k], = self._axes['coin_lines'].plot(self._x_values, np.zeros_like(self._x_values), label=k)
        
        # bar charts
        self._artists['count_bars'] = self._axes['count_bars'].bar(self._channel_keys[:4], np.ones((4,))*1e-5)
        self._artists['coin_bars'] = self._axes['coin_bars'].bar(self._channel_keys[4:], np.ones((4,))*1e-5)
        
        # add legends
        self._axes['count_lines'].legend()
        self._axes['coin_lines'].legend()

        # show the figure
        self._fig.show()
        self._fig.canvas.draw()

        # save the backgrounds
        self._axbgs = {}
        for k in self._axes:
            self._axbgs[k] = self._fig.canvas.copy_from_bbox(self._axes[k].bbox)

        return
    
    def _update_plots(self) -> None:
        # restore the backgrounds
        for k in self._axes:
            self._fig.canvas.restore_region(self._axbgs[k])
        
        # convert plot data to counts/second 
        data = self._plot_data/self._update_period

        # update plot limits
        self._axes['count_lines'].set_ylim(np.min(data[:,:4])*0.9, np.max(data[:,:4])*1.1+1e-5)
        self._axes['coin_lines'].set_ylim(np.min(data[:,4:])*0.9, np.max(data[:,4:])*1.1+1e-5)
        self._axes['count_bars'].set_ylim(0, np.max(data[-1,:4])*1.1+1e-5)
        self._axes['coin_bars'].set_ylim(0, np.max(data[-1,4:])*1.1+1e-5)

        # update count lines
        for i, k in enumerate(self._channel_keys[:4]):
            if not k in self._ignore:
                self._artists['count_lines'][k].set_ydata(data[:,i])
                self._axes['count_lines'].draw_artist(self._artists['count_lines'][k])
        
        # update coincidence count lines
        for i, k in enumerate(self._channel_keys[4:]):
            if not k in self._ignore:
                self._artists['coin_lines'][k].set_ydata(data[:,i+4])
                self._axes['coin_lines'].draw_artist(self._artists['coin_lines'][k])

        # update count bars
        for rect, h in zip(self._artists['count_bars'].patches, data[-1,:4]):
            rect.set_height(max(h,1e-5))
        for rect, h in zip(self._artists['coin_bars'].patches, data[-1,4:]):
            rect.set_height(max(h,1e-5))

        # update count labels
        self._axes['count_bars'].set_xticklabels([f'{k}\n{n:.2f}' for k, n in zip(self._channel_keys[:4], data[-1,:4])])
        self._axes['coin_bars'].set_xticklabels([f'{k}\n{n:.2f}' for k, n in zip(self._channel_keys[4:], data[-1,4:])])

        # blit the figure
        self._fig.tight_layout()
        self._fig.canvas.blit(self._axes['count_lines'].bbox)
        self._fig.canvas.blit(self._axes['coin_lines'].bbox)
        self._fig.canvas.blit(self._axes['count_bars'].bbox)
        self._fig.canvas.blit(self._axes['coin_bars'].bbox)

        # flush events
        self._fig.canvas.flush_events()

# laser monitoring class

class Laser(SerialMonitor):
    ''' Interface for the Nucleo-32 Microcontroller that is monitoring our laser.

    Parameters
    ----------
    port : str
        Serial port to use (e.g. 'COM1' on windows, or
        '/dev/ttyUSB1' on unix).
    baud : int
        Serial communication baud rate (19200 for the Altera DE2 FPGA CC, 9600 for Nucleo-32).
    plot_xlim : float
        The x limit for the plot (in seconds).
    plot_smoothing : float
        The smoothing factor for the plot (between 0 and 1).
    '''
    
    # +++ BASIC METHODS +++

    def __init__(self, port:str, baud:int, plot_xlim:float, plot_smoothing:float) -> None:
        super(Laser, self).__init__(
            port=port,
            baud=baud,
            update_period=0.1, # from device documentation
            channel_keys=['pdv', 'ldc', 'temp'],
            termination_seq=b'\xff\xff\xff\xff',
            plot_xlim=plot_xlim,
            plot_smoothing=plot_smoothing,
            ignore=[],
            rate_data=True)

    # +++ OVERRIDE METHODS +++

    def _read_packet(self) -> Union[np.ndarray,None]:
        ''' Read a packet from the serial connection to the Nucleo-32 chip.
        
        Returns
        -------
        np.ndarray or None
            The array containing the data for each channel from this packet, or None if tha attempt to read a packet failed.
        '''
        # initialize output array
        out = np.zeros(self._num_chan)
        
        # read each 32 bit float
        packet = self._conn.read(16)
        for i, j in enumerate(range(0, 12, 4)):
            out[i] = struct.unpack('<f', packet[j:j+4])[0]
        
        # read the termination character
        if packet[-len(self._term_seq):] != self._term_seq:
            return None
        else:
            return out

    def _init_plots(self) -> None:
        ''' Initialize the live plots that go with this. '''
        # initialize the figure with the four subplots
        self._fig = plt.figure(figsize=(6,8))
        axes = self._fig.subplots(3,1)

        # setup all of the plot titles/axes
        self._axes = {
            'pdv': axes[0],
            'ldc': axes[1],
            'temp': axes[2]}
        
        self._axes['pdv'].set_ylabel('Photodiode Voltage (V)')
        self._axes['pdv'].set_xlabel('Time (s)')
        
        self._axes['ldc'].set_ylabel('Laser Diode Current (mA)')
        self._axes['ldc'].set_xlabel('Time (s)')
        
        self._axes['temp'].set_ylabel('Temperature (C)')
        self._axes['temp'].set_xlabel('Time (s)')
        
        # set limits on the x axes
        self._axes['pdv'].set_xlim(-self._plot_xlim, 0)
        self._axes['ldc'].set_xlim(-self._plot_xlim, 0)
        self._axes['temp'].set_xlim(-self._plot_xlim, 0)

        # turn interactions on and show the plot without blocking
        self._fig.tight_layout()
        plt.ion()
        plt.show(False)

        # initialize all of the lines and such
        self._x_values = np.linspace(-self._plot_xlim, 0, self._num_plot_xlim)
        self._artists = {
            'pdv': self._axes['pdv'].plot(self._x_values, np.zeros_like(self._x_values))[0],
            'ldc': self._axes['ldc'].plot(self._x_values, np.zeros_like(self._x_values))[0],
            'temp': self._axes['temp'].plot(self._x_values, np.zeros_like(self._x_values))[0]}
        
        # show the figure
        self._fig.canvas.manager.set_window_title('Laser Monitor')
        self._fig.show()
        self._fig.canvas.draw()

        # save the backgrounds
        self._axbgs = {}
        for k in self._axes:
            self._axbgs[k] = self._fig.canvas.copy_from_bbox(self._axes[k].bbox)

        return
    
    def _update_plots(self) -> None:
        # restore the backgrounds
        for k in self._axes:
            self._fig.canvas.restore_region(self._axbgs[k])
        
        # don't convert to counts/second
        data = self._plot_data

        # update plot limits
        self._axes['pdv'].set_ylim(np.min(data[:,0])*0.9, np.max(data[:,0])*1.1+1e-5)
        self._axes['ldc'].set_ylim(np.min(data[:,1])*0.9, np.max(data[:,1])*1.1+1e-5)
        self._axes['temp'].set_ylim(np.min(data[:,2])*0.9, np.max(data[:,2])*1.1+1e-5)

        # update stuff
        for i, k in enumerate(self._channel_keys):
            # set limits
            self._axes[k].set_ylim(np.min(data[:,i])*0.9, np.max(data[:,i])*1.1+1e-5)
            # update artist
            self._artists[k].set_ydata(data[:,i])
            # draw artist
            self._axes[k].draw_artist(self._artists[k])

        # blit the figure
        self._fig.tight_layout()
        for k in self._channel_keys:
            self._fig.canvas.blit(self._axes[k].bbox)
        
        # flush events
        self._fig.canvas.flush_events()

if __name__ == '__main__':
    l = Laser('COM10', 9600, 60, 0.5)
