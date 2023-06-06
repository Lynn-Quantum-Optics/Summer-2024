''' ccu.py

This file contains the FPGACCUController class, which is used to interface with the CCU for the experiment. It's most sophisticated method collects count rates for the detectors over some specified time period (with no uncertainties).

FPGA CCU Documentation: http://people.whitman.edu/~beckmk/QM/circuit/circuit.html

Author(s)
- Alec Roberson (aroberson@hmc.edu) 2023
- Kye W. Shi (kwshi@hmc.edu) 2018
'''

# python imports
import functools as ft
import serial
import multiprocessing as mp
import datetime
import csv

# package imports
import numpy as np
import matplotlib.pyplot as plt

# ccu class

class CCU:
    ''' Interface for the Altera DE2 FPGA CCU.

    Parameters
    ----------
    port : str
        Serial port to use (e.g. 'COM1' on windows, or
        '/dev/ttyUSB1' on unix).
    baud : int
        Serial communication baud rate (19200 for the Altera DE2 FPGA CC).
    raw_data_csv : str
        Path to the raw data csv file to write to. If None, no file is written.
    ignore : list[str]
        Any channels that should be ignored.
    '''

    # +++ class variables +++
    UPDATE_PERIOD = 0.1 # from device documentation
    CHANNEL_KEYS = ['A', 'B', 'A\'', 'B\'', 'C4', 'C5', 'C6', 'C7'] # for our setup
    TERMINATION_BYTE = 0xff # from device documentation

    # +++ plotting parameters +++
    # all in units of 0.1s
    PLOT_XLIM = 600
    PLOT_SMOOTHING = 30
    
    # +++ BASIC METHODS +++

    def __init__(self, port:str, baud:int, raw_data_csv=None, ignore:'list[str]'=[]) -> None:
        # set local vars
        self.port = port
        self.baud = baud
        
        # open pipeline
        self._pipe, conn_to_self = mp.Pipe()

        # start listening process
        self._listening_process = mp.Process(target=CCU._listen_and_plot, args=(conn_to_self, self.port, self.baud, raw_data_csv, ignore))
        self._listening_process.start()

    # +++ CLEANUP +++
    
    def shutdown(self) -> None:
        ''' Closes the connection to the CCU and terminates the listening process. '''
        # send close notice
        self._pipe.send('close')
        # close the pipe
        self._pipe.close()
        # wait for the process to close
        self._listening_process.join()
        # close the plot
        plt.close()

    # +++ ALL INTERFACING WITH CCU +++

    @staticmethod
    def _flush(conn, one=False) -> None:
        ''' Flushes the input buffer from the CCU. 
        
        Parameters
        ----------
        conn : serial.Serial
            Serial connection to the FPGA CCU.
        one : bool, optional
            If True, flushes current data until the next termination byte. Defaults to false, flushing all data.
        '''
        if not one:
            conn.reset_input_buffer()

        # skip all data until next termination
        while conn.read()[0] != CCU.TERMINATION_BYTE:
            pass
    
    @staticmethod
    def _read_packet(conn) -> np.ndarray:
        ''' Reads a packet of data from the FPGA CCU.

        Parameters
        ----------
        conn : serial.Serial
            Serial connection to the FPGA CCU.

        Returns
        -------
        np.ndarray
            8-counter measurements from the FPGA CCU.
        '''

        out = np.zeros(8)

        # read 8-counter measurements
        for i in range(8):
            # read in 5 bytes from the port
            packet = conn.read(size=5)
            # reduce to single unsigned integer
            # only bitshift by 7 to cut off the zero bit in each byte
            out[i] = ft.reduce(lambda v, b: (v << 7) + b,
                                  reversed(packet))

        # read the termination character
        if conn.read()[0] != CCU.TERMINATION_BYTE:
            print('Misplaced termination byte! Skipping to next packet.')
            return None

        return out
    
    @staticmethod
    def _grab_data(conn:serial.Serial, current_index:int, writer:csv.writer=None) -> 'tuple[list[np.ndarray], int]':
        ''' Collects all data packets waiting in the buffer from connection.

        Parameters
        ----------
        conn : serial.Serial
            Serial connection to the FPGA CCU.
        current_index : int
            The index of the next packet to be collected.
        writer : csv.writer, optional
            If specified, writes the data to a csv file. Defaults to None.

        Returns
        -------
        list[np.ndarray]
            List of all packets collected.
        int
            The index of the next packet to be collected.
        '''
        # get the collection time for all these packets
        collection_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # initialize list for output packets
        out_packets = []

        # loop through all packets waiting in the buffer
        while conn.in_waiting:
                # read a packet
                packet = CCU._read_packet(conn)
                # if the packet is valid, add it to the buffer
                if packet is not None: # valid packet
                    out_packets.append(packet)
                    # write to csv if specified
                    if writer:
                        writer.writerow([current_index, collection_time] + packet.tolist())
                    current_index += 1
                else: # invalid packet, flush to next termination
                    CCU._flush(conn, one=True)
                    if writer:
                        writer.writerow(['invalid packet encountered, buffer flushed', collection_time] + [0]*8)
        # return the packets and the next data index
        return out_packets, current_index

    @staticmethod
    def _send_data(pipe:mp.Pipe, data_packets:'list[np.ndarray]', current_index:int, next_to_send:int, request:int) -> 'tuple[int, int]':
        ''' Retrieves all data from from the data packets.
        
        Parameters
        ----------
        pipe : mp.Pipe
            Serial connection to the main process.
        data_packets : list[np.ndarray]
            List of the most recent packets collected.
        current_index : int
            The data index of the next data packet that will be collected.
        next_to_send : int
            The data index of the next data packet that will be sent.
        request : int
            The number of packet requests that are still outstanding.

        Returns
        -------
        int : next_to_send
            The data index of the next data packet that has not been sent.
        int : request
            The number of packet requests that are still outstanding after this method.

        '''
        # number of data packets
        data_len = len(data_packets)
        # number of packets that could be send (haven't been sent yet)
        num_valid_packets = current_index - max(next_to_send, current_index - data_len)
        # actual start index of the next valid packet to send
        i = data_len - num_valid_packets

        # start sending until we fulfill the request or run out of packets
        while (i < data_len) and request > 0:
            pipe.send(data_packets[i])
            # increment counters
            next_to_send += 1
            i += 1
            request -= 1
        
        # return the updated counters
        return next_to_send, request
    
    @staticmethod
    def _init_plots():
        # initialize the plot
        fig = plt.figure(figsize=(8,8))
        ax_count_lines = fig.add_subplot(221)
        ax_coin_lines = fig.add_subplot(222)
        ax_count_bars = fig.add_subplot(223)
        ax_coin_bars = fig.add_subplot(224)

        # autoscaling
        ax_count_lines.autoscale(enable=True, axis='both')
        ax_coin_lines.autoscale(enable=True, axis='both')

        # title plots and such
        CCU._title_plots(ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig)

        # open the plots
        plt.ion()
        plt.show()

        # return axes for plotting on
        return ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig
    
    @staticmethod
    def _title_plots(ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig):
        # set axis labels
        ax_count_lines.set_title('Count rates for Detectors')
        ax_count_lines.set_xlabel('Data Index')
        ax_count_lines.set_ylabel('Counts / Second')

        ax_coin_lines.set_title('Coincidence Count Rates')
        ax_coin_lines.set_xlabel('Data Index')
        ax_coin_lines.set_ylabel('Counts / Second')

        ax_count_bars.set_title('Count Rates')
        ax_count_bars.set_xlabel('Channel')
        ax_count_bars.set_xlabel('Counts / Second')

        ax_coin_bars.set_title('Coincidence Count Rates')
        ax_coin_bars.set_xlabel('Channel')
        ax_coin_bars.set_ylabel('Counts / Second')

        # pack
        fig.tight_layout()

    @staticmethod
    def _update_plots(ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig, plot_data_packets, current_index, ignore):
        # get the xdata
        xdata = np.arange(current_index - CCU.PLOT_XLIM, current_index, CCU.PLOT_SMOOTHING)

        # plot all lines
        for k_idx, ax in zip([0, 4], [ax_count_lines, ax_coin_lines]):
            # clear the axes
            ax.cla()
            # plot the current set of lines
            for i, k in enumerate(
                CCU.CHANNEL_KEYS[k_idx:k_idx+4]):
                # skip if ignored
                if k in ignore:
                    continue
                # collect ydata as counts/second
                ydata = np.array([packet[i+k_idx] for packet in plot_data_packets])
                # plot the data
                ax.plot(xdata, ydata, label=k)
            ax.legend()
        
        # plot the bar charts
        for k_idx, ax in zip([0,4], [ax_count_bars, ax_coin_bars]):
            # clear the axes
            ax.cla()
            # collect the most recent ydata as counts/second
            ydata = plot_data_packets[-1][k_idx:k_idx+4]
            # make names
            names = [f'{k}\n{y:.2f}' for k, y in zip(CCU.CHANNEL_KEYS[k_idx:k_idx+4], ydata)]
            # plot the bars
            ax.bar(names, ydata)
        
        # title plots
        CCU._title_plots(ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig)
 
        # update the plots
        plt.pause(0.01)

    @staticmethod
    def _listen_and_plot(p:mp.Pipe, port:str, baud:int, csv_out:str=None, ignore:'list[str]'=[]) -> None:
        ''' Listens to the CCU and plots the data in real time.
        This method is intended to be run once in a seperate process.
        
        Parameters
        ----------
        p : mp.Pipe
            Pipe for communication with the main process.
        port : str
            The serial port to connect to.
        baud : int
            The baud rate to use.
        csv_out : str, optional
            If specified, writes the data to a csv file. Defaults to None.
        ignore : list[str], optional
            List of channels to ignore. Defaults to [].
        '''
        # create output csv if specified
        if csv_out is not None:
            csvfile = open(csv_out, 'w+', newline='')
            writer = csv.writer(csvfile)
            writer.writerow(['idx', 'collection time'] + CCU.CHANNEL_KEYS)
        else:
            writer = None
        
        # open the serial connection
        conn = serial.Serial(port, baud)

        # initialize data buffer
        data_packets = []
        current_index = 0

        # initialize plot variables
        ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig = CCU._init_plots()

        # index of last plot
        # counts up by 1 for every CCU.PLOT_SMOOTHING packets
        last_plot = 0

        # number of data points that will be in each plot
        PLOT_DATA_LIM = CCU.PLOT_XLIM // CCU.PLOT_SMOOTHING

        # initialize plot data
        plot_data = list(np.zeros((PLOT_DATA_LIM, len(CCU.CHANNEL_KEYS))))

        # initialize requests
        request = 0 # number of data points to send
        next_to_send = 0 # index of next valid packet to send

        # flush the buffer once before beginning
        CCU._flush(conn)


        # main loop!
        while True:
            # get the current time for this collection period
            new_packets, current_index = CCU._grab_data(conn, current_index, writer)
            
            # update and trim the data buffer
            data_packets += new_packets
            if len(data_packets) > CCU.PLOT_XLIM:
                data_packets = data_packets[-CCU.PLOT_XLIM:]

            # take any requests
            if request == 0 and p.poll():
                # get request
                request = p.recv()
                
                # check if it's a close request
                if request == 'close':
                    # break out of the loop
                    break
                else:
                    # reset the last sent index
                    # this ensures we only send data collected AFTER the request came in
                    next_to_send = current_index - 1
            
            # attempt to fulfull requests
            if request:
                # use the method for sending data
                next_to_send, request = CCU._send_data(p, data_packets, current_index, next_to_send, request)

            # +++ PLOTTING THE DATA +++

            # check if there is plot data to update
            if last_plot < (current_index-CCU.PLOT_SMOOTHING) // CCU.PLOT_SMOOTHING:
                # smallest index not included in plot
                last_plot_i = len(data_packets) + last_plot*CCU.PLOT_SMOOTHING - current_index
                num_to_add = (current_index - last_plot*CCU.PLOT_SMOOTHING)//CCU.PLOT_SMOOTHING
                # loop through new data to add
                for i in range(num_to_add):
                    # grab the data
                    start = last_plot_i + i*CCU.PLOT_SMOOTHING
                    new_data = np.array(data_packets[start:start+CCU.PLOT_SMOOTHING])
                    # convert to count/sec and smooth
                    new_data = np.mean(np.array(new_data),axis=0) / CCU.UPDATE_PERIOD
                    # add to plot data
                    plot_data.append(new_data)
                # +++ update plots +++
                plot_data = plot_data[-PLOT_DATA_LIM:]
                c_idx = current_index - (current_index % CCU.PLOT_SMOOTHING)
                CCU._update_plots(ax_count_lines, ax_coin_lines, ax_count_bars, ax_coin_bars, fig, plot_data, c_idx, ignore)
                # +++ update last_plot +++
                last_plot += num_to_add

        # outside of the loop, close the connection to CCU and pipe
        conn.close()
        p.close()
        
        # close csv file
        if csv_out is not None:
            csvfile.close()
        
        return

    # +++ PUBLIC METHODS +++

    def count_rates(self, period:float) -> np.ndarray:
        ''' Acquires the coincidence count rates from the CCU over the specified period.

        Parameters
        ----------
        period : float
            Time to collect data for, in seconds.
        
        Returns
        -------
        np.ndarray of size (8,)
            Coincidence count RATES from the CCU. Each element corresponds to a detector/coincidence [A, B, A', B', C4, C5, C6, C7] (in order).
        '''
        # calculate number of data points to collect
        num_data = max(int(period / self.UPDATE_PERIOD), 1)
        # request data
        self._pipe.send(num_data)
        # wait for data
        data_out = []
        while len(data_out) < num_data:
            data_out.append(self._pipe.recv())
        # accumulate data and convert to rate
        actual_period = num_data * self.UPDATE_PERIOD # may be different from period
        return np.sum(data_out, axis=0) / actual_period
