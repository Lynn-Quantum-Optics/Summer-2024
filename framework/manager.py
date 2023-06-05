import json
import time
import csv
import os
import datetime
import copy
from typing import Union
import serial
import numpy as np
import scipy.stats as stats
from ccu_controller import CCU
from motor_drivers import MOTOR_DRIVERS

class Manager:
    ''' Class for managing the automated laboratory equiptment.

    Parameters
    ----------
    out_file : str, optional
        The name of the output file to save the data to. If not specified, a timestamped csv will be saved.
    raw_data_out_file : Union[str,bool], optional
        The name of the output file to save the raw data to. If not specified or false, no raw data will be saved. If True, a timestamped csv will be saved.
    config : str, optional
        The name of the configuration file to load. Defaults to 'config.json'.
    '''
    def __init__(self, out_file:str=None, raw_data_out_file:Union[str,bool]=None, config:str='config.json'):
        # get the time of initialization for file naming
        init_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # +++ sorting out any raw data output +++

        # load configuration file
        with open(config, 'r') as f:
            self._config = json.load(f)
        
        # +++ CCU +++

        # sort out raw data output
        if raw_data_out_file is None or raw_data_out_file is False:
            raw_data_csv = None
        elif raw_data_out_file is True:
            raw_data_csv = f'{init_time}_raw.csv'
        elif os.path.isfile(raw_data_out_file):
            raw_data_csv = f'{init_time}_raw.csv'
            print(f'WARNING: raw data output file {raw_data_out_file} already exists. Raw data will be saved to {raw_data_csv} instead.')
        else:
            raw_data_csv = raw_data_out_file
        
        self._ccu = CCU(
            self._config['ccu']['port'],
            self._config['ccu']['baudrate'],
            raw_data_csv=raw_data_csv,
            **self._config['ccu']['plot_settings'])
        
        # +++ motors +++
        
        self._active_ports = {}
        self._motors = list(self._config['motors'].keys())
        
        # loop to initialize all motors
        for motor_name in self._motors:
            # check name
            if motor_name in self.__dict__ or not motor_name.isidentifier():
                raise ValueError(f'Invalid motor name \"{motor_name}\" (may be duplicate, collision with Manager method, or invalid identifier).')
            # get the motor arguments
            motor_dict = copy.deepcopy(self._config['motors'][motor_name])
            typ = motor_dict.pop('type')
            # conncet to com ports for elliptec motors
            if typ == 'Elliptec':
                port = motor_dict.pop('port')
                if port in self._active_ports:
                    com_port = self._active_ports[port]
                else:
                    com_port = serial.Serial(port, 9600, timeout=1)
                    self._active_ports[port] = com_port
                motor_dict['com_port'] = com_port
            # initialize motor
            self.__dict__[motor_name] = MOTOR_DRIVERS[typ](name=motor_name, **motor_dict)
        
        # +++ getting the output file ready +++

        # check for duplicate output file
        if out_file is not None and os.path.isfile(out_file):
            print(f'WARNING: Output file {out_file} already exists. Data will be saved to {init_time}.csv instead.')
            out_file = f'{init_time}.csv'
        
        # open output file and setup output file writer
        self._out_file = open(out_file, 'w+', newline='')
        self._out_writer = csv.writer(self._out_file)
        
        # write the column headers
        self._out_writer.writerow(\
            [f'start time (s)', 'stop time (s)'] + \
            ['num samples (#)', 'period per sample (s)'] + \
            [f'{m} position (rad)' for m in self._motors] + \
            [f'{k} rate (#/s)' for k in self._ccu.CHANNEL_KEYS] + \
            [f'{k} rate unc (#/s)' for k in self._ccu.CHANNEL_KEYS])
        
    # +++ methods +++

    def start(self) -> None:
        ''' Start the manager. '''
        self._ccu.start()

    def take_data(self, num_samp:int, samp_period:float) -> None:
        # record all motor positions
        motor_positions = [self.__dict__[m].pos for m in self._motors]

        # record start time
        start_time = time.time()

        # run trials
        data = np.row_stack([self._ccu.count_rates(samp_period) for _ in range(num_samp)])
        
        # record stop time
        stop_time = time.time()

        # calculate averages and uncertainties
        data_avg = np.mean(data, axis=0)
        data_unc = stats.sem(data, axis=0)

        # record data
        self._out_writer.writerow(\
            [start_time, stop_time] + \
            [num_samp, samp_period] + \
            motor_positions + \
            list(data_avg) + \
            list(data_unc))

    def configure_motors(self, **kwargs) -> None:
        ''' Configure the position of multiple motors at a time

        Parameters
        ----------
        kwargs : <NAME OF MOTOR> = <SET POSITION IN RADIANS>
            Assign each motor name that you wish to move the absolute angle to which you want it to move, in radians.
        '''
        for motor_name, position in kwargs.items():
            if not motor_name in self._motors:
                raise ValueError(f'Attempted to reference unknown motor \"{motor_name}\".')
            self.__dict__[motor_name].rotate_absolute(position)

    def set_meas_basis(self, basis:str) -> None:
        ''' Set the measurement basis for Alice and Bob's half and quarter wave plates. 
        
        Parameters
        ----------
        basis : str
            The measurement basis to set, should have length two. All options are listed in the config.
        '''
        # setup the basis
        A, B = basis
        self.configure_motors(
            A_HWP=self._config['basis_presets']['A_HWP'][A],
            A_QWP=self._config['basis_presets']['A_QWP'][A],
            B_HWP=self._config['basis_presets']['B_HWP'][B],
            B_QWP=self._config['basis_presets']['B_QWP'][B])

    def close(self) -> None:
        # close the output file (writes lines)
        self._out_file.close()
        # close all of the COM_PORT connections
        for port in COM_PORTS.values():
            port.close()
        # close the CCU connection
        self._ccu.close()

