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
import pandas as pd
from ccu import CCU
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
    def __init__(self, out_file:str=None, raw_data_out_file:Union[str,bool]=None, config:str='config.json', debug:bool=False):
        # get the time of initialization for file naming
        self._init_time = time.time()
        self._init_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # load configuration file
        with open(config, 'r') as f:
            self._config = json.load(f)
        
        # save all initilaization parameters
        self.config_file = config
        self.out_file = out_file
        self.raw_data_out_file = raw_data_out_file

        # initialize other class variables
        self._ccu = None
        self._active_ports = {}
        self._motors = []
        self.data = None # output data holding
        
        # intialize everything if not debugging
        if not debug:
            self.init_ccu()
            self.init_motors()
            self.init_output_file()
    
    # +++ properties +++
    
    @property
    def motor_list(self) -> list[str]:
        ''' List of the string names of all motors. '''
        return self._motors
    
    @property
    def time(self) -> str:
        ''' String time since initalizing the manager, rounded to the nearest second. '''
        return str(datetime.timedelta(seconds=int(time.time()-self._init_time)))
    
    # +++ initialization methods +++

    def init_ccu(self) -> None:
        ''' Initialize the CCU which starts live plotting. '''
        if self._ccu is not None:
            raise RuntimeError('CCU has already been initialized.')
        
        # sort out raw data output
        if self.raw_data_out_file is None or self.raw_data_out_file is False:
            raw_data_csv = None
        elif self.raw_data_out_file is True:
            raw_data_csv = f'{self._init_time_str}_raw.csv'
        elif os.path.isfile(self.raw_data_out_file):
            raw_data_csv = f'{self._init_time_str}_raw.csv'
            print(f'WARNING: raw data output file {self.raw_data_out_file} already exists. Raw data will be saved to {raw_data_csv} instead.')
        else:
            raw_data_csv = self.raw_data_out_file
        
        # initialize the ccu
        self._ccu = CCU(
            self._config['ccu']['port'],
            self._config['ccu']['baudrate'],
            raw_data_csv=raw_data_csv,
            **self._config['ccu']['plot_settings'])
    
    def init_motors(self) -> None:
        ''' Initialize and connect to all motors. '''
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
                    com_port = serial.Serial(port, timeout=2)
                    self._active_ports[port] = com_port
                motor_dict['com_port'] = com_port
            # initialize motor
            self.__dict__[motor_name] = MOTOR_DRIVERS[typ](name=motor_name, **motor_dict)

    def init_output_file(self) -> None:
        # check output file for collisions
        if self._out_file is not None:
            raise RuntimeError('Output file has already been initialized.')
        
        # check for duplicate output or missing
        if out_file is not None and os.path.isfile(out_file):
            print(f'WARNING: Output file {out_file} already exists. Data will be saved to {self._init_time_str}.csv instead.')
            out_file = f'{self._init_time_str}.csv'
        elif out_file is None:
            print(f'WARNING: No output file specified. Data will be saved to {self._init_time_str}.csv instead.')
            out_file = f'{self._init_time_str}.csv'
        
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

    # +++ shutdown methods +++

    def shutdown(self) -> None:
        ''' Shutsdown all the motors and closes the output files. '''
        # close the output file (writes lines)
        self._out_file.close()
        # delete all motor objects and close all com ports
        for motor_name in self._motors:
            del self.__dict__[motor_name]
        for port in self._active_ports.values():
            port.close()
        # shutdown CCU connection
        self._ccu.shutdown()

        # collect output data
        self.data = pd.read_csv(self._out_file.name)
        return self.data
