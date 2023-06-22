''' manager.py

Class for managing the automated laboratory equiptment, including data collection and motor manipulation.

Author(s):
- Alec Roberson (aroberson@hmc.edu) 2023
'''

# python imports
import json
import time
import csv
import os
import datetime
import copy
from typing import Union
import serial

# package imports
from tqdm import tqdm
import numpy as np
import pandas as pd

# local imports
from .monitors import CCU # , Laser # soon!
from .motor_drivers import MOTOR_DRIVERS

# manager class

class Manager:
    ''' Class for managing the automated laboratory equiptment.

    Parameters
    ----------
    out_file : str, optional
        The name of the output file to save the data to. If not specified, no output will be initialized.
    config : str, optional
        The name of the configuration file to load. Defaults to 'config.json'.
    debug : bool, optional
        If True, the CCU and motors will not be initialized with the manager, and will have to be initialized later with the init_ccu and init_motors methods.
    '''
    def __init__(self, out_file:str=None, config:str='config.json', debug:bool=False):
        # get the time of initialization for file naming
        self._init_time = time.time()
        self._init_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # load configuration file
        with open(config, 'r') as f:
            self._config = json.load(f)
        
        # save all initilaization parameters
        self.config_file = config

        # initialize output file variables
        self.out_file = None
        self._out_file = None # file object
        self._out_writer = None # csv writer

        # initialize ccu and motor class variables
        self._ccu = None
        self._motors = []
        self._active_ports = {}
        self.data = None # output data holding

        # initialize the log file
        if os.path.isfile('./mlog.txt'):
            os.remove('./mlog.txt')
        self._log_file = open('./mlog.txt', 'w+')
        self.log(f'Log file opened. Manager started at {self._init_time_str}.')

        # intialize everything if not debugging
        if not debug:
            self.init_ccu()
            self.init_motors()
            if out_file is not None:
                self.new_output(out_file)

    # +++ properties +++
    
    @property
    def motor_list(self) -> 'list[str]':
        ''' List of the string names of all motors. '''
        return self._motors
    
    @property
    def time(self) -> str:
        ''' String time since initalizing the manager, rounded to the nearest second. '''
        return str(datetime.timedelta(seconds=int(time.time()-self._init_time)))
    
    @property
    def now(self) -> str:
        return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

    # +++ initialization methods +++

    def init_ccu(self) -> None:
        ''' Initialize the CCU which starts live plotting. '''
        if self._ccu is not None:
            raise RuntimeError('CCU has already been initialized.')
        
        # initialize the ccu
        self._ccu = CCU(
            port=self._config['ccu']['port'],
            baud=self._config['ccu']['baudrate'],
            plot_xlim=self._config['ccu'].get('plot_xlim', 60),
            plot_smoothing=self._config['ccu'].get('plot_smoothing', 0.5),
            ignore=self._config['ccu'].get('ignore', []))
    
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
                    com_port = serial.Serial(port, timeout=5) # time out for all read
                    self._active_ports[port] = com_port
                motor_dict['com_port'] = com_port
            # initialize motor
            self.__dict__[motor_name] = MOTOR_DRIVERS[typ](name=motor_name, **motor_dict)

    # +++ file management +++

    def new_output(self, out_file:str) -> None:
        ''' Opens a new output file and initializes data collection

        Parameters
        ----------
        out_file : str
            The name of the output file to save the data to.
        '''
        # check if motors and ccu have been initialized
        if self._ccu is None:
            raise RuntimeError('Cannot initialize output file; CCU has not been initialized.')
        if len(self._motors) == 0:
            raise RuntimeError('Cannot initialize output file; No motors have been initialized.')
        
        # check output file for collisions
        if self._out_file is not None:
            raise RuntimeError('Output file has already been initialized.')
        elif os.path.isfile(out_file):
            raise RuntimeError(f'Output file {out_file} already exists.')
        
        # open output file and setup output file writer
        self.out_file = out_file
        self._out_file = open(self.out_file, 'w+', newline='')
        self._out_writer = csv.writer(self._out_file)
        
        # write the column headers
        self._out_writer.writerow(\
            [f'start time (s)', 'stop time (s)'] + \
            ['num samples (#)', 'period per sample (s)'] + \
            [f'{m} position (deg)' for m in self._motors] + \
            [f'{k} rate (#/s)' for k in self._ccu.CHANNEL_KEYS] + \
            [f'{k} rate SEM (#/s)' for k in self._ccu.CHANNEL_KEYS])

    def close_output(self, get_data:bool=True) -> Union[pd.DataFrame, None]:
        ''' Closes the output file

        Parameters
        ----------
        get_data : bool, optional
            If True, returns the data as a pandas dataframe. Defaults to True.
        
        Returns
        -------
        Union[pd.DataFrame, None]
            If get_data is True, returns the data as a pandas dataframe. Otherwise, returns None.
        '''
        # output file
        if self.out_file is None:
            raise RuntimeError('Cannot close output, output has not been initialized.')
        else:
            # close the output file (writes lines)
            self._out_file.close()
            self._out_file = None
            self._out_writer = None

        # grab the data
        if get_data:
            data = pd.read_csv(self.out_file)
            # convert columns to be useful
            data.columns = \
                [f't_start', 't_end'] + \
                ['num_samp', 'samp_period'] + \
                [f'{m}' for m in self._motors] + \
                [f'C{i}' for i in range(len(self._ccu.CHANNEL_KEYS))] + \
                [f'C{i}_sem' for i in range(len(self._ccu.CHANNEL_KEYS))]
        else:
            data = None

        # reset out_file
        self.out_file = None

        return data

    @staticmethod
    def read_csv(filename:str, config:str) -> pd.DataFrame:
        # read config to get motor names and such
        with open(config, 'r') as f:
            cfg = json.load(f)
        motors = list(cfg['motors'].keys())
        df = pd.read_csv(filename)
        df.columns = \
            [f't_start', 't_end'] + \
            ['num_samp', 'samp_period'] + \
            [f'{m}' for m in motors] + \
            [f'C{i}' for i in range(len(CCU.CHANNEL_KEYS))] + \
            [f'C{i}_sem' for i in range(len(CCU.CHANNEL_KEYS))]


    # +++ methods +++

    def take_data(self, num_samp:int, samp_period:float, *keys:str) -> 'tuple[np.ndarray, np.ndarray]':
        ''' Take detector data

        The data is written to the csv output table.

        Parameters
        ----------
        num_samp : int
            Number of samples to take.
        samp_period : float
            Collection time for each sample, in seconds. Note that this will be rounded to the nearest 0.1 seconds (minimum 0.1 seconds).
        *keys : str
            The CCU channel keys to return data for. All rates will be taken and written to the output file, only these rates will be returned. If no keys are given, all rates will be returned.
        
        Returns
        -------
        data_avg : np.ndarray
            The average number of counts per second for each channel.
        data_unc : np.ndarray
            The standard error of the mean for each channel.
        '''
        # check for output file
        if self.out_file == None:
            self.log('Warning: Collecting data without writing to output file.')
        
        # record all motor positions
        motor_positions = [self.__dict__[m].pos for m in self._motors]

        # record start time
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # run trials
        data_avg, data_unc = self._ccu.acquire_data(num_samp, samp_period)
        
        # record stop time
        stop_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # record data
        if self.out_file is not None:
            self._out_writer.writerow(\
                [start_time, stop_time] + \
                [num_samp, samp_period] + \
                motor_positions + \
                list(data_avg) + \
                list(data_unc))
        
        # return the right rates
        if len(keys) == 0:
            return data_avg, data_unc
        elif len(keys) == 1:
            k = keys[0]
            return data_avg[CCU.CHANNEL_KEYS.index(k)], data_unc[CCU.CHANNEL_KEYS.index(k)]
        else:
            out_avgs = np.array([data_avg[CCU.CHANNEL_KEYS.index(k)] for k in keys])
            out_uncs = np.array([data_unc[CCU.CHANNEL_KEYS.index(k)] for k in keys])
            return out_avgs, out_uncs

    def pct_det(self, basis1:str, basis2:str, num_samp:int, samp_period:float, chan:str='C4'):
        # take first measurement
        self.meas_basis(basis1)
        rate1, unc1 = self.get_rate(num_samp, samp_period, chan)
        # take first measurement
        self.meas_basis(basis2)
        rate2, unc2 = self.get_rate(num_samp, samp_period, chan)
        # get percentage and uncertainty
        pct = rate1 / (rate1 + rate2)
        unc = (rate2/(rate1+rate2)**2)*unc1 + (rate1/(rate1+rate2)**2)*unc2
        return pct, unc

    def log(self, note:str):
        line = self.time + f'\t{note}'
        print(line)
        self._log_file.write(line + '\n')

    def configure_motors(self, **kwargs) -> None:
        ''' Configure the position of multiple motors at a time

        Parameters
        ----------
        **kwargs : <NAME OF MOTOR> = <GOTO POSITION DEGREES>
            Assign each motor name that you wish to move the absolute angle to which you want it to move, in degrees.
        '''
        for motor_name, position in kwargs.items():
            # check motor exists
            if not motor_name in self._motors:
                raise ValueError(f'Attempted to reference unknown motor \"{motor_name}\".')
            # set motor position
            self.__dict__[motor_name].goto(position)

    def configure_motor(self, motor:str, pos:float) -> float:
        return self.__dict__[motor].goto(pos)

    def meas_basis(self, basis:str) -> None:
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

    def make_state(self, state:str) -> None:
        ''' Create a state from presets in the config file
        
        Parameters
        ----------
        state : str
            The state to create, one of the presets from the config file.
        '''
        # setup the state
        self.log(f'Loading state preset for {state} -> {self._config["state_presets"][state]}')
        self.configure_motors(**self._config['state_presets'][state])

    def sweep(self, component:str, pos_min:float, pos_max:float, num_steps:int, num_samp:int, samp_period:float) -> None:
        ''' Sweeps a component of the setup while collecting data
        
        Parameters
        ----------
        component : str
            The name of the component to sweep. Must be a motor name.
        pos_min : float
            The minimum position to sweep to, in degrees.
        pos_max : float
            The maximum position to sweep to, in degrees.
        num_steps : int
            The number of steps to perform over the specified range.
        num_samp : int
            Number of samples to take at each step.
        samp_period : float
            The period of each sample, in seconds (rounds down to nearest 0.1s, min 0.1s).
        '''
        # loop to perform the sweep
        for pos in tqdm(np.linspace(pos_min, pos_max, num_steps)):
            self.configure_motors(**{component:pos})
            self.take_data(num_samp, samp_period)

    # +++ shutdown methods +++

    def shutdown(self) -> None:
        ''' Shutsdown all the motors and terminates CCU processes, closing all com ports
        '''
        # motors
        if len(self._motors) == 0:
            print('WARNING: No motors are active.')
        else:
            # loop to delete motors
            for motor_name in self._motors:
                del self.__dict__[motor_name]
        # com ports
        if len(self._active_ports) == 0:
            print('WARNING: No com ports are active.')
        else:
            # loop to shutdown ports
            for port in self._active_ports.values():
                port.close()
        # output files
        if self.out_file is not None:
            self.close_output(get_data=False)
        self._log_file.close()
        # CCU
        self._ccu.shutdown()
