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
from typing import Union, List
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
        self._motors = None
        self._active_ports = {}
        self.data = None # output data holding

        # initialize the log file
        if os.path.isfile('./mlog.txt'):
            os.remove('./mlog.txt')
        self._log_file = open('./mlog.txt', 'w+')
        self.log(f'Manager started at {self._init_time_str}.')
        self.log(f'Configuration file: "{config}".')

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
        self.log('Initializing CCU.')

        if self._ccu is not None:
            self.log('CCU already initialized; throwing error.')
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
        self.log('Initializing motors.')

        if not self._motors is None:
            self.log('Motors already initialized, throwing error.')
            raise RuntimeError('Motors have already been initialized.')

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
        
        # check that we don't currently have a file open
        if self._out_file is not None:
            self.log('Output file already initialized; throwing error.')
            raise RuntimeError('Output file has already been initialized.')
        # check that the file doesn't already exist
        if os.path.isfile(out_file):
            self.log(f'Given output file "{out_file}" already exists; throwing error.')
            raise RuntimeError(f'Output file "{out_file}" already exists.')
        
        # open output file and setup output file writer
        self.log(f'Opening new output file "{out_file}".')
        self.out_file = out_file
        self._out_file = open(self.out_file, 'w+', newline='')
        self._out_writer = csv.writer(self._out_file)
        
        # write the column headers
        self._out_writer.writerow(\
            [f'start time (s)', 'stop time (s)'] + \
            ['num samples (#)', 'period per sample (s)'] + \
            [f'{m} position (deg)' for m in self._motors] + \
            [f'{k} rate (#/s)' for k in self._ccu._channel_keys] + \
            [f'{k} rate SEM (#/s)' for k in self._ccu._channel_keys] + \
            ['note'])

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
            self.log('Called close_output without an output file open; throwing error.')
            raise RuntimeError('Cannot close output, output has not been initialized.')
        else:
            self.log(f'Closing output file "{self.out_file}".')
            # close the output file (writes lines)
            self._out_file.close()
            self._out_file = None
            self._out_writer = None

        # grab the data
        if get_data:
            self.log(f'Retreiving data from "{self.out_file}".')
            data = pd.read_csv(self.out_file)
            # convert columns to be useful
            data.columns = \
                [f't_start', 't_end'] + \
                ['num_samp', 'samp_period'] + \
                [f'{m}' for m in self._motors] + \
                [f'C{i}' for i in range(len(self._ccu._channel_keys))] + \
                [f'C{i}_sem' for i in range(len(self._ccu._channel_keys))] + \
                ['note']
        else:
            data = None

        # reset out_file
        self.out_file = None

        return data

    # +++ methods +++

    def take_data(self, num_samp:int, samp_period:float, *keys:str, note:str="") -> 'tuple[np.ndarray, np.ndarray]':
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
        note : str, optional (default "")
            A note can be provided to be written to this row in the output table which can help you remember why you took this data.
        
        Returns
        -------
        data_avg : np.ndarray
            The average number of counts per second for each channel.
        data_unc : np.ndarray
            The standard error of the mean for each channel.
        '''
        # log the data taking
        if self.out_file is not None:
            self.log(f'Taking data; sampling {num_samp} x {samp_period} s.')
        else:
            self.log(f'Taking data; sampling {num_samp} x {samp_period} s. No output file active.')

        # check for note to log
        if note != "":
            self.log(f'\tNote: "{note}"')
        
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
                list(data_unc) + \
                [note])
        
        # return the right rates
        if len(keys) == 0:
            return data_avg, data_unc
        elif len(keys) == 1:
            k = keys[0]
            return data_avg[self._ccu._channel_keys.index(k)], data_unc[self._ccu._channel_keys.index(k)]
        else:
            out_avgs = np.array([data_avg[self._ccu._channel_keys.index(k)] for k in keys])
            out_uncs = np.array([data_unc[self._ccu._channel_keys.index(k)] for k in keys])
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
        ''' Log a note to the manager's log file.

        All notes in the file are timestamped from the manager's initialization.
        '''
        line = self.time + f'\t{note}'
        print(line)
        self._log_file.write(line + '\n')

    def configure_motors(self, **kwargs) -> List[float]:
        ''' Configure the position of multiple motors at a time

        Parameters
        ----------
        **kwargs : <NAME OF MOTOR> = <GOTO POSITION DEGREES>
            Assign each motor name that you wish to move the absolute angle to which you want it to move, in degrees.
        
        Returns
        -------
        list[float]
            The actual positions of the motors after the move, in the order provided.
        '''
        out = []
        # loop to configure each motor individually
        for motor_name, position in kwargs.items():
            out.append(self.configure_motor(motor_name, position))
        return out

    def configure_motor(self, motor:str, pos:float) -> float:
        ''' Configure a single motor using a string key.
        
        Parameters
        ----------
        motor : str
            The name of the motor, provided as a string.
        pos : float
            The target position for the motor in degrees.
        
        Returns
        -------
        float
            The actual position of the motor after the move.
        '''
        if not motor in self._motors:
            self.log(f'Unknown motor "{motor}"; throwing error.')
            raise ValueError(f'Attempted to reference unknown motor "{motor}".')
        return self.__dict__[motor].goto(pos)

    def meas_basis(self, basis:str) -> None:
        ''' Set the measurement basis for Alice and Bob's half and quarter wave plates. 
        
        Parameters
        ----------
        basis : str
            The measurement basis to set, should have length two. All options are listed in the config.
        '''
        self.log(f'Loading measurement basis "{basis}" from config file.')
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
        self.log(f'Loading state preset "{state}" from config file.')
        self.configure_motors(**self._config['state_presets'][state])

    def sweep(self, component:str, pos_min:float, pos_max:float, num_steps:int, num_samp:int, samp_period:float) -> np.ndarray:
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
        self.log(f'Sweeping {component} from {pos_min} to {pos_max} degrees in {num_steps} steps.')
        # open output
        out = []
        unc = []
        # loop to perform the sweep
        for pos in tqdm(np.linspace(pos_min, pos_max, num_steps)):
            self.configure_motors(**{component:pos})
            x, u = self.take_data(num_samp, samp_period, 'C4')
            out.append(x)
            unc.append(u)
        return np.array(out), np.array(unc)

    def full_tomography(self):

        self.new_output('Full_tomography.csv')

        NUM_SAMP = 5
        SAMP_PERIOD = 3

        for basis in self._config['full_tomography']:
            self.meas_basis(basis)
            self.take_data(NUM_SAMP, SAMP_PERIOD)


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
