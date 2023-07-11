''' manager.py

Class for managing the automated laboratory equiptment, including data collection and motor manipulation.

Author(s):
- Alec Roberson (aroberson@hmc.edu) 2023
'''

# python imports
import json
import time
import os
import datetime
import copy
from typing import Union, List, Tuple
import serial

# package imports
from tqdm import tqdm
import numpy as np
from uncertainties import unumpy as unp
from uncertainties import core as ucore
import pandas as pd

# local imports
from .monitors import CCU # , Laser # soon!
from .motor_drivers import MOTOR_DRIVERS

# manager class

class Manager:
    ''' Class for managing the automated laboratory equiptment.

    Parameters
    ----------
    config : str, optional
        The name of the configuration file to load. Defaults to 'config.json'.
    debug : bool, optional
        If True, the CCU and motors will not be initialized with the manager, and will have to be initialized later with the init_ccu and init_motors methods.
    '''
    def __init__(self, config:str='config.json', debug:bool=False):
        # get the time of initialization for file naming
        self._init_time = time.time()
        self._init_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # load configuration file
        with open(config, 'r') as f:
            self._config = json.load(f)
        
        # save all initilaization parameters
        self.config_file = config

        # initialize output file variables
        self._output_df = pd.DataFrame(columns=self.df_columns) # output dataframe

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

    # +++ class variabes +++

    MAIN_CHANNEL = 'C4' # the main coincidence counting channel key for any preset basis

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

    @property
    def df_columns(self) -> str:
        return ['start', 'stop', 'num_samp', 'samp_period'] + self._config['motors'].keys() + self._config['ccu']['channel_keys'] + ['note']

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
            channel_keys=self._config['ccu']['channel_keys'],
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

    def clear_output(self) -> None:
        ''' Clear (erase) the data in the current output data frame. '''
        self.log('Clearing output data.')
        # just create a brand new output data frame
        self._output_df = pd.DataFrame(columns=self.df_columns)

    def save_data(self, output_file:str, clear_data:bool=True) -> pd.DataFrame:
        ''' Saves the output data to a specified file in pickle format

        Parameters
        ----------
        output_file : str
            The name of the file to save the data to.
        clear_data : bool, optional
            If True, the data will be cleared after saving. Default is True.
        
        Returns
        -------
        pd.DataFrame
            The data that was saved.
        '''
        # output data frame to pickle file
        self.log(f'Saving data to "{output_file}".')
        self._output_df.to_pickle(output_file)
        
        # save copy of data to return
        out = self._output_df.copy()

        # reset out_file
        if clear_data:
            self.clear_output()
        
        return out

    # +++ methods +++

    def take_data(self, num_samp:int, samp_period:float, *keys:str, note:str="") -> Union[np.ndarray, ucore.Variable]:
        ''' Take detector data

        The data is written to the csv output table.

        Parameters
        ----------
        num_samp : int
            Number of samples to take.
        samp_period : float
            Collection time for each sample, in seconds. Note that this will be rounded to the nearest 0.1 seconds (minimum 0.1 seconds).
        *keys : str
            Any channel keys (probably CCU keys, but any are allowed) to return data for. If no keys are given, all rates will be returned.
        note : str, optional (default "")
            A note can be provided to be written to this row in the output table which can help you remember why you took this data.
        
        Returns
        -------
        numpy.ndarray
            Array of ufloats with count rates and count rate uncertianties for each specified channel.
        or
        Any
            If only one channel is specified, a single value is returned (e.g. ufloat for CCU data or str for time data or float for motor position).
        '''
        # log the data taking
        self.log(f'Taking data; sampling {num_samp} x {samp_period} s.')
        
        # check for note to log
        if note != "":
            self.log(f'\tNote: "{note}"')
        
        # record all motor positions
        motor_positions = [self.__dict__[m].pos for m in self._motors]

        # record start time
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # run trials
        ccu_data = unp.uarray(*self._ccu.acquire_data(num_samp, samp_period))
        
        # record stop time
        stop_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # record data in the dataframe
        row = pd.DataFrame(dict(zip(
            self.df_columns, [[start_time], [stop_time], [num_samp], [samp_period]] + [[p] for p in motor_positions] + [[c] for c in ccu_data] + [[note]])))
        self._output_df = self._output_df.append(row, ignore_index=True)
        
        # return the right keys
        if len(keys) == 0:
            return row.values[0]
        elif len(keys) == 1:
            k = keys[0]
            return row[k][0]
        else:
            return np.array(row[keys].values[0])

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
        self.log(f'Loading state preset "{state}" from config file -> {self._config["state_presets"][state]}.')
        self.configure_motors(**self._config['state_presets'][state])

    def sweep(self, component:str, pos_min:float, pos_max:float, num_steps:int, num_samp:int, samp_period:float) -> Tuple[np.ndarray, np.ndarray]:
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
        
        Returns
        -------
        np.ndarray
            Coincidence count rates over the sweep.
        np.ndarray
            Uncertainties in said coincidence count rates.
        '''
        self.log(f'Sweeping {component} from {pos_min} to {pos_max} degrees in {num_steps} steps.')
        # open output
        out = []
        # loop to perform the sweep
        for pos in tqdm(np.linspace(pos_min, pos_max, num_steps)):
            self.configure_motors(**{component:pos})
            x = self.take_data(num_samp, samp_period, Manager.MAIN_CHANNEL)
            out.append(x)
        return np.array(out)

    # +++ shutdown methods +++

    def shutdown(self) -> None:
        ''' Shutsdown all the motors and terminates CCU processes, closing all com ports.
        '''
        self.log('Beginning shutdown procedure.')
        # motors
        if len(self._motors) == 0:
            self.log('WARNING: No motors are active.')
        else:
            # loop to delete motors
            self.log('Deleting motor objects.')
            for motor_name in self._motors:
                del self.__dict__[motor_name]
        # com ports
        if len(self._active_ports) == 0:
            self.log('WARNING: No com ports are active.')
        else:
            # loop to shutdown ports
            self.log('Closing COM ports.')
            for port in self._active_ports.values():
                port.close()
        # output data
        if len(self._output_df) != 0:
            self.log(f'WARNING: Shutting down with {len(self._output_df)} rows of (potentially) unsaved data.')
        # CCU
        self.log('Closing CCU.')
        self._ccu.shutdown()
        # log file
        self.log('Closing log file.')
        self._log_file.close()
