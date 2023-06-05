# Manager

The `Manager` class does it all, manages start up, motors, and data collection. It also can include helpful presets (provided by the configuration file) to quickly set (say) the creation of a state, or the basis of measurement.

### (C) `Manager(out_file:str=None, raw_data_out_file:Union[str,bool]=None, config:str='config.json', debug:bool=False)`

Class for managing the automated laboratory equiptment.

**Parameters**
- `out_file : str` (optional, default `None`) The name of the output file to save the data to. If not specified, a timestamped csv will be saved. _PLEASE NOTE: You should **always** provide an output file. The fact that this is an optional parameter is just for ease of debugging. Warnings will be printed if an output file is not provided._
- `raw_data_out_file : Union[str,bool]` (optional, default `None`) The name of the output file to save the raw data to. If not specified or false, no raw data will be saved. If True, a timestamped csv will be saved.
- `config : str` (optional, default `"config.json"`) The name of the configuration file to load.
- `debug : bool` (optional, default `False`) If True, the CCU, motors, and output file will not be initialized with the class, and will have to be initialized later with the `Manager.init_ccu`, `Manager.init_motors`, and `Manager.init_output_file` methods. Defaults to False.

### (M) `Manager.init_ccu() -> None`
Performs initialization of the `CCU` alone.

### (M) `Manager.init_motors() -> None`
Performs initialization of the motors alone.

### (M) `Manager.init_output_file() -> None`
Performs initialization of the output file alone. **PLEASE NOTE: this can only be done AFTER initializing BOTH the motors and the CCU** (using `Manger.init_motors()` and `Manager.init_ccu()`).

### (M) `Manager.take_data(num_samp:int, samp_period:float) -> None`
Take detector data

The data is written to the csv output table.

**Parameters**
- `num_samp : int` Number of samples to take.
- `samp_period : float` Collection time for each sample, in seconds. Note that this will be rounded to the nearest 0.1 seconds (minimum 0.1 seconds).

### (M) `Manager.configure_motors(**kwargs) -> None`
Configure the position of multiple motors at a time

**Parameters**
- `**kwargs : <NAME OF MOTOR> = <GOTO POSITION RADIANS>` Assign each motor name that you wish to move the absolute angle to which you want it to move, in radians.

### (M) `Manager.meas_basis(basis:str) -> None`
Set the measurement basis for Alice and Bob's half and quarter wave plates. Measurement bases can be added/modified in the configuration file.
        
**Parameters**
- `basis : str` The measurement basis to set, should have length two. All options are listed in the config.

# CCU

The `CCU` class does a _ton_ of stuff, but almost all of it has to do with plotting and live data management. Beyond the constructor, theres only really two methods that you need to worry about.

### (C) `CCU(port:str, baud:int, raw_data_csv=None, **kwargs)`

Interface for the Altera DE2 FPGA CCU.

**Parameters**
- `port : str` Serial port to use (e.g. `'COM1'` on windows, or
    `'/dev/ttyUSB1'` on unix).
- `baud : int` Serial communication baud rate (19200 for the Altera DE2 FPGA CC).
- `raw_data_csv : str` Path to the raw data csv file to write to. If None, no file is written.
- `**kwargs`
    - `xlim_sec : float` Number of samples to plot on the running plots.
    - `smoothing_sec : float` Number of seconds to smooth the running plots over.

### (M) `CCU.count_rates(period:float) -> np.ndarray`

Acquires the coincidence count rates from the CCU over the specified period.

**Parameters**
- `period : float` Time to collect data for, in seconds.

**Returns**
- `np.ndarray` of size `(8,)` Coincidence count RATES from the CCU. Each element corresponds to a detector/coincidence `[A, B, A', B', C4, C5, C6, C7]` (in order).

### (M) `CCU.shutdown() -> None`

Shuts down all processes under the CCU class (including live data listening and collection). Shutdown of the subprocess also writes to and closes any raw data output file (if provided).

# Motor Classes

All specific motors (right now just `ElliptecMotor` and `ThorLabsMotor`) are derivatives of the (abstract-ish) base class `Motor`. Effectively all functional methods and properties are provided _through_ `Motor` base class, though this class should **never be initiated directly** since it is abstract (most of it's important methods are not defined). Below is the documentation for the constructors of `ElliptecMotor` and `ThorLabsMotor`, as well as the public methods which are provided by `Motor`.

### (C) `ElliptecMotor(name:str, com_port:str, address:Union[str,bytes], offset:float=0)`

Elliptec Motor class.
    
**Parameters**
`name : str` The name for the motor.
`com_port : serial.Serial` The serial port the motor is connected to.
`address : Union[str,bytes]` The address of the motor, a single charachter.
`offset : float` (optional, default `0`) The offset of the motor, in radians. In other words, when the motor returns a position of zero, where does the actual motor hardware think it is?

### (C) `ThorLabsMotor(name:str, sn:int, offset:float=0)`

ThorLabs Motor class.
    
**Parameters**
- `name : str` The name for the motor.
- `serial_num : int` The serial number of the motor.
- `offset : float` The offset of the motor, in radians. In other words, when the motor returns a position of zero, where does the actual motor hardware think it is?

### (P) `Motor.name -> str`
The name of the motor (e.g. `"B_C_HWP"`)

### (P) `Motor.type -> str`
The type of the motor (e.g. `"ThorLabs"` or `"Elliptec"`)

### (P) `Motor.offset -> float`
The offset of the motor. When `Motor.pos` returns zero, this will be the position the motor's hardware returns (in radians).

### (P) `Motor.pos -> float`
The position of the motor in radians, relative to the offset position (if applicable).

### (P) `Motor.is_active -> bool`
Returns `True` if the motor is actively moving, and `False` otherwise.

### (P) `Motor.status -> Union[str,int]`
The status of the motor. If this method returns `0`, the motor has a nominal status. Any other return value implies some kind of error or problem.

### (M) `Motor.goto(angle_radians:float) -> float`
Sets the angle of the motor in radians.

**Parameters**
- `angle_radians` : float; The angle to set the motor to, in radians.

### (M) `Motor.move(angle_radians:float) -> float`
Moves the motor by an angle **RELATIVE** to it's current position

**Parameters**
- `angle_radians : float` The angle to move the motor by, in radians.

**Returns**
- `float` The position of the motor in radians.

### (M) `Motor.zero() -> float`
Returns this motor to it's zero position.
        
**Returns**
- `float` The position of the motor in radians.

### (M) `Motor.hardware_home() -> float`
Returns this motor to it's home position **as saved in the hardware's memory**.

**Returns**
- `float` The position of the motor in radians.