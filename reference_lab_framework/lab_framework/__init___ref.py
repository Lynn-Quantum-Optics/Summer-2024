''' Lab Framework Package

Author: Alec Roberson (aroberson@hmc.edu / alectroberson@gmail.com) 2023

This package contains the classes, functions, and modules necessary to manage the experimental setup in the laboratory. It is currently somewhat specific to our own laboratory setup, however the generalities of the code should make it very easy to extend upon for other setups.

'''

# import relevant objects
from .monitors import SerialMonitor, CCU, Laser
from .motor_drivers import Motor, ElliptecMotor, ThorLabsMotor, MOTOR_DRIVERS
from .manager import Manager
