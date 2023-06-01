## 5/31/23 eve/night sesh
MP: A
Debugged parsing of positions from Elliptec motors. Only tested code with the UVHWP. B_C_HWP offset pre-testing: 0x500.

My code can now
- encode and decode negative numbers (yay!)
- read the motor position
- all move commands now readout the actual motor position now -- so we don't have to rely on target positions we can use the actual positions
- wait for the motor to finish moving, instead of waiting an arbitrary amount of time
- recieve and warn the user about error codes such as out of range errors

Things learned
- Range of motor is +- 360º from home
- Home brings motor to ~5e-5 radians from home

B_C_HWP home offset post-testing: 0x500 (no change)

## 5/31/23
MP: A, O, R
We noticed that the N2 tank has 2250psi, whereas last night it had 2300. We will continue to monitor over the week and see if we have a leak.

Reset QP home to what appears to be 0 based on laser allignment (value is given in hex and corresponds to some number of pulses; left as a comment). 243 degrees is new home for PCC. Notch coming top down is the ideal location (since the reflection alligns with the BBO one).

Took sweep to determine optimal B_C_HWP: the data in HH appears to be pretty messy, so we infer that most of light from BBO is VV. We found using preliminary analysis and an initial sweep of 2.1 to 2.3 with 20 steps and 5 samples per step alpha = 2.21. We will fit the data to $\sin^2(\theta - c)$ to have a better estimate with uncertainty. We tried a more narrow fit, from 2.18 to 2.22 with 10 steps and 10 samples per step; however, this data was quite messy and did not prove immediately useful. For future reference, callibration sweeps can be done using a method the LabWorkstation.py (take_callibration_measurments_1d, which takes Oritentation(theta, phi) as inputs along with the number of steps, samples, and the component name. Note that this is a method that can only be called on a Workstation object.) Think about most efficient way to change basis (does it in HH.)

## 5/30/23
Members present: Alec, Oscar, Richard, Lynn
We tried to run basic_experiment.py, which should call a Bell's inequality experiment. We were unable to see any change in the coincidences that would indicate projections in differnet bases, so we began debugging. 

We confimed the big Thor motors turn on command after wiggling one of the USB cables. We discovered that on the USB hub that the Elliptec motors connect to, you must depress the buttons to the side in order to provide power. While we could not get the Elliptec motors to move on their own, we found that unpluggling and replugging the USB for the hub caused the the Bob creation HWP and the PCC motors to return to some position -- that is sometime today after we got the setup tour (and also, based on the lettering on the Bob C HWP, prof Lynn observed that it had in fact changed); however, we were unable to have these move on command.

The issue was that the Elliptec motors' ids had been all reset to 0, so we had to unplug each motor from the PCB, disconnect COMM5 using the program ELLO, then reconnect and set the addresses from A, B, C gradually, each time disconnecting and reconnecting. For more info, read documentation for Elliptec motors.