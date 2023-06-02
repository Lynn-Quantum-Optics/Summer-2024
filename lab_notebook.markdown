## 6/2/23
MP: A, O, R, L
Main room was a bit warm initially but not feels like 64 which matches what we want. The teperature in the experiment room seems pretty constant; was 17.9C when I (O) arrived and is 18.0C about 40 mins later.

For reference: bottom left knob of BBO affects a vertica translation, so this will change the HH pair production; likewise adjusting the upper right will rotate horizontally, affecting the VV pair production.

C0: alice
C1: bob, H
C2: dne (would be opposite for A)
C3: bob, V

## 6/1/23
MP: A, O, R, L
N2 tank looks nominal. Fitted $sin^2$ to B_C_HWP data and found a min of 2.202 radians. We set the home offset by loading elliptec.py, creating an ellipetic motor object for the B_C_HWP and called ._set_home_offset() when it was at a location of 2.202 radians. However, we are now getting a Serial Port Failed to Open error. We did unplug the main USB so this may have reset the port labeling? We opened Ello and now cannot see either COM5 or COM7. We also checked device manager and cannot see either COM5 or COM7. We realized that COM8 has the 3 preparation; COM9 is the old COM7. For reference: to move a component, call resposition.optical_component(Orientation(Theta, Phi), component_name).

Alec wrote a streamlined code: motordrivers.py replaces elliptec.py. B_C_HWP moves now, but was not set to home using the method (before, Alec had set these manually). Home offset appears to be 0.05 radians. Alec thinks there may be a range of pi/4 radians for where you can set the home offset (which is where it thinks 0 is?). We managed to reset the home manuualy as follows:
- compute position as byte encoded hex: h = hex(int(current in hex, 16) + int(home in hex), 16)
- .com_port.write([port identity]so0000[h bits])
The error was in the number of bits sent in the function: it was sending 4, but needs to be 8. ome for B_C_HWP has been set to 2.202 radians. 


Now performing sweep for UV_HWP. full 360 degrees: first do steps = 36, samples per measurement = 5. Confirmed that creation plate works and everything works. Fitted plots for UV.

Over lunch, A testing and debugging custom control scripts.

confirmed (nonzero) ccu data is being collected
noticing descrepencies in FPGA CCU behavior. documentation says the module should update every 100ms but in reality it seems to update more on the order of < 1ms
i was forgetting to flush the buffer, now things look nominal (~100±10ms). now CCU interfacing code performs perfectly. now debugging manager class
have not run into any major issues, just a ton of tiny bugs so far


After lunch: We tried unplugging the motors from the BUS and the BUS from the USB. When we cut the USB connection, the COm port changed but the address remained the same. Also, the home remained set! Which suggests that disconnecting the USB will keep the homes.

UV: pi.
QP: 0
PCC: 0
BC: 0
TEST: -0.4

We are unpluggling the USB hub from computer and power. COM9 is now BC and TEST; addresses unchanged. COM5 are creation. The positions are as follows:

UV: pi
QP: 0
PCC: 0
BC: 0
TEST:-0.4

Thus disconnecting the USB hub had no effect on the motors other than reassigning the serial ports. So powering off the computer should in theory not be too terrible! However, if the BUSes lose power, then the home is reset. But if you know where it was before AND the home is unchanged, then we should be fine. However, when we unplugged the TEST motor it seemed like its home was reset to no seeming pattern. It seems like the mysteries of the Elliptec motors exceed those of quantum mechanics. The measurement HWP/QWPs will not move if their power supplies lose power, so in this event as long as they are at known locations the calibration should be recoverable.

Results from HWP:
max HH: 1.374
max VV: 0.588
equal: 0.201
ampltiude: 16858 vs 21619 for HH vs VV, which is a factor of 1.282.

Configuring QP: -40 to 40, 20 steps, 5 samples per step in DA basis. We ran into an error: one of the terminal windowsfor the ccu is just saying waiting for ccu log, so we are using Alec's method to run the sweep instead. We suspect it's some kind of code incompatibility. Data is collected but we do not have a thumb drive so will fit it tomorrow.



Over lunch, A testing and debugging custom control scripts.
- confirmed (nonzero) ccu data is being collected
- ~~noticing descrepencies in FPGA CCU behavior. documentation says the module should update every 100ms but in reality it seems to update more on the order of < 1ms~~ 
- i was forgetting to flush the buffer, now things look nominal (~100±10ms). now CCU interfacing code performs perfectly. now debugging manager class
- have not run into any major issues, just a ton of tiny bugs so far

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