## 7/17/23
MP: A

Started laser at 0820. Going to start by removing Bob's PBS, after a little bit more practice in the optics lab. Just removed it. It went great! I took it super slow and I don't think there were any sudden jolts that could have dislodged the PBS. I put it on the table next to the lenses for now.

### Calibrating Alice's QWP
Whipped up a quick program `calibration/AQWP.py` to do a sweep of Alice's QWP. Setting up BBO and such right now. Briefly booted Kinesis to ensure that all motors are still homed (they were). Started the program, count rates look very good (since Bob's PBS was removed we are seeing many more C4 coincidences than otherwise would for these angles ~2000).

Noticed that at UVHWP=22.5 (I think?) there was not much of a difference in detections as I rotated the measurement waveplate. Testing UVHWP=0 seemed to work okay though (significant difference in detection rates). I used 0 for the rest of the measurement waveplate testing.

Did three sweeps of Alice's QWP to narrow find the extremal value. With the UVHWP at 0 degrees, alpha is around 90 degrees and so we are looking for a minimum where the fast axis is horizontal which I found to be at -8.703564837399183+/-0.018681468778457503 = -8.704 +/- 0.018 using a quadratic fitting technique. I'm putting this value into the config file for Alice's QWP's offset for all future testing.

### Calibrating UVHWP
To find where the UVHWP is perfectly vertical we know this means we will be producing only VVs, and so we want to minimize the detection counts when we set Alice's QWP to the calibrated zero position so that we are measuring HX coincidence counts. I did a rough and a fine sweep, and using a quadratic fit I came up with an offset of -1.02928163397581+/-0.008774118772442973 = -1.029 +/- 0.009 degrees.

### Creating VV
The UVHWP angle is really all we need to create the VX state (UVHWP->0). I'm putting an entry in the config file for this. I'll use the 4.005 deg value for the PCC and 0 for the QP. I'm not putting an entry for Bob's Creation HWP yet since we still don't care about that photon.

### Calibrating Alice's HWP (~ 1100)
Now that we can make VX I'm going to put Alice's HWP back in and sweep that to minimize C4 detections which should happen when its fast axis has an angle of 0 degrees to the horizontal. Based on old config data, this should be *somewhere* around 0 degrees ish. I put the HWP in _extremely_ gently, since I had just realigned it yesterday and noticed that the post is a little bit jiggly in its holder.

I'm doing another two sweeps here, one coarse and then a fine one to establish the zero position that minimizes these count rates. I'm noticing with jus tthe coarse sweep that this position is going to be ~ -9 degrees which is a lot more than in the previous config file, however, I think we can chalk that up to the fact that those used to have the zero position at around zero on the dial, and the new (hardware) offset (i.e. difference between Kinesis reading and dial reading) for all measurement motors is now ~ -9 degees.

Indeed, the second sweep indicates that the offset angle that minimizes counts for this motor is -9.470004030055916+/-0.01174697291490038 = -9.470 +/- 0.012. I'm now putting this offset into the config file for A_HWP.



## 7/16/23
MP: A

Started laset at 1055. Kinda sad about all these motor problems! Going to first try restarting all ThorLabs motors. 

## Basic Setup Stuff
- Starting by launching ThorLabs APT user and setting all positions to zero.
- Alice's measurement waveplates do appear to have maintained their zero position despite the crash yesterday.
- Closing ThorLabs APT and unplugging and re-plugging all ThorLabs motor power units.
    - APT User crashed while doing this :( But I relaunched it to double check that the crash didn't affect anything then I closed it correctly.
    - So all motors read 0+/-0.0010 (deg) in ThorLabs APT when I unplugged them.
    - While doing the restart I'm downloading and installing the newer ThorLabs Kinesis software for motor control (the replacement for ThorLabs APT User) this will require a reboot which I'll do after confirming motor zeroing after restart using the old APT User
    - First unplugged all PC connections, then power supplies, then re-plugged in power supplies then PC connections.
- Only 3 Motors appeared in device manager after the reboot... double checking PC connections
    - Slight jiggling of all the MicroUSB ports to the control units made the fourth show up. Don't know which one didn't show up originally, but they are all here now.
- Launched APT User... only three showed up! Missing SN ending in 904 (Alice's QWP)
    - Rebooting device manager.. took so many tries! It crashed at least 3 times.
    - Also noticed APT User didn't entire close out in task manager ahg. Ended those tasks restarting APT user now.
    - Still missing 904. Restarting just that motor and re-connecting it now.
    - Restarting APT user then only showed 2 motors??? But found some APT user tasks, ended them, and restarted again.
    - **FINALLY SEEING ALL FOUR MOTORS AGAIN**! All perfectly zeroed (no more +/- 0.001) so indeed this does reset the zero position of the motors

## Kinesis Software
- Rebooting this PC now. Took a while but went A-okay. Finishing Kinesis install now.
- Booted up Kinesis and it was so much more informative
    - Every motor hit "Motor Error Detected: (Value - Position Error(0)) (Value - Position Error(0))"
    - Every motor also hit "Unable to resolve Issue ("Internal serial number is incorrect")"
    - The program forced me to do some default settings stuff for each motor, so I'm going to reboot Kinesis to see if the errors persist.
- Rebooted Kinesis, errors persisted and it was missing Alice's QWP (SN...904)
    - Re-pluged in both sides of the micro USB cable, and rebooted again. Errors persisted but Alice's QWP showed up.
    - Note all except Alice's QWP (SN.904) are "Homed" in Kinesis.
- Trying a few things to resolve errors
    - Without restarting Kinesis, closed and re-connected to all motors. Didn't fix it.
    - Changed all motors to use device persistent settings on start up. Then closed and re-connected to all the motors (again without closing Kinesis). Didn't fix errors. At this point now SN.901 and SN.667 are "not homed" according to kinesis (the other two are?). All motors still read 0 position here.
    - Closing Kinesis as I work on this.
- Googling around it would seem our setup is really _really_ old and support is quite lacking. Going to try a firmware update but I don't know if that will help our specific problem. Tried the Kinesis Firmware Update Utility but none of the firmware on these controllers is programmable. Ah!
- Tried booting Kinesis with "ignore device startup settings." This did not fix any errors but now all motors say that they are "homed".
- Rebooting without this option shows that all except SN.904 are "Homed"
- Going to try not to think about the errors anymore.

## A little more messing around with Kinesis
- Uninstalled Kinesis 1.14.37
- Installing Kinesis 1.14.36
- Nothing fixed
- Realizing Kinesis' lack of actuators is actually quite a problem here.

## Notes for an ideal world
I think the best case scenario here is we replace each of the TDC001 controllers with TDC101 controllers that have updated firmware/drivers that will almost certainly work better with everything else in the lab.

## Trying to get things workin in python
- Manually using `thorlabs_apt.Motor` worked to modify the position of Bob's HWP.
- The Elliptec ports seem to have gotten goofed up (likely due to the restarts). Running Ello to figure it out. `B_C_HWP` and `B_C_QWP` have moved to `COM7`.
- Basic commands seem to be working.
- If anything **FIXED** the issues that we have 

## Basic motor testing
- Bob's QWP
    - Startup position -97.08 consistent with offset. Dial reads 9ish degrees.
    - `goto(0)` sent it to 106ish degrees.
    - `hardware_home()` returned dial to 9ish.
    - **OKAY** 
- Bob's HWP
    - Init pos 6.12, dial at 37.somethingish.
    - `goto(0)` sent to 31ish
    - `hardware_home()` sent to 37ish
    - **OKAY**
- Alice QWP
    - Init position -0.45, dial at 0.3ish degrees.
    - `goto(20)` caused a crash! "Exception: Getting status failed: Time out while waiting for hardware unit to respond."
    - The motor moved! After the crash the dial reads 3.2ish degrees.
- During shutdown, all Motors except A_QWP successfully zeroed (however they were all already at 0 so not sure if that means they all *work* work).
- Exiting python, booting APT User.
    - Christ. ThorLabs APT now shows all motors at 0 position, even Alice's QWP that moved from it's zero position before the crash. This means that **Alice's measurement waveplates are almost certainly not aligned anymore.**
- I'm confused why this one crashed and the others didn't.
    - I notice that in Kinesis that is now the only motor giving the "Motor Error Detected: (Value - Position Error(0)) (Value - Position Error(0))" error now! Fascinating!!! (Note: all of the motors are still giving the serial number error.)
    - I don't know why... hmmmmmmm!
- Restarting Alice's QWP (SN.904) since it has already lost it's home and is now the only one causing problems.
- I tried moving the USB for Alice's QWP to the USB hub on the table instead of the PC port it was using.
- Booting Kinesis again to checkout error logs. 904 is still giving the position error.
- Tried moving it using ThorLabs APT. No motion, instant USB communications error. Recommended restart. Trying that now. This time I'm restarting it without unplugging the USB.
- Booting Kinesis. The motor didn't even appear! Accidentally booted User APT and it did appear. Trying to move it crashed at 0.0031 degrees, showing the same FT_IO_ERROR.
- Attempting to reconnect the cable to the same port on the computer that it was using before.
- Booting Kinesis shows the same position error.
- Trying to move the motor in python resulted in the same "internal error"
- Launching Kinesis real quick to make sure nothing changed, then I'm going to try a careful system reboot. Yep nothing changed, same position error only on SN.904.

## Updating Thorlabs APT?
- version was like 1.0.?? really really old newest version is 3.21.6
- a bit nervous because I was expecting it to ask me if i wanted to uninstall the old one and my plan was to figure out how to maintain both of them but it never asked it just automatically uninstalled it. hopefully this works i guess!!!
- It works exactly the same. Same error and everything lol.

## Whole system reboot
- Rebooting computer and all ThorLabs motors.
- Fixed nothing, now different motors are throwing errors. I'm going to take a break.

## Returning at 1952
- Status: SN904 not connecting SN667, SN646, and SN901 are all throwing SN error and position errors. Agh.
- Attempting manual homing of SN667. This actually did get rid of the position error!!!
- Disconnecting SN646, and 901, then manually homing those and 904 and reconnecting.
    - Manual homning on 904 seemed to just cause power cycles and on 901 the active light is just continuously lit.
    - Reconnecting to 646 resolved position error.
    - 904 still failed to load.
    - 901 still has position error.
- Rebooted 901, still had position error. Homed it in Kinesis, then disconnected and reconnected and it resolved the position error.
- Manually disconnected, rebooted, and reconnected SN904. Now connection is successful but I am seeing device not responding.
- Restarted Kinesis. Problem persisted.
- Unplugged 904 motor from servo and plugged it back in. While doing this, 667 randomly started having issues i.e. the position error (the 904 servo wasn't even connected to the computer or to power! Agh.)
- Manually homed 667, didn't resolve. I closed Kinesis, gave the velocity knob a little flick up then down, and it seemed to get better? Restarted kinesis and yep! No more position error.
- Back to 904. Powered back up, reconnected to PC. Attempted to connect in Kinesis and it worked!! The connection. We still have the position error. I tried hitting home in kinesis and it seemed to power cycle? Threw a ton of errors and had to reconnect. Position error persisted. Every single thing I do to this motor turns it off and on again. Touching either jog button or the velocity knob, even when the computer is not connected.
- Launching APT user to see if it has any help here. It just threw a ton of errors no matter what I touched so I'm going back to kinesis.

## Deciding that SN904 is totally busted
I grabbed SN83828234-I2-M0 from optics lab. I want to try using it with AQWP motor but first I want to make sure that I won't break it. So first I'm going to disconnect AHWP and AQWP from 667 904 respectively and try conecting AQWP to 667 to see if that can establish control over the motor. If that doesn't break the 667 control unit, then I will try swapping in the one I stole from optics lab.
- Connected AQWP to 667. Booted Kinesis, and it is so much more operational, BUT! It refuses to home WITHOUT throwing the position error. GAHHHHHH! I'm  debugging...
- I literally just flicked the velocity knob up/down a few times and the active light STOPPED blinking! (It had been blinking continuously before this).
- It worked! I was able to home it and not get the position error.
- I'm going to really quick try connecting AHWP to 904 to see if that works? Loaded it into Kinesis and got the position error. Tried to home and it WORKED!?!? It homed and everything didn't throw the error! INSANE.
- Didn't need to use the one from optics lab! Woot woot. I'm going to try rebooting kinesis and see what happens?
- I'm going to try powering those two off and swapping back to see what happens! Weeeeee
- Both of them showed up with the position error. Homing 667 worked but homing 904 halted communication and a second attempt caused a power cycle. Swapping back to run some more tests.

## Everything is Working?
- Swapped back and reconnected in Kinesis. Both initialized with position errors, but Homing was able to resolve the errors for both.
- Booting up ipython to try interfacing with `thorlabs_apt`
- AQWP starts at position 0, dial reading 8.8ish; sent to 20 and dial read 28.8; sent back; **OKAY**
- AHWP starts at position 0, dial reading 8.8ish; sent to 20 and dial read 28.8; sent back; **OKAY**
- sent AHWP to 180 to test `is_in_motion` -> all good!
- BHWP and QWP work as well. Zero is around 9ish degrees...hmmm
- So that all works!!!

## Fixing zero offset? But actually messing everything up instead.
- In above testing, all home positions were at around 9 degrees. I'm wondering if we can make that be zero.
- I'm going to mess with SN901 (BHWP) since why not. It's been fine overall. I went into ThorLabs User APT and changed the Home offset from 4 to 0. Unfortunately, the User APT home routine just spins it in circles whoops!
- Booted Kinesis and 904 has the goddamn position eror again! I'm trying to home 901 and it failed. Moved to some random angle and just sat there with homing blinking. 904 and 901 are raising the position error still.
- Rebooted 901 and 904. 904 showed up in Kinesis with position error that was able to be fixed. 901 had to be initialized manually and also raised the position error, however homing caused a power cycle.
- Didn't even reboot 901, just connected and disconnected via Kinesis and was able to home it and resolve the error.

## I'M NOT MESSING WITH THE MOTOR SETUP ANYMORE
So here is the situation: Alice's HWP and QWP swapped motor drivers. The errors are consistently buggy in both Kinesis and User APT but at least kinesis seems to be able to home things consistently.

### Testing Kinesis Homing!
- at home, Alice's QWP dial reads ~8.5 degrees
- in ipython i moved Alice's QWP (now 667) to 90 degrees. The dial now reads 98.5ish
- i quit ipython, and reboooted the motor controller
- booted up kinesis, where the motor indeed now shows position 0 and that it is not homed.
- homed the motor in kinesis and the dial is now at 8.5 ish degrees!! kinesis homing works!!! **AHHHH THIS IS HUGE**

## 7/16/23 continued: RECALIBRATING EVERYTHING

- _NOTE: While taking photos for documentation, I bumped Alice's QWP a bit. I would like to check it's alignment later._
- Calibrating A's QWP is actually the first step, so I'm going to check the alignment of that right now. Went just fine. The mount was actually super loose and theres no vertical control but i got it retroreflecting as much as possible. Then I tightened the HELL out of the mounting. While I did that, I also went ahead and also made sure that Alice's HWP on the magnetic mount was also retro-reflecting as much as possible.
- Then I got everything set up and checked the count rates and they were incredible!!! On the order of 3000 which is perfect, exactly as they used to be.

## Calibrating AQWP and Creating HH
- I have already removed Alice and Bob's HWPs, I left Bob's creation HWP in because it doesn't actually matter right now since...
- The first step of the procedure that I came up with is the scary removal of Bob's PBS. The time has come. But not today. It is midnight. Signing off <3

## 7/15/23
MP: A

Started warming the laser at 9:48, although I don't think there will be any issues with that since the first thing I want to do is produce only HH states. Much of today's lab notes will be encapsulated by the calibration routine latex document I will be writing as I go. Looking back at data/drift_diagnosis_min_VV we see that the VV counts are minimized at the (effectively) exactly the same angle when the setup was cold or warm, and so indeed warming up the laser doesn't actually matter this morning, so we should be able to get going sooner than expected.

I was kind of hoping for the measurement QWPs to be on magnetic mounts as well so that we could fully re-calibrate **everything** but I can still make it work I think.

2156: after slacking with Lynn we are going to try *not* to re-calibrate **everything** by first taking some data for a well-known state.

Well first things first i re-pip installed the lab framework (now on version 1.0.7) to incorperate motor homing into the shutdown procedure.

Noticed nitrogen flow was kinda strong so I turned it down a bit, though it still feels substantial. Also noticed the placement of the nitrogen lines was quite iffy so i got them nice and close.

Okay the experiment failed immediately trying to move Bob's QWP. I shut everything down and I'm going to come back tomorrow.

## 7/14/23
MP: R, (later) A

Came to warm up laser at 9:33 am. Started running experiment at 10:40 am. I reran the alpha = 45, beta = 18 degrees point and obtained the new data. I then tried rerunning the alpha = 60 degrees case, but an error appeared early on in the experiment. The error was "Exception: Getting status failed: Time out while waiting for hardware unit to respond." I am now going to retry running the code at 2:41 pm, but I am unsure whether the same error will appear again. After rerunning, the same error appeared immediately. I will now try exiting VSCode and running Ello, then coming back to see if the error fixes itself. Running Ello did not fix the error.

A here to debug these motors let's check it out. Ahh hmm so first weird thing is that this is a problem with the ThorLabs motors, not the Elliptec Motors. This is not looking like a normal error that we've seen before. Problem seems to be with Bob's measurement HWP and is two-fold: (1) the checking if the motor is moving every 0.1s causes something in the motor to fail and (2) once failed, the motor doesn't want to come back to life.

I tried booting User APT while running the Manager and only Bob's HWP showed up - as though it was not in touch with the Manager at all! Furthermore every software that pings the motors position is returning 6.12. B_HWP's offset is -6.12, so this means that the motor hardware believes it is in the zero position. Looking at the dial it appears to be at ~41 degrees. This is not good, because reconstructing Richard's code indicates that the motor should have been set to a true position of 196.337 degrees. There is nothing special about the 41 degree number, and so it would seem to me (especially from the stack-trace of the error message I saw) that the Motor failed mid-move, giving us no way of knowing it's true position or how to get it back to zero. Huge yikes!

I found some documentation on returning motors to the manufacturer's home position, but I'm going to try one more thing (moving the motor 10 degrees) before doing anything with that. Moving the motor and sending commands seems to work, but given the loss of home position i tried the manufacturer's homing sequence which simply did not work :( the program crashed once more, losing the home position for Bob's measurement QWP as well :(. After this I gave up trying to fix things, but the good news is that control over the thorlabs motors appears to have been re-established.

## 7/13/23
MP: A, R

A here, removed bob's creation QWP and started warming up the laser at 8:43. Desk thermometer reads 18.8C.

R: I updated interesting_state_sweep to run with the new framework, but added the old analysis file to the folder for convenience since my code heavily depends on that formatting. I also edited full_tomo by creating a new file called full_tomo_updated_richard.py which adds a temporary fix to the methods in full_tomo by converting the ufloats from take_data back into floats and uncs. I obtained new data for alpha = 45, beta = 0, and the purity and fidelity are looking a lot higher at ~98-99%. The angle is also 40.98 degrees, which is likely the shift of 4 degrees that occurred. I'm now running alpha = 60 degrees.

## 7/12/13
MP: A, O

A here coming to test out the new code framework with uncertainties and such! I've started by installing the lab_framework package from the new repository I setup this morning and now I'm going to get the lab setup to run a few tests with it :)

Set everything up at 12:12. Didn't warm up laser because this data doesn't matter. I double checked everything
- made sure nitrogen flow was minimal but present. came up with another way of checking this -- cover the ends of the tubes for ~1s and listen for a hiss when you let go. this is much easier than feeling for the breeze when the room is already chilly.
- photodetector small power supply was like 1.5V (should be 2V?) and 0.7ish A (should be 1A?)
- laser themometer at 19.4C desk thermometer at 19C

After testing and verifying the new lab_framework, Lynn came down to help with a preliminary mounting setup for Bob's creation QWP. It went well! But we are missing a part so it is on a temporary mount for now, good enough for us to try out some calibration scripts but a far cry from the final mounting.

## 7/11/23 
MP: O

Temp in lab looks good: thermostat says 65.3F, in the curtained portion the original thermometer reads 18.8C, and the new one we got says 19.7C. Laser started warming up at 10:38. Plan is to retake HH/VV sweep wrt QP but for 20 x 1s; will then retry making E0 states with negative angles (and correct witness uncertainties). Replaced the N2 tank and got another one in position. For reference, there are 2 full tanks in Breznay lab we can use.

At 12:21, started taking QP data. Inside curtain thermometer reads 19.0 C, in main room 66.5F. Sweep finished at 13:05.

At 13:32, trial 29: running E0(60, 36) using improved HH, VV sweep to test calculations. 91.8% fidelity; Problem is magnitude of off-diagonal imaginary components. Just realized inconsistency in value of QP rot (neg or pos); fixing.

Trial 30: implemented phi func fix. Magnitude of diagonal term quite off; going to original VV / HH, and also fixed sign on phi.

Trial 31: testing w negative phi, original BBO correction: fidelity 98.6%!! Off diasonga mags for both real and im look much better, like .205 vs .25 in imaginary, 0.857 vs 0.85.

Trial 32: running full sweep, eta = 45, 60. Note: we were moving the couch during this time, so after this run finishes I will probably redo it just to be safe.

Trial 33: rerunning sweep eta = 45, 30 (30 since Richard did 30 and 45 again bc of couch disturbance; also PhiP and PsiP for the record. 19:26--There was an error reading in the eta = 30 setting, but got PhiP and PsiP and eta=45. Turning off N2 but keeping laser on and will come back in a few hours to rerun eta=30.

At 23:10, got back to lab and it's feeling a little on the warm side: thermostat says 68.5, thermometer 19.9C. Trial 34: eta = 30 data. Everything off at 12:43.

## 7/10/23
MP: R

Started warming up laser at 10:30. I changed the beta sweep from 0 to pi/2 to .001 to pi/2 due to a division by zero error which I will try to fix later. Importing from rho_methods also seems to be having issues. The interesting_state_sweep file ran without issue until the end of the first iteration where I forgot to close the output. Fixed the issue then ran it again, and it succeeded in running fully through. I will process the data tomorrow.

## 7/7/23
MP: O
Started warming up laser at 20:15. Plan is to re-take data for E0; eta fixed at 45, 60, and 22.5 degrees, sweeping chi from 0 to 90; change is using VV / (VV+HH) and HH / (VV + HH) in BBO and sweeping negative QP angles. At 22:44, started taking data using 'ertias_2_fita2.csv' in oscar/machine_learning/decomp.

Trial 27: fix eta = 45, 60, 22.5 and iterate chi over 0 to 90 in 6 steps like before, but this time I've changed the calculations based on last night's fit to use negative values of QP rot. Realized making states with -chi again, since I still returned -phi in QP function in InstaQ.

Trial 38: retrying with new calculations. For some reason, eta = 60 degrees results in low (~80% fidelity), which is coming from off-diagonal components: seems like imaginary components are close in mag and have correct sign, but real parts have same issue as trial 26 in terms of magnitude but also have incorrect sign. Same with eta = 22.5 degrees: magnitudes close-ish, but wrong sign on real part; magnitudes actually quite close! 0.703 vs 0.707.

## 7/6/23
MP: O
Started warming up laser at 17:52. At 22:28 started collecting data; running QP angle vs phi from -45 to 0; will use to inform QP rot sweep of HH and VV. 

Seems like  sweeping -45 to 0 reveals there are several regions which give us full 0 - 360 coverage, so running HH VV sweep wrt HH, HV, VH, VV basis over -45 to 0. I'm thinking we could sweep over negative angles for QP in InstaQ since it's more compact, so should have higher count rates; did 5x1 measurement. Problem is 

Getting occasional error in sweep: 
    raise RuntimeError(f'Sent instruction "{self._addr+inst+data}" to {self} expecting response length {require_resp_len} but got response {resp} (length={len(resp)})')
    RuntimeError: Sent instruction "b'Bma0001FB24'" to ElliptecMotor-C_QP expecting response length 11 but got response b'BGS0A' (length=5)
Just CNTRL-C + reload seemed to work. Also found that instatiating new manager obj for each configuration of state made this issue go away.

After plotting, max HH and VV in HH and VV basis realized low counts; problem was didn't set measurement basis correctly; 3000 counts now for HH in HH, yay! Accidentally had the wrong sweep angles, so had to run again :(

## 7/5/23
MP: A
Started warming up the laser at 0823.

Alec here: I just came back in to do my test. I don't know who completed the rest of the setup today but I am noticing the nitrogen tank is _critically_ low (~300psi). Talked to BJ and we can just swap it with the spare we have in here, and he will come in sometime and just swap the now empty one for another spare.

**NOTE**: we should make a sign for when we are running experiments because we don't know when BJ will be coming by in the next few days and don't want him to interrupt anything.

1304 A: ran my (corrected) sweep for checking correlation between HH/VV and HV/VH.
1316 A: Powered down detectors, blocked laser, packed up BBO and everything. Getting ready to swap N2 tanks.
1319 A: Closed tank, purged the regulator.
1322 A: All went well, now going to swap the positions of the full and empty tanks so that the regulator can reach.

Borrowed some hands from Prof. Brake's lab to hold the N2 tanks and help swap the regulators. Was fun and I think they liked it :) All went well and everything is back to operational. Shutting down the laser now (1342) for the day.

--
At 17:43, O started warming up laser to take data on HH and VV states sweeping the QP from 0 to 38.3 degrees. Started taking data at 18:45. Fitting VV / HH (QP rot). Plot showinf VV / HH > 1 ?? Thought HH exceeds VV counts... also a bit noisey, so retakijng 10  x 1 s sweep. 

Got fit, now PhiP UV_HWP looks be at 24 degrees. Trial 24: PhiP with
    23.94779209337987,38.299,0
yielded 96.8% fidelity, 95.1% purity, a .4% improvement from my previous results and .7% improvement from preset for fidelity! 

Trial 25: PsiP:
    24.495671033003333,26.10255916324198,45.0
96.3% fidelity, 95.7% purity.

Trial 26: eta = 45, 60 and chi from 0 to 90 (with fit func for VV / HH)
    E0 (45, 0): 45.0,18.369888570153112,45.0,0.9999999999999998
98.5% fidelity, 98.2% purity.
    E0 (45, 18): 40.61574916425849,17.811568077191748,44.999997995332805
99.0% fidelty, 98.7% purity!
    E0 (45, 36): 37.01747029255738,17.740156375689754,45.0
99.3% fidelity, 98.8% purity!!
    E0 (45, 54): 32.83419140697312,18.212276187711478,44.76148960325736
98.4% fidelity, 97.5% purity.
    E0 (45, 72): 27.841030629344782,17.922416212538486,44.71126751213072
97.5% fidelity, 95.7% purity.
    E0 (45, 90): 23.884281132684468,18.09415493858101,44.52700103522797
96.6% fidelity, 94.7% purity.
    E0 (60, 0): 39.03263879532898,38.01428458035744,44.519056793395045
97.8% fidelity, 98.7% purity.
    E0 (60, 18): 36.93625499084846,9.704567434793242,44.999999603208025
98.4% fidelity, 98.8% purity.
    E0 (60, 36): 34.22465116816317,12.755510373909463,44.494589981153
97.9% fidelity, 97.8% purity.
    E0 (60, 54): 31.104191953808133,13.467682329830504,44.38291758457631
98.2% fidelity, 97.8% purity.
    E0 (60, 72): 27.352578800083396,14.249820021812848,44.95401383269358
97.0% fidelity, 95.5% purity.
    E0 (60, 90): 23.916491360652447,14.340308744299092,44.9348855047236
96.9% fidelity, 95.3% purity.

Laser shutoff at 02:46.
## 7/4/23
MP: O
Started laser at 11:45. Realzed that trial 17's PhiM was really PhiP, which the updated fit agrees with! (had wrong minus sign in HWP matrix). @ 12:52, starting data. Trial 19: running PhiP and PsiP, since flipping the B_HWP by 45 will give us PsiM and PhiM:
    PhiP: 21.220354099848972,38.299,0.0009163323381782993
96.4% fidelity!! 
    PsiP: 21.337566181446963,25.951366671457016,45.0
96.8% fidelity!!
Trial 20: 6 sample E0 states: cos eta PsiP + e^(i chi)*sin eta PsiM
    45, 0: 45.0,13.107759739471968,45.0
98.9% fidelity!! 98.3% purity.
    45, 18: 40.325617881787,32.45243475604995,45.0
95.8% fidelity, 98.1% purity.
    45, 36: 35.319692011068646,32.80847131578413,45.0
73.9% fidelity, 98.1% purity. 2nd row, 3rd element off by cc.
    45, 54: 29.99386625322187,32.59712114540248,45.0
40.7% fidelity, 97.1% purity
    45, 72: 26.353505137451158,32.91656908476468,44.71253931908844
16.0% fidelity, 96.0% purity.
    45, 90: 20.765759133476752,32.763298596034836,45.0
2.3% fidelity, 94.5% purity? 
In final 3, Consistent decrease in purity as well as fidelity; rerunning 45,0 to rule out extraneously factors in the lab.

Trial 21: E0 45, 0 yields 98.8% fidelity, 98.3% fidelity, E0 45, 18 yields 95.4% fidelity, 98.2% purity (same settings as before) 

Will try a different eta value--pi/3. Trial 22: 
    E0 60, 0: 36.80717351236577,38.298986094951985,45.0
98.5% fidelity, 97.7% purity.
    E0 60, 18: 35.64037134135345,36.377936778443754,44.99999
91.9% fidelity, 96.9% purity. 2nd row, 3rd elem off by complex conjugate.
    E0 60, 36: 32.421520781235735,35.46619180422062,44.99998
75.2% fidelity, 97.2% purity. 3rd row, 2nd elem off by cc.
    E0 50, 54: 28.842682522467676,34.97796909446873,44.61235
51.4% fidelity, 94.6% purity. 2nd row, 3rd elem off by cc.
    E0 60, 72: 25.8177216842833,34.72228985431089,44.74163766
33.5% fidelity, 95.6% purity.
    E0 60, 90: 21.614459228879422,34.622127766985436,44.9666 
23.4% fidelity, 95.4% purity.

Trial 23: flipping sign in chi calculation; trying 60, -90. Agrees theoretically with the experimental matrix for E0 60, 90. Will recheck fidelity calculations but for the minus states.Taking data on the E0 60, -90 state to try to match E0 60, 90 theory.
    21.201565428724564,14.15176549160343,44.802745028367205
96.1% fidelity to theory E0 60, 90!!

Checked the experimental E0 45, 18 to theory E0 45, -18, and got 96.7% fidelity! Will track down the minus sign.


## 7/3/23
MP: O
Started laser at 2:45. Had talk with Prof. Lynn earlier and removed PCC from calculations, removed phase from BBO calc, using Alec's QP vs phi data fitted to a curve in QP calculations. Setting PCC to 4.005 degrees as per the phi_plus.

Started taking data at 4:42.

Trial 15: PsiP based on changes described above.
    20.8108889219335,17.696653112243197,44.6587696163696
41.2% fidelity, 94.7% purity... missing the element off main diagonal in middle...UV_HWP angle looks good, B_HWP looks good. phi corresponds to:  
Trial 16: PhiP.
    21.109626323605763,8.798403700895438,2.130978125874551e-08
94.7% fidelity, 94.7% purity. Yayy!! But why is PsiP so bad?
Trial 17: PhiM, PsiM.
    20.83285827774238,38.1826344182438,0.5216514440098672
    20.894233176313282,5.1385597616734895,44.99807621169933
PhiM: matrix looks good EXCEPT missing off diagonal signs... so like 0 fidelity. 38.138 deg rot corresponds to phi = 0!! So it's making PhiP. PsiM: 96.5% fidelity!! 38.182 degrees rotation is 0 phi. Plotting my get_phi function on lab computer reveals it blows up? Issue was didn't convert back to radians for sec..

---
Retrying after fixing the get_phi function. Trial 18: PhiM, PsiP, PhiP,PsiM
    21.515402512198687,27.518074317391733,0.22015241973423158
97% fidelity.
    21.232646569572733,25.57333802946146,44.99994134071441
95% fidelity.
    21.20077993918437,19.216339991442524,0.00012493912175692444
47.7% fidelity -- missing off diagonal entries... problem was missing more deg to rads in get_phi.

## 7/2/23
MP: O
Started laser at 11:58AM. N2 tank looks ~100 PSI lower than yesterday (800 currently). Started full tomo at 1:02. 

Voltage: 0.74 at noon, 0.77 at 3:15.

Added PCC rotation and BBO angle to calculations. Trying
    30.36798114838227,-18.733916497315626,325.0448358763723,0.23200941293187563
UVHWP, QP, PCC, B_HWP
Got 59.9% fidelity. Rerunning preset PhiP for sanity check: got 96.3%. Changed theoretical angle of BBO to match Ivy's; retying tomo with 
    42.00970401137467,16.104299813987264,153.07306905699525,0.33805272547956783
77.5% fidelity. Trying calculations ignoring BBO angle but with correct VV/HH value
    11.97123887473845,2.291134930229936,10.15900765763413,4.124816734946772
86.2% fidelity! Trial 7: Adding 4% random noise in form of Werner state (still w/o BBo angle):
    28.94069719483748,-38.56999665032338,51.27978279691683,0.55709529393374
87.0% fidelity. Trial 8: adding back BBO angle into calculations (15.67 deg), keeping random noise:
    34.27962952182128,-8.056738901545698,162.16280197276214,7.406772698815368e-06
84.6%. Since best trial to date used completely theoretical BBO, trying that with random noise. Theoretical result with completely theoreitcal BBO and preset PhiP is 95.6%. Trial 9 on PhiP, with theoretical BBO and 4% random noise and .967 threshold (actual state has theoretical fidelity 96.9%):
    6.7410391723475565,-34.40483411502409,198.43489846691298,3.09261081570572e-06
84.6%. Trial 10: same settings as trial 9, but trying PsiM:
    33.074939030622865,-22.297722896570058,56.764877657872944,45.0
46.6%. Trial 11: PsiM  with no noise, theoretical BBO:
    29.930457985054893,-21.873246542663157,52.00229138327332,43.77360595684996
42.6%. Problem is instead of like .5, -.5 in middle, it's .668, 0.068. Trial 12: PsiP with no noise, 15.67deg BBO in calculations.
    2.226563168917415e-07,-37.549459873637595,248.13895611602422,43.84182739949773
51.2%. Trial 13: no noise, theoretical BBO on PsiP:
    2.7179454889665373,-34.12724934711968,244.75312445410287,42.42870169562849
To be sure, plugged theoeritcal settings back into program and got sensisble matrices. And radians correctly converted to degrees. 54.5%. Measured rho looks very sparse: maybe not enough light -> restrict range. But still earlier trials with less extreme QP angles and poor counts. Trial 14: noise, theoretical BBO on PsiP:
    6.71300323380349,26.386607556725494,340.0389824890846,44.97247940950898
79.0%.
Also, full tomo runs in about 10 mins for 5x1.

## 7/1/23
MP: O
Started laser warmup at 1:44. Going to test my non-idealized Jones decomp method on several states, including PhiP, PsiM, and then a selection of E0 states depending on results.

Voltage reading: 0.75V.

3:04--Making PhiP using decomp settings, decomp_exp.py. 
25.744714514968123 -23.54387908592889 0.2789953298532202 (UV_HWP, QP, B_C_HWP)

Issue: live plots not loading and "PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: './mlog.txt'". I used the function .configure_motors(). Tried running full_tomo.py, which worked. Tried closing terminal sesion and restarting, but same WinError 32. Says "loading measurment basis "HH" from config file. Taking data; saimpling 5x1 s. Note: "HH"". Trying replacing with individual move functions--didn't work. Tried just calling .make_state('phi_plus'), but same permissions error. Got it to work! The issue was I didn't put my code within a if  __name__ == '__main__' block.

Actually started taking tomo data at 3:56. Signs are correct! Magnitude is off: 0.62 and 0.373. Fidelity 86.8%. Trying again with ideal BBO. Can correction given in Ivy's thesis, but this will involve solving for the angle of the BBO. Actually amde PhiM, but this is because I named the state wrong in jones.py. Better than last time -- fidelity 89.1%, magnitudes still a bit off. Could implement gradient descent tuning in experiment setup! For ref, tomo took about 20 mins. Actually not -- would take too much time. But could potential integrate some kind of adaptive feedback. Trying substituting s0 = [[1/a], [0]]:
22.73996272574533,32.895070033699014,0.025591125837580483
Went down to 44%. Normal cleanup.

## 6/30/23
MP: A

Started laser warming up at 11:18. Going to get some lunch and come back to run my sweep tests in the afternoon.

This is Richard. I came in to do a full tomography at 12:30. I noticed that the current on the 2V power source seems to be below the 1A it says on the sticky note. --> A: How much lower???

Alec again, powered off detectors now using the voltmeter to check out the power supply voltages.
- left 1.92 -> 2.00 V
- right top 30.0 V
- right bottom 4.94 -> 5.00 V

Powered up the setup again and saw 1.5 A drop to 0.75 A 

Normal cleanup at 1530.

## 6/29/23
MP: A

Started the laser warming up at 0827. Returning now at 1037 to finish setting up.

Starting today with a renewed and revised quartz plate sweep! The improvement in speed to the quartz plate sweep script cannot be overstated. With sampling at 3x0.5s old script took ~30 min, the new one is more like 10 min. I did a quick sweep then spent a while fuffing with the data until I ultimately messed it up beyond repair. Now I'm going to run a more precise sweep with sampling at 5x1s.

Performed sweeps for all of the components that I wanted to today! Yay! Cleanup went normally :)

## 6/28/23
MP: A

Started laser warming up at 0934. My state measurement script is working _swimmingly_ now! I loaded up the calibrated "phi_plus" state and recorded the values

- $\alpha=44.55\pm 0.26^\circ$
- $\beta=5.77 \pm 0.24^\circ$
- $\phi = 10.75\pm 0.85^\circ$

Pretty close to phi+ if I do say so myself! Super normal day otherwise. Left everything setup all day, taking it down right now (1530).

AHHH! In my hurry to be on time I accidentally turned on the room lights for ~30 seconds before turning off the detectors so I had to boot up the CCU monitor again. Nothing is broken it looks like! THANK GOD!

## 6/27/23
MP: A, R

Laser warmed up this morning. So I am assuming that it's already warmed up lol. Not that it's super important. I'm just going to be testing my measurement script :) 

Just had so much trouble with the CCU. No matter what I did it just appeared to be sending blank info. Turned it off and on, checked the COM port, nothing! Then I just unplugged and plugged back in the USB and boom. Works again.

Measurement script looks great! The previously "calibrated" phi_plus state came in with
- $\alpha=45.70\pm 0.21^\circ$
- $\beta=5.39\pm 0.21^\circ$
- $\phi=10.10\pm ?^\circ$ (Uncertainty calculation broke... got like 1500 degrees)

Nominal shutdown.

## 6/26/23
MP: R

After updating the state setup program to correct the angles, ran the program and found that the data and graph did not match at all. The data did not fit to a quadratic and had no minimum. After checking the positions of the measurement wave plates using APT User, I found that the half wave plates were in the correct position while the quarter wave plates were not. I then manually moved the motors to be measuring in the AA basis since I believe that is what we want when the incoming state is Psi plus. The data did not change substantially after changing the basis to AA. The graph looks like it has a maximum, which might be where the minimum is supposed to be. I also added a very simple full tomography method to the manager.py file for later use to check the created states. Looking at the angles to make Phi plus, it looks like the angle for the QP occurs at the same spot the maximum occurs on the graph, meaning that currently the correct angle for the QP should maximize counts. I realized that I didn't turn B_C_HWP in order to make Psi plus instead of Phi plus, so that is likely why the counts are maximized. This suggests that the measurement wave plate angle calculations are correct. I can't seem to find an experiment to determine the angle that the B_C_HWP should be positioned at to flip H into V. It should in theory be 45 degrees, but I would like to run an experiment to be sure. The experiment confirms that the angle is 45 degrees. I am shutting down for today but will check the code for the U_V_HWP angle tomorrow.

## 6/23/23
MP: R

 Arrived at 9:00 am to turn on laser to warm up. Came back and began running experiments at 12:30 pm. There were no errors on start-up. I messed up again and got the same data analysis error when trying to read the data. I realized that I was using the column names from the CSV and not the manager output. I read the documentation, and I just forgot about this. Now that I have updated the code, it no longer runs. I think there is a problem with the sweep function because it doesn't give you an error when you tell it to move. I am getting the access denied to COM9 error again. After lunch, I came back and ran the experiment again and the errors disappeared. I believe that they have something to do with the Managers not shutting down correctly. I have been shutting down the managers using Ctrl + C to interrupt the program. After shutting down Manager and retrying the QP sweep a second time, the program ran without error. The angle that minimized counts for the QP was 17.33 degrees, so I turned the QP to that angle and began the UV_HWP sweep. An error in the analysis code for the UV_HWP sweep turned up, so I will perform the analysis on my own computer. After this, the setup was shutdown for the day.

## 6/22/23
MP: R

Performed a preliminary test of the code to generate one of Eritas' interesting states. Arrived at around 9:40 and turned on laser to warm up. At around 1:30, came back to lab to begin testing. Encountered some errors in the process. Changed all instances of "CHANNEL_KEYS" in CCU code to correctly be "_channel_keys". There was an error with being unable to access COM9, which connected to Bob's Creation HWP. After opening Ello, the port and address of Bob's Creation HWP was correct, but the error was still occurring after more trials. After closing Ello in the Task Manager, closing VSCode, and checking to see if there were any other managers open, I reran the code and the error still persisted, but the file ran correctly after a reran it a couple more times. Obtained a graph of the QP angle vs counts and determined that the minimum is around 17 degrees visually, but there are problems with the analysis code that I need to fix. Based on the graph, 10 degrees seems like a good range for the second QP sweep. This graph should in theory show the minimum when the state created is Psi plus, since I set both alpha and beta to be 0. Shutdown setup at around 3 pm to fix code. After the meeting I came and checked the home of B_C_HWP, which was vertical, meaning the home position has not changed.

## 6/16/23
MP: A

I did a bunch of coding from home, came in at 10 or 11 to get everything up and running. Switching to using blit for plotting and a generalized `SerialMonitor` class that is extended by `CCU` and `Laser`. Tested `CCU` class it's all working now! It's a little bit faster than it used to be too. Now installing mbed OS serial drivers to get the laser monitor up and running.

Oh god I hate windows 7. To get the mbed controller working on USB we need to install
- ST-Link Driver
- ST-Link/V2-1 Firmware Upgrades
- 


## 6/15/23
MP: A

Took a day off yesterday to do the mathematics and consider the physical reality of our theory. We now suspect that the UV HWP may not be causing any problems at all! Today we are seeking to investigate the laser as a potential cause of drift. To do this, we will run the exact same drift experiment as the previous days, however we will let the laser alone warm up for about 3 hours before the test.

- Laser turned on and left blocked at 9:23. Waiting until at least 12:23 to begin the experiment.

In the mean time, I will be toying around with an arduino/raspberrypi/whatever tools I can get my hands on in order to monitor the temperature (and other statistics) about the laser. While these tools might be _nice_ to have while running this current test, the current test will allow us to determine if this minimally-intensive "warming up" procedure will allow us to stabilize our detections for further experiments, and so it is taking precedence.

- Returned around 12:30 after a morning's hard work on the laser-monitoring software/hardware.
- Got BBO setup. Re-configured the nitrogen regulator since we went through about 500psi on the 13th (it should be going much slower).
- Turning on detectors.
- Looked over the code to make sure everything is in order. We'll be running `drift_experiment.py` to measure HH and VV coincidences every other minute. We're setting the total run time for this experiment to three hours so that we can observe any hints of drift we might see, as well as any large initial drifts.
- In the spirit of keeping everything else cool for as long as possible, I'll start the script and then I'll unblock the laser (since the first part of the script is calibration and getting all the motors oriented anyways).
- Starting the script at 12:40, laser unblocked within the same minute.
- Script is running smoothly. It got the motors setup REALLY fast (perhaps they were already in place from a the last set of experiments?) and is now taking data
- Anecdotally, there is already a difference in counts (VV > HH) which we did not see on the cold run.
- Script just finished minute 2 of collection and is starting minute 3. Leaving the lab at 12:44 for the experiment to run.

- Returned at 15:43, experiment seems to have completed less than a minute ago.
- Shutting down experiment now, as the rest of the day I'll only be fuffing with the laser.
- Reconfiguration of the nitrogen regulator seems to have been a success, only used ~100psi today and the flow of nitrogen is still noticeable at the outlets.
- Renaming and uploading data to github.

## 6/13/23
MP: A

The results from last night suggest that the UVHWP could indeed have temperature fluctuations influencing the results. In light of this, today we are running an experiment to hopefully determine in which configuration (hot or cold) the half-wave plate best performs it's function as a half wave plate. The strategy will be to minimize HH coincidence counts while the detector is cold.

- I am keeping everything cold as much as possible prior to the experiment, however some testing and calibration will be required to make sure we will collect the data we want. I will try to let things cool in between runs, and mark significant times in this notebook.
- One thing I am doing however is warming up the laser. I turned on the laser at ~10:25 and left it on while writing the code for these tests and calibrations and such.
- Unblocked laser at 10:55 to manually verify the premise of the test.
- Confirmed that VV is ~ minimized when C_UV_HWP -> 45 degrees
- Blocked laser again at 10:58.
- Unblocked again at 11:07 to debug the calibration script.
- First try had an error in the fitting script, second try began at 11:12.
- Second try failed to output. Reblocked laser at 11:14.
- Unblocked at 11:17 to further debug calibration. Reblocked at 11:20. Calibration script works! Going to let everything cool for a few minutes before the final cold calibration.
- Unblocking at 11:26 for the final cold calibration run --> data saved to framework/min_VV/cold_*
- Left the **laser unblocked** and the **detectors off** for about 2 hours over lunch. Returning at 1:26 to take the warm calibration data.
- Completed calibration for the warm setup, the results are suscpiciously simmilar to the cold results! What the heck!?

- Now I am attempting to run pretty much the same script but using it to find the angle near 0 that minimizes HH detections to observe the difference.
- Run into weird and unexpected bugs.
- Shut down the setup at 2:51 when it became clear these issues would be plaguing the rest of the day's work.

## 6/12/23 (Drift Experiments)
MP: A

### Procedure
- be as fast and as thourough as possible in the lab. if the drift is exagerated during the first moments of operation of the experiment, we would like to measure that.
- this also means we will need to load and test the code (as much as possible) prior to any experimental setup.
- for this first drift experiment, we will only measure coincidence counts in the $\ket{HH}$ and $\ket{VV}$ basis
    - _Note: I am considering a more full exploration of the state over time, perhaps a kind of tomography drift experiment. But for now it seems reasonable to limit the scope to the one aspect we're interested in._
- The scheme of the test is as follows. Every minute, we will have our apparatus orient it's measurement half and quarter waveplates to measure in either the $\ket{HH}$ or $\ket{VV}$ basis (alternating).
- We then commence 30 seconds of data collection which will be broken into 6 x 5 second intervals, from which the mean and standard error of the count rate can be gathered.
- I have opted to not do any kind of live-plotting or live data output for the time being. I will do a short trial run to make sure that nothing crazy is happening, but I think any kind of live-data output would make me more likely to check on the experiment and since people coming and going is something we want to control for I've opted against that option.

This experimental procedure should give us lots of options for how we choose to aggregate and analyze the data. But first we must take it! I've just written the code that performs this experiment (`framework/drift_experiment.py`) and so this begins the lab session for completing the experiment.

### Experimental Notes
- Entered ~10:30. Lab is cold, if anything the back room is warmer than the front! Computer was very slow getting started, but now I think we're getting there.
- Began with a git pull and then ran `drift_experiment.py` to see if it works (lasers and detectors powered off). I'm letting this run for a few loops to make sure that it doesn't encounter any bugs immediately.
- Let it run for a few (~3) minutes, encountered one error fairly immediately: used modulo instead of integer division on lines 34 & 37.
    - Saved the log and output data for this run as `drift_experiment/debug0_all_data.py` and `drift_experiment/debug0_mlog.txt`
- Ran it again and it appears to be working normally. I got up and visually inspected to ensure that the only motors it manipulated inbetween trials were the measurement ones, the C_UV_HWP, C_QP, C_PCC, and C_B_HWP all remain still.
- I also ensured that the timing was safe by letting it run for a few minutes and seeing that it had a good amount of time in between measurements. (~18 seconds, well enough buffer for any weird motor mishaps if it takes the basis motors a little longer to get aligned.
- I let this test run for a good bit longer, saved the logs and data as `drift_experiment/debug1_all_data.py` and `drift_experiment/debug1_mlog.txt`.
- At this point I committed all these changed (including this notebook) and pushed the code.
- Now at 10:50, I am getting up from the computer desk to setup the experimental apparatus for data collection.
- Nitrogen at 710psi, laser thermometer at 19.4C
- I realized at 10:54 there was one part of the script I forgot to test. I turned off the detectors and laser both of which had been on for < 2 min to test this final data output. Given the nature of the experiment, this took a few minutes.
    - Saved to `drift_experiment/` as `debug2_all_data.csv`, `debug2_select_data.csv`, and `debug2_mlog.txt`
    - Caught one small "bug"(?) -- forgot `ignore_index=True` flag on `pd.DataFrame.to_csv`
- Turning on the laser and detectors again at 11:00
- Laser thermometer now down to 18.6 C
- Doing one last git-push with all these changes
- Ran the command to begin the experiment script at 11:03.
- First time the detectors and laser have been active -- count rates look nominal (a couple (1-3) hundred thousand raw counts, ~ 1500 for coincidence counts of HH and VV for phi_plus) 
- Waited around in the lab until 11:07 to make sure nothing has broken yet. Everything looks good. I'm going to make a sign for no one to come in and turn of the lights in the front room and let this run for **4 hours** now.

- Re-entered the lab at 11:43 because in preparing some analysis, I've realized that I've introduced a fatal bug in the termination sequence for the program (it's actually the same bug as before - i misused and misplaced the `ignore_index=True` flag and I should have used `index=False`)
- After examining the script and for a few minutes, I've determined that I will be able to salvage the same data from the manager's raw data output, and so I will not be terminating program execution early.
- Program execution is nominal so far. One misplaced termination byte occurred as I was entering but it doesn't seem out of the ordinary.
- Leaving the lab again at 11:46. Lights off, door closed.

- Re-entered for the final time at 2:54 (minute 230). Program is still performing nominally.
- When the experiment began I noticed count rates were relatively equal, but now they are noticeably unequal, so I am hopeful that this data collection will reveal some measurable drift. I am going to return to the front room to wait for data collection to finish before pulling the analysis/plotting code I wrote today.
- As expected, the code resulted in error and I had to manually interrupt it at 3:06
- Now turning everything off before analysis
- Laser thermometer 18.7 C

### Drift Experiment #2: Exact Same one, again
- Note that this is happening at 5:24, so we expect things are still relatively hot.
- While taking some pre-experiment thermal photos, the thermal camera's power cable exerted a small force on the far mirror.
- Laser at 18.3 C
- N2 at 525psi
- Running the exact same script but increasing the runtime to 5 hours.
- Started script at 5:35
- Leaving lab at 5:38, everything running nominally

- Returned aroun 10:45. Program shut down as expected.
- Turned off N2 supply at 275psi
- Took thermal immages of optical components using the buisness card as a background.
- Completed analysis and pushed changes.

## 6/9/23
MP: A

Here to do some tests with some of the math that I've been working on. Laser temp 19.0C at 10:55. Double checked detector power supply voltages with ampmeter. Digital logic was at 4.89 -> 5.00V. All detectors in working order.

First test: balancing HH/VV using my script, but with the previously calibrated quartz plate setting of -24.1215 degrees (instead of the incorrect -0.421 degrees, which is actually the radian value). Previously calibrated value was 63.4 and my script came up with 65.2439869938821, so it seems there is *some* drift. Pretty significant drift honestly. I updated this value in the config. I actually kept running this balancing script and the drift is indeed quite significant. I did notice some temperature changes throughout the day: cold when i came in but then it got VERY warm, relatively speaking (laser at 19C) and has been getting colder and colder throughout the day. The drift has (anecdotally) only really gone one way throughout the day, so I think it could be possible that it's due to temperature changes. It's definitely worth talking to F&M at least.

No other notes on the apparatus, left it all day until close.

## 6/6/23
MP: A, O
Room was a bit warm (68 F) but by the time we turned on the laser (~10:30) the laser thermometer was down to 18C (<20C) so we proceeded. Got no VV coincidences for phi+ --> found error in config file where A_HWP = 0 for VV measurement (should be 45). Out of curiosity I tested running the script with CCU.PLOT_SMOOTHING = 30 (update plot every 3 seconds, maybe faster performance?) and there was literally no difference in run time.

Finally got balancing script to run all the way through without issue! Continuing to do narrower and longer runs with `balance.py` to determine optimal angle for C_UV_HWP for phi+ (or phi-) -> I will update this value in `config.json` line 109 when done (previously was 63.4 degrees).

Optimization idea: let `CCU` (class) calculate means and SEMs of trials to not waste time returning values. Previously, the first sweep in `balance.py` (`SAMP=(5,3), N=20`) took 6:30 (minimum time for sample collection alone would be 5 min). After making the change, the same sweep took 6:30. Nice! No bottlenecks there. I am going to leave the change in there for now, since I think it is the cleaner way to do things.

I also made another efficiency related change that sped up the un-blocking of motors after they've moved. I'm noticing some weird "unexpected response" errors coming from the `Motor` module, but it's unclear why.

## 6/5/23
### Morning
MP: A, R
Relatively normal morning, A debugged code using the computer. Used detectors but not laser. Learned that face-ID can cause anomalous coincidence counds up to 4/sec at the desk or 8/sec on the other end of the room -> **need to address this**. Bumped 5V voltage knob _before_ turning on detectors but noticed, disconnected detectors and checked with multimeter and it was still at 5V. All detectors (that were operational) are (still) operational.

### Evening
MP: A, O, R, L (mostly A in lab)
Doing code overview. When aranging the nitrogen lines I (A) lightly grazed BBO housing, not with any appriciable force, and the counts continue to align with what we saw last time, so I don't think there is any difference in alignments.

## 6/2/23
MP: A, O, R, L
Main room was a bit warm initially but not feels like 64 which matches what we want. The teperature in the experiment room seems pretty constant; was 17.9C when I (O) arrived and is 18.0C about 40 mins later.

For reference: bottom left knob of BBO affects a vertica translation, so this will change the HH pair production; likewise adjusting the upper right will rotate horizontally, affecting the VV pair production.

C0: alice
C1: bob, H
C2: dne (would be opposite for A)
C3: bob, V

We adjusted the BBO crystal by monitoring the count rates of C4 in HH and VV, the former around 3100 and the latter at 3250 or so.

We determined the correct offset for the UV_HWP when we want HH=VV to be -22.3 degrees from home. We ran an initial sweep of -40 to 3 degrees for the QP in the DA basis with 25 steps and 25 samples. We are re-running from -.1 to 0.05, for 25 steps and 25 samples.

We turned the UV HWP by 90 degrees to try and find a minumum in the DA basis in the range -.4 to -.35; however, this yielded a maximum; so we are rotating the UV HWP by 45 degrees since this should change H->V, V->H.

New setting for Phi+:
- UV_HWP: 22.3 + 45 degrees --> this minimizes the DA counts, which would correspond to Phi- and we want Phi_
    --> tuned value is 63.4 degrees.
- QP setting: -.421 +- e-8 => tuned value is 63.4 degrees

For PCC: want to minimize DA counts. Trying sweep from -10 to 10 degrees in DA basis, 20 steps and 25 samples (since the coincidence count rates are low). Comparing C0, C1, C3 counts: C0 and C1 higher than C3, but not the same--stray light that's not entangled perhaps is entering; also detectors aren't as well alligned as they could be? For C3, the the detector is oriented Point is we can't really compare C4 to C6 numbers.

PCC: for Phi+ is 0.0699 [EDIT: RADIANS] (calculated using weighted avg for 2 fits). Purity: 0.947. Counts:
- DD: 1546
- AA: 1503
- AD: 46.8
- DA: 35.9
We could optimize this if we wanted to use this state, but since we are not, we won't do this at the moment. So we're callibrated!! Laser party!

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
- was able to achieve a basic sweep measurement of the UVHWP ! :)
- using `take_data(5,1)` should take 5s per iteration and it actually ended up taking around 

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
