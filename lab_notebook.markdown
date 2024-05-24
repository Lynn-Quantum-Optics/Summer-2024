## 8/11/23
MP: 0

There will be a scheduled power outage on 8/12 and 8/13, so I (TWL) came in to check on motors and shut down computer.  PCC looks visually vertical.  BC HWP fast axis is visually vertical, with the side labeled "fast axis" on top.  Not sure about UV HWP.  Of the T-Cube motors, S/N ...904 is not currently being recognized by APT User or found by APT Config.  This is Alice's HWP.  APT User reports that the other 3 motors are all at 0.  The current dial settings (by eye and in degrees) of the four motors are:  AHWP=9, AQWP=8.5, BHWP=8.5, BQWP=8.5.  These could all be the same 8.5-9 for all I know, as my visual resolution on these dials is not terrific.

Clearly some things will need to be fixed and recalibrated before we can run again.  I'm not changing anything right now but will just shut down the computer and see where we are after all the planned power outages are over.

## 7/29/23
MP: O

Came in to check on the setup, and turns out the lab computer was shutdown, presumably due to the same power outage that happened on Friday. I tried running an experiment (both just loading the manager in an ipython terminal and then running framework/decomp_expt.py), and there's an error in initializing the manager: "Elliptec motor has no attribute _type".... Not sure status of callibration.

## 7/26/23
MP: O

Three goals tonight: first, full tomo on PsiM to check sign of antidiagonal elements as well as general setup function. Second, redo UVHWP vs phi sweep now that I have corrected the sweep file (m.sweep returns angles, C4!!). Third, depending on results of two, take full tomo data on states sweeping QP to extract phi.

One: for PsiM, using previous settings: trial 35:
    UV_HWP 24.582709288443688
    QP 323.733168204273
    B_C_HWP 45.0
result: fidelity 94.9%, purity 94.4%. Diagonals bit imbalanced: .566 and .454; also, antiadiaongals real part mag .45 (correct sign), but with -.12i imaginary part...
    [[ 0.00578295+0.j          0.01839501+0.00791342j -0.01370648+0.00188341j
    0.0035268 +0.0004162j ]
    [ 0.01839501-0.00791342j  0.56620353+0.j         -0.45404027-0.12177766j
    0.0175761 +0.00256338j]
    [-0.01370648-0.00188341j -0.45404027+0.12177766j  0.42335712+0.j
    -0.01430214-0.01318927j]
    [ 0.0035268 -0.0004162j   0.0175761 -0.00256338j -0.01430214+0.01318927j
    0.0046564 +0.j        ]]
Result not quite as good as before (~96%) but within a degree of fidelity and purity. 

Second: redoing UVHWP vs phi sweep. Success! Plot for 10 increments from 0 to 45 reveals a sharp increase in phi around 5 degrees and then a slightly positively sloped plateau and then an increase at 45 degrees. Redoing sweep with 30 increments: very nice qubic! The jumps near the ends make complete sense bc at UVHWP=0, we produce only VV and at 45 only HH, which have meaningless phases.

Third: take 6 full tomos sweeping UVHWP from 2 to 43; try to put on same curve as sweep with actual phi, 30 iterations. Trial 36.

Fourth: do 30 iteration QP sweep: -38 to 0. take 6 full tomos sweeping QP: -36 to -2

## 7/25/23
MP: A

No notes from yesterday as to what experiments were being run/where things went wrong :( All I know is the ThorLabs motors threw the same error and I'm going to try to fix them again. Booting Kinesis to check out the situation and figure out what is going wrong.

901 Is the one that is freaking out (Bob's HWP). Pressing home in kinesis failed to move it. Disconnecting in Kinesis, manual jog worked, so I'm reconnecting in kinesis. This worked, and then sending the home command worked as well. In the mean time, I'm sending all other devices to zero (through kinesis). This succeeded, and we ended up with all of them at around 9 degrees on their dials.

Turned on detectors and booted up an ipython terminal to check count rates.
- making state VV and measuring in basis HH produces 30.1 +/- 1.4 counts/sec (5x3s samples)
- making state HH and measuring in HH produces 2833+/-14 counts/sec (5x3s samples)
- making HH and measuring in VV produces 9.0 +/- 0.4 counts/sec
- making VV and measuring in VV produces 3035 +/- 14 counts/sec

Count rates look good!

I'm going to try loading in v1.1 of the lab framework just to see what happens. I don't know if it will really _solve_ the errors that we are seeing, but hey, maybe it helps?

O here. Ran into a key error but reverted to previous lab framework version and copied the channel_keys into the config. Problem with sweep_fits_updated: giving really low count rates as UV HWP -> 0 even when QP = 0; B_C_HWP = 0 whole time. Alec says issue may have to do with setting angles after initializing but before sweeping.
## 7/24/23
MP: A

Opening kinesis. All motors read 0 and none are at 9 degrees so they will all need to be re-homed. Since they are all homed in kinesis, this will have to be done semi-manually.

Testing on 904:
- disconnected from USB
- attempted manual jog that failed
- powercycle
- manual jog disconnected from USB worked
- reconnected to USB
- kinesis failed to connect with it
- powercycle while connected to usb
- reconnect in kinesis succeeded (with position error)
- rehoming in kinesis succeeded
- dial now reads 9, should be good to go

For the rest (667, 901, 646):
- disconnect within kinesis
- powercycle while connected to usb
- reconnect in kinesis
- rehome in kinesis

The above only worked for 646. I had to disconnect from 901 and 667 (within kinesis), then re-do the same process for them one at a time.

This still did not work for 901 however. I tried then powercycling and manually homing it, which caused another power cycle. At this point I resorted to diconnecting the USB, power cycling, jogging back and forth a few times, reconnecting the the PC and homing in kinesis which finally worked.

All ThorLabs motors are now homed properly.

## 7/23/23
MP: O
Laser was on from Friday. Experiment 1: setting PhiP, but then analyzed UV_HWP=10, 22.5, 45; taking full tomo so we can try my method and Alec's for measuring phase. Visually, the measurement motors do not look at 0--actually more like 10 degrees.

Purity is looking really low at 10 (.7778) and 22.5 (.4926), but at 45 it's .9475... Tested on settings found in Richard's updated full tomo for PhiP (but left Bob at 0 instead of 45, so realized making PsiM): purity of .9213. Inspected the setup and noticed that the BBO N2 tube was slightly intersecting the beam, so I fixed it and am retaking PsiM. I also noticed that Bob's C_HWP is partially intersecting the beam as well...This slightly increased purity: .92256.

A here. I checked out the pump beam hitting BCHWP and it was a small intersection but definitely present. I also noticed that BQWP was misaligned fairly significantly from retroreflective when I removed BCHWP to realign it.

I realigned BQWP and BCHWP. Now BCHWP doesn't intersect the pump beam at all.

O here. Beam path looks good! At 22:00, rerunning PhiP using preset callibration. Got error: 
    Exception: Getting status failed: Time out while waiting for hardware unit to respond.
Error persists after killing terminal. Checking kinesis and APT user confirms that it is one of the Thorlab motors; in particular, Bob HWP. Unplugged and replugged motor, homed--dial reads about 40 degrees. Properly dswhutdown manager object after the fact as well. Will need to recallibrate.

## 7/21/23
MP: A

Started warming up the laser at 0700 so that I can get started with calibration right away. Back now at 0920.

Before running phase_finding I opened kinesis and noticed that 646 was at 3455.355mm but also visually at ~9 deg (i.e. home). To be safe,  I used kinesis to send it to zero before quitting out and running the tests. At zero it is still visually at 9 degreees (i.e. the hardware home).

I wasn't super great at writing down everything I did today but basically I just calibrated phi plus. I started with a full QP sweep then I had some sign errors so i rotated the UVHWP to 67.5 to get rid of them. Then I did a ratio tuning sweep and then another phi sweep and then another ratio tuning sweep to really narrow in on the state. Unfortunately, I really do think the drift of the laser is the major limiting factor in these experiments, so I don't know if it's really all that possible to do much better than we have already done in the past.

I'm running a full tomography of phi plus now to check it out. The full tomography crashed twice but I got the data the second time and fixed the bugs.

## 7/20/23
MP: A
Testing file outputs for today are in BCQWP/fwrv_freq_tuning

The last thing we did yesterday was mess around with the fwd/rev frequencies of the Elliptec Motors. Unfortunately, we accidentally did a reset on the B_C_HWP motor (instead of the B_C_QWP). First things first, I'm going to try to re-run the same testing program we've been using to reproduce the error (`BCQWP_err.py`). Indeed this did show that the error is still present (0).

Note: Factory reset/search on BCHWP left everything as

- M1 FWD: 80.1
- M1 REV: 102.4
- M2 FWD: 78.8
- M2 REV: 100.3

Then I opened Ello and did a factory reset on the correct motor (BCQWP). This did not change anything, but Ello says we must re-search the frequencies. Doing this achieved

- M1 FWD: 80.5 -> 80.1
- M1 REV: 101.0 (no change)
- M2 FWD: 79.7 (no change)
- M2 REV: 103.1 (no change)

Seeing the change, I re-ran the same testing script with the exact same parameters (1). The same error is still present. SAD! I'm going to try now using the "fix" button which warns that it may cause malfunction? But how much worse could it really get.

- M1 FWD: 80.1 (no change)
- M1 REV: 101.0 -> 101.7
- M2 FWD: 79.7 -> 80.1
- M2 REV: 103.1 -> 103.8

I notice that if you don't hit the set button, the changes don't get made in the device (i.e. after fix, read settings will reset the change). After this change I re-ran the same script with the same parameters (2).

I'm going to try something else now. I'm re-searching the frequencies real quick so now we have

- M1 FWD: 80.1
- M1 REV: 101.7
- M2 FWD: 79.7
- M2 REV: 103.1

Then I'm disconnecting from ello and re-installing a version of the lab framework where the Elliptec motors are required to have an integer value for the ppmu. Now I'm re-running the same script with the same parameters (3). The test pretty convincingly showed that a jump was still present, but to be absolutely sure I'm going to re-run the same test once more with finer sampling and longer sample times (4). I am noticing that we are actually having it jump from like -1 degree to zero degrees... I am thinking that maybe narrowing in on the range might shed more light on the issue? So now I am setting the testing script sweep to be from -2 to 2 instead of -5 to 5. We'll see what happens (5)? Indeed the jump is still there. For one last sanity check I am going to reinstall the old version of the lab framework (where ppmu will not be an integer, instead something like 398.22) and run the same zoomed in test to see what happens (6).

Still there!!! Ahg. I'm going to use Ello to do cleaning and optimization and come back after the meeting to check on it. Ello seems to have completed cleaning and optimization just fine! Now I am running the same sweep as before (optimization/4) to assess if it helped to cut down the error. It did not help :( Indeed the error is still there!!! Darn it. I'm thinking about what our next steps here are going to be

### Briefly testing the fast-axis offset thing...
I tried again booting an ipython terminal and doing AQWP/BHWP/BQWP/BCHWP/UVHWP/QP/PCC to zero and AHWP to 45. This time when I found the minimum of the C4 count rates it DID occur when the fast axis was (visually) vertical. I think that this might be due to some errors on my part but I really don't know what. The console logs (not saved) back up everything that I noted earlier...

### Misc notes/checking things
- ppmu on BCQWP motor is the same as other elliptec motors (UVHWP/BCHWP/PCC)

Tried using weird commands like search frequenies, also verified that motor info matched calculated values. 

Next step ideas:
- Just now, reset the motor frequencies to defaults using ello so first thing is to test that when I get back. But shoot! That was on bob's creation HWP. Oh well. they all had the same problem (according to testing) anyways.
    - Relevant test: with HWP at 0.1155 deg count rates at 7.55+/-0.76
    - HWP at 360-0.1155 count rates were 5.5+/-0.44
- Also might be worth trying to use ello to "fix" the motor frequencies. 

Also what the heck! Now the problem is back where count rates are NOT being minimzied when the motor is visually vertical?? I'm so confused! And I just did some really concerning tests, rotating the motor back and forth 180 degrees and noticing count rates going all over the place. I need to record this data but first I want to make absolutely sure that the ThorLabs motors are "homed".

### Homing ThorLabs 
tldr; I tried a bunch of things that didn't work. long: I tried just opening Kinesis and pressing home, but this zeroed the position of the motors and did not move them back to their 'factory' home of ~9 degrees. Shoot! So I unplugged Bob's QWP from the PC, power cycled it, and used the manual homing function of the driver. This seems to have done the trick for that motor. I tried homing Alice's QWP manually without unplugging from the PC and it didn't work. I tried disconnecting all 3 other motors from kinesis and manually homing but this caused continuous rotation. I power cycled Alice's HWP then manually homed before re-connecting the USB cable. This worked, so I did the same on the other devices.

**WHAT DID WORK:** connecting all motors to PC, disconnecting from them in Kinesis, power cycling all of them, then reconnecting in kinesis and pressing home.

### I also Double-Checked Detector Voltages at This Point

### Very Bad No Good Data
I wrote a small script to just rotate the wave plate 180 degrees back and forth and take 30 seconds worth of data each time. I'll run 30 trials so this should take about 30 minutes. Lets see what happens!

This revealed exactly what I suspected -- the QWP must be loose inside of its housing as not only does the count rate change due to turning the motor back and forth, but the two lines also diverge from each other, so the 180 degree rotations which should not be affecting the waveplate's affect are indeed changing _something_. As I mentioned, I believe the QWP is loose in its housing. I am going to remove it from the apparatus for now to complete the calibration of Bob's waveplates and check the calibrations on the others.

### Checking in on Bob's Creation HWP
Since some of the frequency settings changed, it is probably a good idea to check that it does not suffer from the same 360 degree rotation error that the other one has. I'm modifying `BCQWP_err.py` to run a simmilar test for BCHWP, putting this code in `BCHWP_err.py`.

I ran the first sweep (0) but I'm realizing that BCHWP still had a small offset in the config file (of 0.118) and that may affect things since the manager will the angle. For example 359.9 would be 360.018 which would be changed to 0.018 which might obsucre any jump? I don't think it would, but to be sure I'm going to zero out the offset in the config (`config_ZO_BCHWP.json`) and re-run the experiment (1). This experiment to me confirmed the absense of a jump.

I did notice a bit that Bob's creation HWP's configured zero might not exactly be at the desired position, but since it was configured before Bob's QWP I think the best corse of action is just to continue on with...

### CHECKPOINT B (First Try)
The code for this is in `checkb/check.py`. Since Bob's QWP was goofed up, I've commented out the code that was intended to mess with it. This first attempt (-1) went pretty poorly. Some of the offsets were up to 5 degrees off! I'm really not sure what happened here. I really want to re-run checkpoint A to ensure that everything there is groovy. This will involve removing Bob's PBS.

### CHECKPOINT A (Testing Again)
I went ahead and re-vamped the `checka.py` script. Now I'm running the test again to see how we are doing on the calibration of Alice's components. In the mean time, I have removed Bob's PBS and Creation HWP to simplify re-calibration when we move back to that side of the apparatus.

I ran this script once with some coarse sampling just to get a vibe for what was going on (-1). The data was mighty coarse, but suggests we are still at least within the ballpark of calibrated. I'm re-running with finer more detailed sampling to get the final updates to use. Note that in these runs, the magnitude of the minima found were practically identical to the minima found in the previous sweeps a few days ago (~30 counts/sec).

#### ROUND 0
- UVHWP extrema: 0.022+/-0.020
- UVHWP Update: -1.381 + 0.022 -> -1.359
- AHWP extrema: **0.202**+/-0.021
- AHWP Update: -9.351 + 0.202 -> -9.149
- AQWP extrema: **0.116**+/-0.035
- AQWP Update: -8.385 + 0.116 -> -8.269

#### ROUND 1
- UVHWP extrema: -0.054+/-0.014
- UVHWP Update: -1.359 + -0.054 -> -1.413
- AHWP extrema: -0.018+/-0.015
- AHWP Update: -9.149 + -0.018 -> -9.167
- AQWP extrema: **0.12**+/-0.04
- AQWP Update: -8.269 + 0.118 -> -8.151

#### ROUND 2 (FINAL -- NO MORE UPDATES)
UVHWP extrema: 0.073+/-0.019
AHWP extrema: 0.072+/-0.021
AQWP extrema: -0.04+/-0.04

### Bob's QWP
I reinstalled the PBS and ran the calibration script (after zeroing the calibration value in the config) to find a minimum detection angle of 88.80605032403813+/-0.022804069066468032 = 88.806

### Alignment
Realigned (using redlight retro-reflecting) Bob's QWP, then HWP, then removed HWP and aligned Bob's creation QWP. All seems well, count rates (checked in ipython) still good, maybe even a little higher (~3250 max).

### Bob's Creation HWP
Calibration found a zero position of -0.04332648860882957+/-0.019312100787063153 = -0.043 degrees for the creation HWP. Put 

### Bob's HWP
I reinstall Bob's HWP on the magnetic mount and at this point I removed Bob's QWP from the config file. It was sufficiently screwed up enough that we saw the need to remove the plate from the housing. At this point I ended up doing two coarse-ish sweeps of the HWP (0) before doing a finer sweep about the minimum, which would appear to be at around -15.625032742716252+/-0.01652463743600131 = -15.625. I put this into the config file.

### CHECKPOINT B FINALLY
With offsets for Bob's QWP, HWP, and CHWP updated in the config, I can now finally run with the second checkpoint. The first time I ran it (0) I accidentally left coarse sweep parameters in the file (-4,4,15,5,1). Subsequent tests will use fine sweep parameters (-4,4,20,5,3) however I don't see a problem with using these coarse sweeps to at least narrow in a bit on the minima for the offsets. 

#### ROUND 0
These calibration data used a coarse sweep, but indicated that all errors were within 0.1 degree! Crazy! I'm running the sweeps again more carefully to double check this before determining the setup to be calibrated.



## 7/19/23
MP: A
Started laser at 837.
Back at 1107 to get going.

### Fixing the issue with Elliptec BCQWP Motor
First I am going to reproduce the problem by modifying the code to perform a manual sweep (instead of updating the manager to an older version) and using angles modulo 360. I put this code in `BCQWP_err.py`. I also confirmed that all the ThorLabs motors were homed, just to be safe. I was able to reproduce the error and saved this as the file `BCQWP/optimization/BCQWP0.png`

I wrote a small script to send the optimize motor instruction and wait to listen to the but the Elliptec motor documentation was relatively unclear about the response that should be expected and I also wrote the script pretty poorly so now I am just waiting "several minutes" for the instruction to be executed. I opened an ipython terminal, connected to the serial port and just waited to get the 'BGS00' status message. This really did take several minutes, about 10 if I had to guess.

Now that it is complete, I'm running the same script that reproduced the error a few minutes ago (`BCQWP_err.py`) and observe the output. Uh oh! We hit an error from the motor that was just optimized -> `BGS02` which is a mechanical timeout error according to Elliptec documentation. Well. If at first you don't succeed, right? I tried re-running the same script and I hit the same error.

The mechanical time out status keeps coming up, not sure what is going on. I'm going to connect to the motor using Ello, as the documentation says we might be able to do the homing/optimizing/cleaning and such from there. Indeed there are buttons for that, but after sending the motor to its home position it came back to life! It was able to move without timing out and such. So I went back to python and it was also able to move there. Then I re-ran the test.

This did not fix the problem weirdly enough. Still have a jump in the graph, and furthermore we are seeing different count rates than before optimization! Much higher, even though the home position should not have changed. I'm going to boot the manager in an ipython terminal to see what is going on with these count rates!

All angles here are based on calibrated zeros
- UVHWP to 0
- QP to 0
- PCC to 4.005
- AHWP to 0
- AQWP to 0

These count rates should be tiny no matter Bob's settings, and indeed they are! Less than 20, around 10. Moving AHWP to 45 so that we will see count rates based on Bob's basis. Now I'm sending

- BQWP to 0
- BHWP to 0
- BCHWP to 0

Now seeing count rates on the order of 1200. Now I'm going to visually inspect things real quick. And I'm officially quite confused. The count rates should be low but they are decidedly not. I hit the ThorLabs motor error while trying to mess around with BHWP to see if that was the problem. YIKES!

### Debugging ThorLabs Motor error

I'm booting Kinesis and I see the fatal position error. I'm requesting that it home the motor. It just sat there blinking homing for a while, and did not move the motor. In the mean time I sent all the other motors to their zero positions and they look consistent with the recorded home of ~9 degrees. I tried disconnecting, reconnecting, and homing again. Same problem.

**THE FIX**: Disconnecting in kinesis, Jogging the motor back and forth a couple times manually, then reconnecting and homing in kinesis.

While I was here I homed all the other motors but they had already been homed so nothing really happened. I visually checked that all the thorlabs motors were at about 9 degrees at this point and they were.

### Back to debugging count rates
Booted ipython and performed the following
- UVHWP, QP, AQWP, AHWP to 0, PCC to 4.005
- Witnessed count rates between 10 and 30.
- Moved A_HWP to 45.
- Count rates jumped to ~1000 for both C4 and C6.
- Moved BQWP, BHWP, BCHWP, and BCQWP to zero but count rates remained high, ~1200 on C4 and ~900 on C6. BCQWP is not calibrated, but its fast axis is visually relatively vertical. However, if I move the fast axis to -30 degrees from the vertical, we see a minimum of ~5 counts/second! I hypothesize that I likely wrenched this component in there so hard that I knocked the plate holding the crystal about 30 degrees! Yikes. Whatever, as long as we can find the minimum, right?

### Back to debugging the jump
It still exists!!! I'm going to try restoring the motor to factory settings to see what happens. The process is described in section 4.5 of the ELL-14-Manual.pdf from ThorLabs website. To be minimally disruptive, I'm going to disconnect the new motor from the bus distributor, and then use the hand-held controller that shipped with the motor to drive it seperately. I couldn't find the one that shipped with it so I used the one from the other motor that we had ordered. This process went just fine.

Visually it appeared just a bit better, and the data (optimization/3) seems to suggest it might be a bit better as well. Unfortunately, the piece of the sweep that we were looking at (near a weird, anti-symmetric maximum) makes it difficult to determine just how poorly it is doing. I decided to use ipython/Manager to determine where the minimum position now is (~40 degrees), then I used Ello to reset the motor's home position ('factory' home offset of 10 -> 50) and then I re-ran the script that reproduces the error.


## 7/18/23
MP: A

Started the laser at 0922. Going to wait about an hour.

Back at 1111. Setup the BBO nitrogen flow. Started by connecting Bob's Creation QWP to the bus distributor. Then I booted up Kinesis to ensure that all the ThorLabs motors were still homed (they were), followed by Ello to ensure that the creation QWP has the correct address. It was not, (since it was disconnected) and had to be reassigned to address "B".

### Experimenting with Elliptec Homing

I initialized Bob's CQWP motor outside of the manager using the `ElliptecMotor` class. I tried playing with it and learned that indeed the Elliptec Motors will lose their home positions if power is cut, and so they should always be returned to the zero position before leaving them alone if this is a possibility.

### Making Sure Everything is Relatively Okay
I started by running a sweep on BCHWP (2, though I forgot to use the calibrated zero of BHWP, and so the data was quite goofed up) and a sweep on BHWP (3) to ensure that those calibration values essentially stayed the same over night. The sweep on Bob's measurement HWP revealod a minimum of -0.022 (well within 0.1 deg! yay!)

### Aligning and Calibrating Bob's Creation QWP

After the basic test, I got the lights back on and worked on aligning Bob's creation QWP by retroreflecting the redlight laser. This process went swimmingly!

Now I'm running the code in `calibration/BCQWP` to perform a coarse sweep (0) to find the angle that minimized C4 coincidence count (while Alice is measuring in the V basis). The coarse sweep found the minimum ~ 2.4 degrees, so I'm running another sweep across the 8 degree range centered on 2.4 with finer measurement parameters. Weirdly, this did not produce satisfactory results! There ended up being a lot of noise near that minimum, and so I am taking a slightly broader sweep with more precise sampling from -5 to 10 degrees. But even this one is looking super weird! Broadening the range once more to -10 to 10 degrees.

The QWP is giving quite a bit of trouble. We are seeing weird bumps in the plot around zero, and we also are seeing the motor rotating a full 360 degrees when it crosses zero. That along with the shifting minimum is making us think perhaps the mounting is indeed not as secure as it needs to be. I'm going to take it out, wrench it in there real good, realign it, and see what happens!

I think I was able to get it a little bit tighter! But that didn't fix the issue! I was able to fix the problem by allowing the code to accurately pass on negative angles to the motors, but the real problem is that the motors don't know have an accurate "this many pulses = 360 degrees". Going to debug tmro.

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

### Checkpoint A
I added a folder `calibration/checka` that has some files to do a peace-of-mind mini-sweep on each of the components that have been calibrated so far, just to check that they indeed minimize the count rates around 0 degrees. Unfortunately, the peace-of-mind check is not giving me much peace-of-mind.

See `calibration/checka/seven_sweeps/mini_notebook.md` for bad ideas (too small ranges on the sweeps made the offset adjustments meaningless).

Now we're going to use 16 data points between -4 and 4 degrees, 5 x 3s samples per point in order to narrow in on the offsets for the calibrated componentes (UVHWP, AHWP, AQWP) before moving on to Bob's side.
- UVHWP -1.029 - 0.368 -> -1.397
- AHWP -9.470 - 0.026 -> -9.496
- AQWP -8.704 + 0.23 -> -8.474 (this one had weirdly a lot of noise)
I forgot to save the plots for these sweeps. Sad.

The second set of sweeps completed (savedd in `calibration/checka/sweeps1`) and indicate updated values
- UVHWP -1.397 + 0.109 -> -1.288
- AHWP -9.496 + 0.157 -> -9.339
- AQWP -8.474 - 0.03 -> -8.504

The third set of sweeps are in `sweeps2`. These are looking super promising! I'm going to make some small changes and then 
- UVHWP -1.288 - 0.053 -> -1.341 **NOTE: I actually forgot to update this one! But took the best data yet anyways.**
- AHWP -9.339 - 0.083 -> -9.422
- AQWP -8.504 + 0.119 -> -8.385
The adjustments appear to be getting smaller, which is quite promising. Hopefully after this final run of updates we will have some offsets within 0.1 degrees.

These data (in `sweeps3`) are great!!! Offsets have finally dipped below 0.1 degrees but the UVHWP offset of -0.093 and the fact that I forgot to update it last round makes me want to try just one more round to see if we can't squeeze a little more out of it. Here are the updates I made for this round:
- UVHWP -1.288 - 0.093 ->  -1.381
- AHWP -9.422 + 0.071 -> -9.351
- AQWP had 0.02 +/- 0.04

Final round of sweeps are in `sweeps4_final`. These gave us offsets of
- UVHWP -0.006 +/- 0.022
- AHWP -0.022 +/- 0.018
- AQWP -0.024 +/- 0.027
I am not going to modify the config file offsets for these motors anymore! So they are
- UVHWP: -1.381
- AHWP: -9.351
- AQWP: -8.385

### Checking Alice's Measurements in the V basis
I did a little sweep of Alice's measurement HWP to check that counts were maximized around 45 degrees. It looked fine!

## Moving to Bob's Side

Now we are locking in the UVHWP and will not be messing with it or its offset any longer. I have removed Bob's Creation HWP and measurement HWP to perform the sweep/zeroing on his measurement QWP. I also put the PBS back in _very carefully_ and it seems to have gone okay!! The count rates still look very decent.

### Bob's Measurement QWP

So the offset for Bob's QWP came out to 88.88671845585692 +/- 0.022142714825859715 based on the fits in the calibration folder.

### Bob's Creation HWP

Sweeps indicated an offset of 0.11797772180123152+/-0.01615794120861485 = 0.118. I added this to the config file.

### Bob's Measurement HWP

Sweep fits gave us -14.805680946708112+/-0.01413407436314716. I added this to the config file.

### PCC
Make HH + VV, measure in A/D and minimize AD and DA counts (best job at getting a zero phase shift between HH and VV).


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

## 2/06/2024
MP: Lev
Booted Kinesis and AQWP (904) was not connecting--fixed by power cycling a few times. Attempted to run aqwp.py to start calibration procedures but only got serial number error back. Had to run to class so goal for next time is to fix serial number error. 

## 2/08/2024
MP: Lev, Lynn, Alec

Encountered errors in Kinesis right away, i.e. all of them throwing serial number and home position errors. Managed to fix 667, 901, 667 by a random assortment of manual homing, jogging, disconnecting, and reconnecting. Learned 646 had a faulty power cable which has since been replaced and for the first time in my three weeks in the lab, everything works! Today is the day we start calibrating. Upon first running aqwp.py we received a __typ error, which was fixed by ensuring that the assignment of COMM in ello matched that of config.json (COMM7 had been set to 0 instead of A).

1640 Ran AQWP.py, which did read data into the aqwp_sweep2.csv file. However, errored out with m.shutdown due to an error with the C-PCC elipptec motor not returning to home. I have a suspicion that it is due to COMM5 being weirdly out of assignment but must run! So my goal next time is to try to reassign COMM5 correctly and then re-run.

# 2/10/2024
MP: Lev
1410 Ran AQWP.py and successfully read data into csv -- ran into the same problem as 2/08 where it errors out when trying to shut down EllipetecMotor-C_PCC (expected response length 11 but got response b'CG502'). Will try to debug now. Started by loading up ello and manually jogging COMM5 to make sure assignements were correct, which they were! When first trying I had lots of 'mechanical time out' errors which were solved by changing jog angle to a different angle (20 deg) then back to what it had been before (82.286 deg).

1450 Rerunning aqwp.py to see if manually jogging the pcc had any affect on error output.
1502 Saved to trial 2.It did! Absolutely no errors when running the curve fit this time, although it did return a graph that looks almost like half of the plot from thecalibration manual, and only went degree-wise from -15 to -3... and as I write this I realize it is because sweep params was set to that! Will now rerun from -40 + 40 deg, as the slope of this trial did not zero out across the tested angles.\
1508 Saved to trial 3. At the start had a little Attribute Error "module '__main__' has no attribute '__spec__", which was fixed by CTRL-C and restarting ipythoin. This trial shows a minimum at approx. deg = -2, so I will run a fine sweep around that area to determine the exact new home angle of aqwp.
1522 Saved to trial 4. This produced the image in AQWP_sweep4.png, the minimum of the sweep being (by finding the minimum of the quadratic fit): -423/500, or -0.846 deg. This means that AQWP home (for now) is -0.846 off of its current home. 

Was planning on calibrating the UVHWP too but got a tad confused with what components I have to remove to run the trial, so instead calibrated the QP by setting up a screen in front of the laser and changing the angle until the retroreflection was good. This returned an angle of -13.806, but I'm not entirely sure about the method (adjusting from software until it looks straight and retroreflects semi-nicely) so may come back to this.

1643 It is still Bob's PBS that has to be removed for the UVHWP trials, so doing that now. Data in Trial 2 for first sweep.  It looks great, but because the minimum is not centered, I want to run one more sweep across a larger range of degrees to ensure it is still good. That said, the UVHWP angle is therefore +1.327 deg from its current home. 

1700 Turned off everything and went to get dinner. Goal for next time is to run a larger sweep for the uvhwp to ensure the minimum is correct, then continue calibration.

# 2/15/2023
MP: Lev, Lynn
1150 Turned on laser.

1215 Running UVHWP calibration with a wider range to confirm 2/10/2024's calibration. Finished running and the new minimum seems to be +2.331 deg, which
is very different from the calibration given in 2/10/2024.

1235 Running UVHWP calibration again zoomed in on 0 to 3 to get a finer search. Ran into a "main has no _spec_" error--fixed by restarting terminal. This time got 1.85 deg... I'm not sure why its changing so much! I'm going to retake the exact same trial to see if it matches this trial.

1255 Running it again got +2.01 deg. I don't know the resolution on this equipment so I'll ask later. For now I'm going to take it as necessitating a finer search around the 1.5 -> 2.5 range.

1335 Running UVHWP 1.5 -> 2.5 range. Got "Optical parameters not found: Number of calls to function has reached maxfev = 800" error--hypothesis is this comes from data getting more and more scattered the smaller the range and I went way too small on range. I'm going to run this oneeee more time with a 0>4 deg range to see if its in the range of trials 4 and 5.

1355 Running UVHWP once again (trial 7) from 0 -> 4 deg. Got +1.706 deg. As these values will get updated anyways in checkpoint A, I'll use the mean of +2.016 deg to calibrate AHWP. Will run later today.

1525 Tried to run AHWP after reinstalling AHWP. Ran into error code "VX", fixed by homing motors in Kinesis. 

Calibrated QP to get home to be -16.97 deg from current home, and also swapped nitrogen tanks. QP was calibrated by adjusting the angle in Ello until retroreflection occured directly beneath the laser. Some short term goals are now to reset all Thorlabs motors to be zeroed at their physical zero (the one we see on the motor itself) by power cycling once each looks to be at their zero--for the current calibration this means that I will add however many degrees it took to get to the new zero to the calibrated zero, and to also continue calibration overall. 

# 2/17/2024 
MP: Lev

Goal for today is to, before continuing with the calibration, to set all the Thorlabs motors homes to the zero we physically see on the motors. This wil help us going forward to confirm when the device is actually zero. This means that we will be updating the config file with likely all new data! Yay! #edit three hours later, no yay :(.

Step one involved an hour of getting the motors to connect to kinesis and with enough restting, manual homing, and replugging in USB cables they did. Step 2, 3 hours later, was to realize you can't do this in Kinesis.

For these motors, the home is set in the factory so we can't change that. The only thing we can do is change the 'home offset' in APT User. However, I tried this and could not get any change to actually occur (if anything, APT User just makes the motor try to 'home' infinitely). 

Goal for next time is to try to troubleshoot this, and if no solution can be found then continue with calibration.

# 2/22/2024
MP: Lev, Lynn

1300 Have not been able to change the homes yet, so far I've tried power cycling at 0 to change home, making a custom actuator profile for the device in kinesis, changing the home offset in apt user (which does not seem to have an efffect in kinesis), and homing in kinesis at various point. After approx. 2.5 hours of trying the above, the motors are infinitely homing when manually homed. One last time trying to home them through APT user with the home offset set such that they *should* be homed at the physical zero we see on the motors. Let run for ~15 minutes and still no homing. 

1700 Everything is in order! We have learned that APT User is way better for communicating with the motors but to **not use** the home button, instead to set to zero by typing in 0. We manually set each motor's 'home' to zero by moving the motor to the visual zero in APT User, closing the program, power cycling the motor, then reopening the program. No home-offset work was needed. Kinesis was spitting out reverse limit, ssn, and zero errors, which we fixed by not using Kinesis anymore. Goals are now to fully recalibrate from scratch, starting with AQWP. We will take out the UVHWP during this calibration to increase the strength of the signal we measure which should improve the resolution of calibration. 

# 2/23/2024
MP: Lev

1300 Laser on, day 1 of not using Kinesis. Ran into some issues with 904 on apt user, but solved by power cycling at visual zero. Goal for next 2 hours is to begin calibration again. 

1330 AQWP (previous trials moved to new folder in aqwp folder). Tried taking out the UVHWP to see how that helps calibration. Ran two trials (1,2) and got +8.429 deg as the new calibration. While the error bars are a bit off, this will all be adjusted in checka so is ok for now. 

1410  UVHWP Trial 1 looked good 

1645 Had to leave but now taking a finer sweep around the zero of trial 1. Got the change to be -0.899 deg. Goal for next time is to calibrate AHWP then use the three calibrations in checka to get a better measurement of them! 

# 2/26/2024
MP: Lev

1745 Laser on, goal for today is to calibrate AHWP and go through checka. Note I had to put AHWP back in for this. After a coarse then fine sweep, got +8.533 deg for a zero homing. Troubleshooted 'VX' key error before realizing it was because config.json has 'VV' not 'VX' (which are the same thing) to run make_state on. 

1910 Tried running checka, continually getting error with AHWP and getting a quadratic fit from it. Going to run a sweep on AHWP to see if the calibration is wrong for some reason. For some super odd reason, the calibration is really really off for AHWP.. it is now returning -4.523. 

2050 Now rerunning checka with updated AHWP value in config.json. After running, none but the UVHWP actually had extrema in the -4 to 4 region we're sweeping in checka. I honestly have no clue why. Goal for next time is to redo the AHWP and AQWP calibrations so that we can hopefully succesfully run checka. Did not end up using the table below, but keeping for posterity.

Here are the initial values put into config.json pre-checka:
| Motor | Initial deg | Final deg | Change in deg |
|-------|-------------|-----------|---------------| 
| UVHWP | -0.899      |           |               |
| AHWP  | 8.533       |           |               |
| AQWP  | -4.523       |           |               |

# 2/28/2024
MP: Lev

1600 Laser on, going to start by recalibrating AQWP (trial 3) to see if it is centered around zero (which it should be as config.json was updated). Got near -8.861 off the current calibration, so I will go through and recalibrate AQWP more carefully, (trial 1 and 2) going from -15 to -5 and then finer if necessary. Got -8.226 deg from current home. 

1650 Now trying to recalibrate the UVHWP to see how off it is from the current home (trial 3). Got -1.253 which is in the range of the checka minisweeps so I'll use that for now. 

1700 Will now replace AHWP and check its calibration to ensure its good before continuing to checka (trial 4). +4.17 deg in first coarse trial, will run a finer trial around that to home in (trial 5). Data from that trial was a bit all over the place so I'll use trial 4. This seems ok because they're both still in the range of <0.5 deg from each other. Upon looking at the table above, I may have swapped the QP and HP values on 2/26 on accident--now will run checka to see. 

1530 Updated the values into the below table and config.json.
| Motor | Initial deg | Final deg | Change in deg |
|-------|-------------|-----------|---------------| 
| UVHWP | -1.253      |           |               |
| AHWP  | 4.17       |           |               |
| AQWP  | -8.226      |           |               |

1800 Ran into same maxfev error as last time but for the UVHWP, signaling the calibration of UVHWP is really off for some reason. Goal for next time is to recheck the UVHWP calibration, and also to confirm with Alec that I'm doing this right!

# 2/29/2024 
MP: Lev

1200 Laser on, goal for today is to get through checka. Will start by rerunning checka with larger bounds on the mini sweeps (-8 to 8 as opposed to -4 to 4) to mitigate the maxfev error (sweeps4). Deg update 1 was:
UVHWP Update: -1.253 + -0.019 -> -1.272
AHWP Update: 4.170 + -10.564 -> -6.394
AQWP Update: -8.226 + 17.834 -> 9.608.

1250 Will now run checka again (sweeps 5) w/ new offset variables. Will keep larger sweep for this trial. This time got 
UVHWP Update: -1.272 + 0.139 -> -1.133
AHWP Update: -6.394 + 12.109 -> 5.715
AQWP Update: 9.608 + -31.386 -> -21.778.

1330 Latest change led to maxfev error on AQWP so I will split the difference of movement in AQWP and retry. Have to go to office hours though so will be back in one hour. 

1430 Back, retrying with (sweeps7) UVHWP Update: -1.272 + 0.139 -> -1.133
AHWP Update: -6.394 + 12.109 -> 5.715
AQWP Update: 9.608 + -31.386 -> -10.
This time AHWP maxfev error -- broadening the mini sweeps to -20 -> 20 deg in hopes of finally calibrating this! (sweeps 8). No maxfev error, got UVHWP extrema: -0.26+/-0.05
UVHWP Update: -1.133 + -0.257 -> -1.390
AHWP Update: 5.715 + -11.602 -> -5.887
AQWP Update: -10.000 + 22.799 -> 12.799. Will update all and run again at 20. (sweeps 9).
UVHWP Update: -1.390 + -0.032 -> -1.422
AHWP Update: -5.887 + 13.667 -> 7.780
AQWP Update: 12.799 + -24.997 -> -12.198

1700 This did not work at all, the numbers just keep hopping around semi-randomly. My goal for next time is to begin calibration carefully from the beginning! Which is not great but hopefully I can get it to work.

#3/5/2024
MP: Lev

1900 Laser on, will be moving all current calibration data for aqwp, uvhwp, and ahwp to respective folders "failed_calib_feb". Began calibration by remove AHWP.
Got -12.789 deg.

2020 Going for UVHWP. Got +0.613 deg.

2100 Went for AHWP and first go (coarse sweep -20 - 20) looked like a delta function--honestly unsure why, but I'll come back in next time, try a larger area sweep, and if that is still being weird begin trouble shooting. So far promising I suppose. Also checked voltage readings on power supply just to ensure that was functioning and the multimeter read correctly so all is ok there. [3/7/2024 update, realized i had the wsweep set -20 to -20 so that was why it was so weird.]

# 3/7/2024
MP: Lev

1430 Laser on, realized I didn't update the AHWP to use the newly measured zero for aqwp, so will do so now and run a -30 to 30 sweep. Notice apt user was not recognizing 3/4 of the wave plates but I will ignore it for now and see what happens if I just ignore it. Also updated the new calibration for the UVHWP into config.json instead of waiting until end--previous was -1.390. I'm unsure if the +0.613 deg from two days ago is what it should be or what the change should be, so I'm going to try change such that new value in congif will be -0.777.

1325 AHWP ran successfully, got +5.220 deg. Prev. calib for ahwp was -5.887, so
will update to -0.667 and rerun ahwp once more--this should tell us if it is best to update from prev. calibration value or if what the program returns is the new absolute value. 

1335 IMPORTANT RESULT -- the value the plots return for all calibration files (except checks I believe) are meant to be updated from the config!!! So, the step at 1325 today was the right thing to do! Will now run checka with updated variables!
Note aqwp was at +12.799 deg in config so now will be 0.01 deg.

# 3/14/2024
MP: Lev

1200 Laser on, everything looks fine post-power outage; 904 was not showing in apt user but showed after power cycle. First will run checka but am bumping the number of measurements to 30 from 20 to hopefully get a more accurate read. Ran into typ error when initialing motors; fixed by going into ello and fixing the comm addresses. Note I had to disconnect all pieces manually from board, the plug one in and connect to COMM in Ello and set its address, then disconnect in Ello and plug in a second element and connect to ello and sets its address, and so on. For more info ctrl-F type (fixed above).

1250 Trial 4 of checka run. Got maxfev error on UVHWP and assuming it is due to everything getting messed up during the power outage; will try redoing uvhwp calibration (and pray that does not mess up everything else!). Will start from trial 4. Ran into could not access COMM7 error, fixed by ending VSCode process in task manager. Got -22.3 on uvhwp, will update in config and run a ahwp to make sure thats reasonable pre-checka.

1445 ahwp trial 5. Still shwoed 0 so now going for checka!

1500 Checka sweeps5 (input for sweeps5 in config in table below.) 

1530 Success!! Furthermore, everything is $\leq 0.1$ deg change! Goal for next time is to ensure we can create a VV state and finish the whole calibration! See total change from checka below.

# 3/15/2024
MP: Lev

1200 Goal for today is to finish calibration! Going to start by rerunnning checka once more (trial 6) to ensure it all is working fine. 

1500 After a bunch of checkas it is getting closer to a <0.1 deg difference but not quite there; will come back in tomorrow and do more trials. 

# 3/17/2024
MP: Lev

1230 Laser on, rerunning checka. Success, will now ensure we can create VV state successfully. Data below on final changes post checka. Measv returned 45 deg within uncertainty, so Alice's side is fully calibrated! Now for Bob.
| Motor | Initial deg | Final deg | Change in deg |
|-------|-------------|-----------|---------------| 
| UVHWP | -22.3     |    -22.142      |  0.158             |
| AHWP  | -0.529        | -0.401          |    0.128           |
| AQWP  | 0.210       |   0.371        |            -0.161   |

1400 Removed BHWP and BCHWP as per calibration instructions and fine swept BQWP (trial 1/2) to get +98.800 deg. Now reinstaling BCHWP and sweeping (trial 1 first) to get +60.051 deg.
1440 Recalibrated QP by retroreflection to be new home offset of 316.494 deg (got the negative of -43.506 deg but ello doesn't like negative home offsets).
1615 Now for BHWP, got +10.432 deg. Checkb failed on bhwp so repeating around 0 after updating config.json and got -5.225.
1700 Dinner time, goal for next time is to run checkb and calibrate the pcc, therefore finishing calibration!

# 3/19/2024
MP: Lev

1200 Laser on, goal for today is to finish calibration. Repeating all my work from 3/17 except the hwp bc I realized I haven't been updating config from original value. For qwp got 98.80-91.354=7.446 deg. Checked around zero, good.

1250 Reinstalled bhwp, checked and calibration was fine about zero. Reinstalled bchwp, got similar results but updated by + 6 deg.
Note (1630) that I installed them in the wrong order here!

110 Running checkb but must go to class; will return post class. 

1430 Came back between classes; checkb returned large differences which is unfortunate; rerunning.

1600 Back, even worse. Will begin calibration of Bob's side from the beginning :(. Starting by removing BCHWP and BHWP. Got new BQWP to be 7.478 and then finding BCHWP (post-reinstall) to be -30.519.

1900 Left for 2 hrs to get dinner / go to work; reinstalled bhwp and now calibrating. Got -6.197, now for checkb trial 1. Bob's side has decided it hates me and checkb does not give me anything good. I will retry Wed or Thurs, trying to recalibrate bob's side from the start. Saw Alec did this in the past from the start so will follow exactly what he did next time (although i thought I did? unsure.)

# 3/21/2024
MP: Lev

1425 Laser on, will carefully document calibration of right side, but first running checka to ensure that is still good. Checka fully failed, meaning somehow alice's plates got uncalibrated... how? I do not know. I am going to carefully recalibrate everything.

### Calibrating AQWP
Began by carefully removing bobs pbs and placing it on table. Sweep 0 UVHWP was set to 0 deg in file and -22.250 in config. Ran a course sweep then fine sweep around zero. After that, it actually showed AQWP as calibrated? Mostly at least, 0.627 move but due to low resolution leaving current calib. 

### Calibrating UVHWP
Going to start with a semi-fine sweep because this should already be correct. I realize at this point that AHWP should also have been taken out. I genuinely don't know if that messed up the aqwp calibration (I'm assuming not bc it calibrated around current 0) but alas. Was already running course uvhwp sweep when realized so will see what happens with the uvhwp. Got 1.6520068984667955+/-0.025013947644061018.

I want to re-run aqwp calib (course sweep for vibes) to see if removing ahwp makes a difference. It did not! That's wonderful! Now doing the same with the uvhwp. Got nearly, the same, 1.648 off (same as before so will adjust config from -22.250 to -20.602)

### Calibrating AHWP
First carefully put it back into the setup, then ran a course then fine sweep. Less than a 0.1 change so should be fully calibrated; what in the world happened with checka then??

### Running checka
Running 16 points between -4 and 4 deg with 5x3s measurements. Possible reason for sweeps earlier today being bad was Bob's PBS being in, and also bob's wave plates having weird settings in config. 
It worked perfectly, although wasn't able to get within 0.1 on the uvhwp; will retry next time. 

# 3/23/2024
MP: Lev

Once again and hopefully for the last time, goal today is to finish calibration. Will continue following Alec's notes from 7/17/2023 and being very careful with everything.

### Running checka
First step is to re-run checka to ensure all is good and hopefuly get UVHWP calibration within 0.1 deg uncertainty. Running a slightly more comprehensive sweep of 32 points between -4 and 4 deg with 5x3s measurements (sweeps 4). Got up to sweeps7 but uvhwp is refusing to actually calibrate within 0.1, so I'm going to leave it where it is at this moment (-20.916 deg) and try out measv to see if that is functional. Measv is perfect so I will consider alice's side calibrated.

## Bob's Side
First step is to take out the CHWP and BHWP and put back in Bob's PBS. 

### BQWP
Began by running course calibration around where BQWP is currently set to. Got +7.525 +/-0.024874019051229424 deg.

### BCHWP
Perfect try one.

### BHWP
First carefully reinstalled then with a course then fine sweep got it to be -6.096 +/-0.026597619118315742

### Running checkb
Now for the moment of truth, running checkb. BHWP and BCHWP were great with <0.1 deg but BQWP was off by 0.5 deg, will rerun with new config stuff next time I'm in.

# 3/27/2024
MP: Lev
### checkb
1600 Laser on. Am attempting one more trial of checkb with -4 to 4, 16 sweeps at 5x3, to ensure that calibration did not somehow get messed up entirely. BHWP extrema: -0.279+/-0.018
BHWP Update: -6.085 + -0.279 -> -6.364
BQWP extrema: 0.004+/-0.026
BQWP Update: 6.998 + 0.004 -> 7.002
BCHWP extrema: 0.282+/-0.018
BCHWP Update: -30.524 + 0.282 -> -30.242
Updating and running once more.BHWP extrema: 0.284+/-0.016
BHWP Update: -6.364 + 0.284 -> -6.080
BQWP extrema: -0.523+/-0.032
BQWP Update: 7.002 + -0.523 -> 6.479
BCHWP extrema: -0.253+/-0.019
BCHWP Update: -30.242 + -0.253 -> -30.495
Updating and running once more (not changing ) Got
BHWP extrema: -0.546+/-0.019
BHWP Update: -6.080 + -0.546 -> -6.626
BQWP extrema: 0.487+/-0.028
BQWP Update: 6.479 + 0.487 -> 6.966
BCHWP extrema: 0.516+/-0.010
BCHWP Update: -30.495 + 0.516 -> -29.979. It just somehow keeps getting worse? 
What I am going to try is going back to my original values right after calibration, put those in the config, then run a slightly better check.

Could not get it within 0.1, this is a rabbit hunt. I will take out the wave plates next time and try recalibrating the b side. :(

# 3/28/2024
Reattempting checkb once; if it does not work I will attempt to fully recalibrate bob's side from the beginning. (sweeps6). Close so trying again, (sweeps7).
This did not work at all and I just wasted an hour. BQWP seems well calibrated so I will just take out BHWP and try recalibrating BCHWP with a very fine sweep around zero. After tring to do so and finally running checkb, BQWP is 0.5 off 
BHWP extrema: 0.034+/-0.017
BHWP Update: -6.279 + 0.034 -> -6.245
BQWP extrema: -0.559+/-0.026
BQWP Update: 7.525 + -0.559 -> 6.966
BCHWP extrema: -0.034+/-0.017
BCHWP Update: -30.618 + -0.034 -> -30.652.

Gonna mess with things a bit, #s pre messing
            "type": "Elliptec",
            "port": "COM7",
            "address": "A",
            "offset": -30.618
        },
        "B_HWP": {
            "type": "ThorLabs",
            "sn": 83811901,
            "offset": -6.096
        },
        "B_QWP": {
            "type": "ThorLabs",
            "sn": 83811646,
            "offset":  7.525.
Met with Alec and learned three things: 1. Run manager in python terminal using from lab_framework import Manager, then m  = Manager('./config.json'), then using goto commands with m., and 2. I can try doing -15 -> 15 or -10 to 10 for calibration bc that might be better! Also that if not, 0.2 is totally fine. I wasn't able to get it closer after the last few trials so leaving it at  "B_C_HWP": -30.371 },"B_HWP": -6.264 },"B_QWP":   6.762.

Goal for next time is to calibrate the phi state and also the pcc -- will do so once alec gets back to me on how!

# 4/4/2024
MP: Lev

Began by running phase_finding.py in phi_plus folder--then used plotting.py after subtracting pi from all points above zero and adiing pi to al points below--the graph looked wack without and we did this bc we want the data to be the same at 0 and pi (for some reason? unsure why). Got -25.3742 as the QP angle (resuls under phi_sweep_adj.png). Input this angle into ratio_tuning and ran that; it returns what the UVHWP for phi plus should be set to. Input this value and the QP angle value into the config (QP:-24.1215 -> -25.3742, UVHWP 65.39980 -> 88.265). Created phi_check to make sure you're creating the phi state, turns out we are not and also cannot trust results bc HH dne VV. Will troubleshoot later.

# 4/10/2024
MP: Lev

As something clearly went wrong last time with calibrating the phi plus state, I am trying to get a handle on what the entire phi_plus folder is doing if run correctly.

We want to find phi plus which is the state 1\sqrt(2) HH + VV. The general form of a state we can make is psi = $cos(\theta)\ket{HH} + e^{i\phi}\sin(\theta)\ket{VV}$.

We find that $tan^2(\theta)$ is the ratio of VV to HH and we can find phi in terms of multiple expectation values as in the calibration doc. The steps then go.
1. Get a correct phase difference between these two terms. We thus set the UVHWP to 45 deg then perform a sweep of the quartz plate to measure the phase as a function of quartz plate angle. This is done in phase_finding.py. It gives when the phase shift between the two terms is equal to zero.

2. Knowing this, we can fine tune the ratio of hh to vv to
 get 1/2. We can use the theta equation in pg. 14 of calib doc and find the angle the uv hwp is at when theta = 45 gives an even superposition of hh and vv with no phase difference. This is done in ratio_tuning.py. We then put the returned uv hwp angle into config.

3. We then want to calibrate the PCC by creating phi plus, then sweeping PCC in AD DA basis looking for when phi is minimized.

# 4/11/2024
MP: Lev, Lynn

Today we identified the UVHWP, BCHWP, and PCC as having lost calibration, and then manually calibrated the UVHWP (seeing that all of Alice's and Bob's side were good) by checking for maximal counts in hh basis and minimal in vv basis, and that at 22.5 deg there are equal ish counts in hh and vv basis. Output in vv_sweep_go.csv and uvhwpcheck1.png. The goal forward is to start from beginning of pcc/phi plus calibration next time!

# 4/15/2024
MP: Lev

Goal for today is to repeat pcc/phi calibration for reasons described above. Will be methodical in doing so to ensure followability if it doesn't work.

### phase_finding.py 
Run phase finding.py taking a +- sweep of the QP around where we expect it to be--our goal is to find where the phase shift between the two terms equals zero. Got -20.6134 after correcting data by adding pi to negative values and subtracting pi from positive values. 

### ratio_tuning.py
Inserted the -20.6134 into the QP angle for ratio_tuning and ran with a 80 to 50 deg sweep as I don't want to miss valuable data by going too exactly too fast. I start with TRIAL3 as to not lose last calibration's data. It ran and I got Pi/4 at 69.10217807167456 deg. We now update the config.json for phi plus with these two values!

Attempted to run pcc_sweep but ran into nothing errors (sent instruction runtime error) so must leave as it is 10 pm. Will resume next time!

# 4/25/2024
MP: Lev
It has been awhile as I was gone at nationals--goal for today is to calibrate pcc.

1030 Will rerun pcc_sweep now. After a lot of trouble shooting with the 'sent instruction runtime error', solved it by unpowering the QP it was bugging out on and then recalibrating. 

1545 Found QP angle = 0.45 deg at minimum phi which is good! Rebuilt the pcc_sweep file as well as a new plotting_pcc file and got good initial results, will come back soon to get closer on those! Found purity of ~3000 which is not good at all bc it should be 1. 

# 4/29/2024
MP: Lev

Laser on at 12. Today I will investigate why the purity values are so high. I found that at pcc = 0, we are in the state y = 1\sqrt(2)
(DA + AD) meaning y = 1\sqrt(2)(HH - VV). To remedy this I manually checked at 40 deg intervals of the UVHWP where we have HH>>Hv, RR<< RL, and AA >> AD. This could not be found because at everyu 40 degree integral, either the AA or RR inequality was not satisfied. I am truly unsure. This took awhile to check every 40 deg shift and results are on my iPad. 

2030 I built uvhwp_calib which takes over an hour to run, which errored out during plotting and did not return good data, as an attempt to figure out where the purities of all three inequality pairs would align. I think I need to figure out the theory a bit more before continuing. Will be back Thursday likely. Also must replace nitrogen.

# 5/14/2024
MP: Lev, Paco, Stu

Laser on at 9, fixed weird basis measurment, i.e. that we had HH>>>HV, DD>>>AD, but not RL>>>RR--> BQWP was 90 deg off! Woohoo!! Obtained 92.7% purity at 2.894736842105263 pcc deg. Updated in config. Repeating phi plus calib for iterative improvement of purity.

### phase_finding.py
Started with off HH/VV counts, manually changed -153.9970823589124 to -154.85 as that returned better counts. Now re-running phase_finding. Obtained a zero at , changed from -20.6134. Goal for tomorrow is to use this to recalibrate phi_plus.

# 5/15/2024

# ratio_tuning
Found -154.90620864065068, changed in config for phi_plus from -154.85.

# pcc_sweep
Found purity of 95.2% at exact same 2.89 deg value of PCC!

# accurate purity check / redoing phi plus calib post bchwp input
Modified pcc_sweep.py to sweep at the pcc calibrated point much more accurately (10 measurements, 5 secs each) and found purity to be 9473+/-0.0006. Reinstated BCHWP and found it to be calibrated at -30.949568. 
Attempted to run lev_phi_state_1 but ran into fitting error (M=5, N=4)--realized it is due to the addition of the BCHWP changing the phase difference by -1 on H. Must repeat ratio tuning at 45 degrees past our current calib. Added 45 deg to uvhwp calib for phi_plus then rerunning phae finding and ratio tuning. Phase finding gave -20.5717, changed in config from -21.5377. Ran ratio tuning and obtained -111.39816886500309 which was then changed from -112.

Goal for tomorrow is to purity check this and then run lev_phi_sweep_1.

# 5/16/2024
Found purity at start to get 0.9496+/-0.0011.

# 5/22/2024
MP: Lev, Paco, Lynn

We did not lose calibration! Which is incredibly exciting overall, as it means that the UPS worked. We did notice that the BBO got knocked a slight amount leading to unequal VV/HH measurements, which was fixed by maximizing VV counts in the VV basis in phi_plus with QP at 0, and vice versa (45) in HH. Goal for tomorrow is to recalibrate phi plus as current purity is 93.61. 

# 5/23/2024
MP: Paco, Lev, Stu

Recalibrating phi plus, first run got a nice 94.51 purity, running again. Obtained 94.44 percent with results converging (meaning same PCC value).

# 5/24/2024
MP: Paco, Lev, Stu

Goal for today is to get data on the psi+- bell state at eta = 45 deg. We changed QP to -24.7869 from -21.01976. Changed UVHWP from -112.04979023180509 to -114.13271291632401.

Changed and unchanged the rotation of the PCC to maximize counts, measuring in HH basis with QP -> 0 and UVHWP -> 45, other settings at phi_plus calibrated. Got 3033+/-17 counts as maximum and left it there. 

Upon recalibration we changed the UVHWP to -111.60380473889805 