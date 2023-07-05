## 7/5/23
MP: A
Started warming up the laser at 0823.

Alec here: I just came back in to do my test. I don't know who completed the rest of the setup today but I am noticing the nitrogen tank is _critically_ low (~300psi).

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
