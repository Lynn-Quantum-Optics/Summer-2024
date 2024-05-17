
I came up with these offsets for the minima (on top of the offsets above)
- UVHWP minimia at -0.359 +/- 0.021 **UPDATING OFFSET** -1.029 -> -1.388
- AHWP minima at 0.028 +/- 0.028 **LEAVE AS IS**
- AQWP minima at 0.27 +/- 0.05 **UPDATING OFFSET** -8.704 -> -8.434
So I updated the offsets above in the config file and now I'm running the same check again. I moved the data from the first check into `checka/check0`

Now we have gotten the offsets down further but not quite to my liking. So I'll make the following updates
- UVHWP: -1.388-0.112 -> -1.500
- AQWP: -8.434+0.05 -> -8.384
- AHWP: -9.470+0.167 -> -9.303
I moved the data for that run into `checka/check1`. For subsequent tests, I'm lowering the sampling to 5x3s samples instead of 5x5s, to speed up these iterations.

I did one check here, but I'm actually going to delete the data I took and do one quick thing which is eliminate all extra light on the other side of the room by covering all the exposed LEDs. My gut feeling is that this won't actually help all that much, but I hope it can lower the uncertainties a bit. Then I made the following updates
- UVHWP -1.500+0.156 -> -1.344
- AHWP -9.303+0.052 -> -9.251
- AQWP -8.384 + 0.08 -> -8.304
based on the data in `checka/check2`

For the next experiment, I'm lowering the sampling even further to 5x1s, but I'll take 21 data points per sweep instead of 11. This `check3` revealed great data with relatively high uncertainty. Without modifying the config file, I'm going to re-take this data with 21 5x3s samples between +/- 2 degrees (down from +/- 3).

The data taken here is in `check4` and prompted the following set of changes
- AHWP -9.251 + 0.031 -> -9.220
- UVHWP -1.344 - 0.141 -> -1.485

Retook data and feeling a bit defeated by how the HWPs seem to just be bouncing back and forth. The data is in `check5` and suggests that we should use an offset of -0.111 for AHWP. Going to try one more thing -- updating the values not based on the offset it asks for, but half of the offset. So
- AHWP -9.220 - 0.111/2 -> -9.276
- AQWP -8.304 + 0.16/2 -> -8.224
I'm keeping the 5x3s sampling but only taking 16 samples in the -2 -> +2 degree range for the next round.

- UVHWP -1.485 + 0.13/2 -> -1.420
- AHWP -9.276 + 0.075/2 -> -9.239
- AQWP -8.224-0.23/2 -> -8.339