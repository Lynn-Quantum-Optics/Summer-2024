# Basic Usage

```py
import manager

# initialize the manager
m = manager.Manager()

# print out list of available motors
print(m.motors)

# move a motor to pi/4
m.UV_HWP.rotate_absolute(3.14 / 4)

# take 5 data points, 3s of collection per data point
m.take_data(5,3)
# these 5 points will be averaged and appear as one line in the output table

# move a different motor to pi
m.UV_PCC.rotate_absolute(3.14)

# take more data
m.take_data(5,3)

# write out
m.close()
```

The manager handles all data collection by writing it out to a CSV file as you are performing the experiment, so as to save partial data even when an experiment goes awry.

# Specifics About Data Collection

We get updates from the CCU every 100ms, so you can only select sample periods of up to 0.1s, any less and it will be rounded up. If asked to collect 5 samples with a sample period of 3s (e.g. `m.take_data(5,3)`), the manager will call on the CCU manager with `self._ccu.get_count_rates(3)` five seperate times. It will then take the average of those five runs and compute the SEM for each channel seperately, which are promptly written out to the csv.

# Backend
One day, some (or all) of this will break and it will need to be debugged. Here are my (alec's) notes on the backend for these components to assist future generations.

## FPGA CCU

[More documentation on this controller can be found here](http://people.whitman.edu/~beckmk/QM/circuit/circuit.html).

The CCU communicates with the PC through a serial port. It transmits information in packets that come every 0.1s, however, there is a significant buffer, so to get live data you should flush the port then wait for the termination byte. Speaking of, let's look at the format of a packet.

Each packet is 41 bytes long, broken up into 8 chunks of 5 bytes each (one for each counter, in order C0 -> C7), with the last byte reserved for a termination byte (`0xff`) marking the end of the data packet.

    [5 bytes encoding counter 0]
    [5 bytes encoding counter 1]
    [5 bytes encoding counter 2]
    [5 bytes encoding counter 3]
    [5 bytes encoding counter 4]
    [5 bytes encoding counter 5]
    [5 bytes encoding counter 6]
    [5 bytes encoding counter 7]
    [1 termination byte]

For each counter, the 5 bytes encode some sort of a multi-byte
unsigned integer. To avoid clashing with termination bytes, only the
first (right-most) 7 bits of each byte are used (the 8th bit is always a
`0`). The bytes are encoded in reversed order, so that the first byte encountered corresponds to the least-significant binary digits in the number encoded.

For example, the number `2101` is `100000110101` in binary, and would be sent as

    00110101 01000000 00000000 00000000 00000000

An example representation of a data packet containing counter
values [2718, 281828, 4, 59045, 235, 360, 2874, 71352] is
(first row highlights delimiter ``0`` bit columns as ``0`` and
actual data bits as ``x``)::

    0xxxxxxx 0xxxxxxx 0xxxxxxx 0xxxxxxx 0xxxxxxx
    -------- -------- -------- -------- --------
    00011110 00010101 00000000 00000000 00000000
    01100100 00011001 00010001 00000000 00000000
    00000100 00000000 00000000 00000000 00000000
    00100101 01001101 00000011 00000000 00000000
    01101011 00000001 00000000 00000000 00000000
    01101000 00000010 00000000 00000000 00000000
    00111010 00010110 00000000 00000000 00000000
    00111000 00101101 00000100 00000000 00000000
    11111111

Note the termination byte `0xff` shown at the end of this example communication.

## Elliptec Motors

These motors also communicate via a serial port. For these motors, the ThorLabs documentation is quite helpful, especially when debugging commands.

- [Documentation for all Elliptec interfacing](https://www.thorlabs.com/Software/Elliptec/Communications_Protocol/ELLx%20modules%20protocol%20manual.pdf)
- [Documentation for bus splitter](https://www.thorlabs.com/drawings/bd903c049cf9699e-5B57E5EE-C0C4-BF03-0D58E9F183BDA628/ELLB-Manual.pdf)
- [Documentation for optic rotator](https://www.thorlabs.com/drawings/5c17c9ccc1cd5d4e-BD11B7E8-9CBB-97EE-5A1878A0E98EDE53/ELL14K-Manual.pdf)
- [Documentation for stage rotator](https://www.thorlabs.com/drawings/5c17c9ccc1cd5d4e-BD11B7E8-9CBB-97EE-5A1878A0E98EDE53/ELL18-Manual.pdf)

There's not much to say about the interfacing outside of the documentation, but here are some of my notes from working on it:
- The `in` command gets a _travel_ (`travel`) parameter (in degrees, for us) and a _pulse/m.u._ (`ppmu`) parameter. In my mind, pulse per measurement unit should be number of pulses we send to get the device to move one measurement unit (one degree). However, from trial and error I have learned that the `ppmu` we get back is actually the number of pulses for the _full range of travel_, and so we have to calculate the true `ppmu = ppmu / travel`.
- You need to be sure that your code can both send and recieve 2's compliment encoded negative numbers. Luckily, there is a really easy way to get the 2's compliment in python. For an eight bit number `x`, the two's compliment is just `(x ^ 0xff) + 1` using the XOR operator `^`.
- The finnicky thing is that these motors use byte-encoded ascii-encoded hexidecimals to send all numerical information. It's just something you have to work with.
- For the serial object, note that the `s.readall()` method will timeout, whereas the `s.read(n)` will wait indefinitely to read `n` bytes.

## ThorLabs Motors

Thank GOD for the ThorLabs motors, they have their own python interface through the package `thorlabs_apt`. I haven't found the need to look at any documentation here yet, it's all been pretty self-explainatory. I did learn two lessons though:
- Calling `motor.move_home()` just spun the motor in a circle forever. Not exactly sure why, but I think it has to do with the blocking behavior, I think there is a more specific function like `motor.done_moving_home` or something, but instead of using that we're just going to move it to position zero.
- When having trouble connecting motors, the following lines in a python terminal can help debug
```py
>>> import thorlabs_apt
>>> thorlabs_apt.list_available_devices()
[(<device kind>, <serial num>), ...]
```

# Contributing Authors
- Alec Roberson (aroberson@hmc.edu / alectroberson@gmail.com) _2023_
- Ben Hartley (bhartley@hmc.edu) _2022_ (components of `motor_drivers.py`)
- Kye W. Shi (kwshi@hmc.edu)  _2018_ (components of `ccu_controller.py` and portions of this `README.md` were taken from their documentation)
