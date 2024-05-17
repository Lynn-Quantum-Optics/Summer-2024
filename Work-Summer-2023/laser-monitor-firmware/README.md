# Laser Monitoring

The project here is to be able to collect the data made available by the output pins of the laser in order to better understand and control for these factors in our experiment. The setup I have designed uses a `NUCLEO-32 STM32L432` microprocessor to perform the analog-to-digital conversions before passing these data onto the main computer.

The code I have written for the microprocessor is included in this folder (`main.cpp`) for completeness' sake, however it cannot be compiled locally since it relies on the `MBED` OS version 6 (or greater?). To compile it, you would need to go to [the keil studio cloud](https://studio.keil.arm.com).

## Temperature Data

The temperature data in the code was gathered from the original specifications from the manufacturer of our laser, dated November 2005. Furthermore, since the voltage will never be an exact value from the table, we apply a linear interpellation in between the tablulated voltages to obtain our experimental voltage.

## Pin Connections

Below is a table of pin connections for the hardware, given the firmware that I have written.

|Chip Pin #|Laser Pin #|Purpose|
|-|-|-|
| `A1` | `4` | Photodiode monitoring voltage |
| `A2` | `7` | Laser diode current monitoring voltage |
| `A5` | `9` | Temperature monitoring voltage |
| `GND` | `14` | Ground signal for voltage monitoring |

## Communication Scheme

The microprocessor reads the voltage signals in 32 bit (4 byte) floats, which then may be further processed (to say, convert to a temperature) but will remain 32 bit floating point numbers. To send these to the main computer, the code packages each floating point number into four bytes (little endian, i.e. least significant bits first) which are then sent along the serial port of the USB cable to the main computer. After sending the three relevant floating point values, a termination sequence of 32 1s is sent. A packet then consists of 16 bytes as follows:

    4f b1 61 3f // photodiode monitor voltage
    1e d1 f9 44 // laser diode current monitor voltage
    ba cc 1d 41 // temperature interpellation
    ff ff ff ff // termination sequence

The microprocessor will send one of these packets every 0.1 seconds, though it should be noted that **the accuracy of a single has significant uncertainties**. See the next section for more info.

## Calibration and Testing

To make sure everything was in working order, I hooked each input line up to known variable voltage supply and confirmed the output aligned with what we would expect. This is why we aren't using the `A0` pin, which I found consistently reads 50mV less than the applied voltage. Besides that, all seems to be working just fine for now.

Through these experiments and testing I discovered that the uncertainty of a single reading is pretty damn huge, sometimes it will be +/- 200mV, but generally stays less that +/- 100mV. To combat this, while still allowing for our device to update every 0.1s, I have employed an averaging of measurements built into the firmware. Every 0.1s, the microprocessor takes 10 measurements for each voltage signal, averaging them before passing them onto the main computer.

## Example

An example piece of code, `example.py` is included alongside this readme. This code is great for debugging and testing. It just flushes the buffer of the serial port, and then loops 
