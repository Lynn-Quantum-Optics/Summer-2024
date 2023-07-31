// this is the code that is running on the laser monitor
// NUCLEO-32 microprocessor. it is compiled on 

#include "mbed.h"
#include <vector>
#include <cstdlib>
#include <string>


// assign pins and such

DigitalOut LED(LED1); // main led

AnalogIn PDM_INPUT(A1); // photodiode monitor pin
AnalogIn LDC_INPUT(A2); // laser diode current monitor pin
AnalogIn TEMP_INPUT(A5); // temperature input pin

BufferedSerial PC(USBTX, USBRX) ; // 9600 baud , 8 -bit data , no parity

// update period for the monitor
#define DELAY 100ms
#define NUM_MEAS 10

// message termination sequence
static const size_t TERMINATION_SEQUENCE_LENGTH = 4;
static const uint8_t TERMINATION_SEQUENCE[TERMINATION_SEQUENCE_LENGTH] = {0xff, 0xff, 0xff, 0xff};

// benchmark voltage/temperature readings (from laser documentation)
static const int N_BENCH = 48;
static const float VOLT_BENCH [N_BENCH] = {
    0.950996483,
    0.988555222,
    1.026862732,
    1.065891044,
    1.10560988,
    1.145986745,
    1.18698702,
    1.228574089,
    1.27070946,
    1.313352918,
    1.356462679,
    1.39999556,
    1.44390716,
    1.488152041,
    1.532683927,
    1.5774559,
    1.622420602,
    1.667530436,
    1.71273777,
    1.757995134,
    1.803255416,
    1.848472054,
    1.893599212,
    1.938591962,
    1.983406439,
    2.028,
    2.072331362,
    2.116360728,
    2.160049906,
    2.203362406,
    2.246263529,
    2.288720439,
    2.330702222,
    2.372179931,
    2.41312662,
    2.45351736,
    2.493329247,
    2.532541399,
    2.571134938,
    2.609092965,
    2.646400526,
    2.683044569,
    2.71901389,
    2.754299082,
    2.788892466,
    2.822788029,
    2.855981348,
    2.888469518
};
static const float TEMP_BENCH [N_BENCH] = {
    -0.003050765,
    0.997046962,
    1.997145503,
    2.997244855,
    3.997345018,
    4.997445987,
    5.99754776,
    6.997650336,
    7.997753711,
    8.997857883,
    9.99796285,
    10.99806861,
    11.99817516,
    12.99828249,
    13.99839061,
    14.99849952,
    15.9986092,
    16.99871966,
    17.99883089,
    18.9989429,
    19.99905568,
    20.99916922,
    21.99928353,
    22.9993986,
    23.99951443,
    24.99963102,
    25.99974836,
    26.99986646,
    27.9999853,
    29.00010489,
    30.00022523,
    31.00034631,
    32.00046813,
    33.00059068,
    34.00071398,
    35.000838,
    36.00096275,
    37.00108824,
    38.00121444,
    39.00134137,
    40.001469,
    41.00159739,
    42.00172648,
    43.00185628,
    44.00198678,
    45.002118,
    46.00224992,
    47.00238255
};


// function to toggle the led state
void toggleLED() {
    LED = !LED.read();
}

// function to get a linear interpellation of the temperature
float linInterpTemp(const float& voltage) {
    // check for bottoming out
    if (voltage < VOLT_BENCH[0]) {
        return TEMP_BENCH[0]; // min temp
    }
    // counter for loop
    int i = 0;
    // loop until we find the range for the voltage
    while (1) {
        if (i == N_BENCH - 1) {
            // reached end of loop array -> max temp
            return TEMP_BENCH[N_BENCH - 1];
        } else if ((VOLT_BENCH[i] < voltage) && (VOLT_BENCH[i+1] > voltage)) {
            // found the range for the voltage -> exit loop
            break;
        } else {
            ++i;
        }
    }
    // calculate a linear interpellation based on benchmarks
    float x = (voltage - VOLT_BENCH[i])/(VOLT_BENCH[i+1] - VOLT_BENCH[i]);
    return (TEMP_BENCH[i+1] - TEMP_BENCH[i]) * x + TEMP_BENCH[i];
};

// function to send floating point values over the USB
void sendFloat(const float& x) {
    // make a byte array
    uint8_t bytes[sizeof(x)];
    // copy the data of the float into the array
    std::memcpy(bytes, &x, sizeof(x));
    // write the bytes to the serial port
    PC.write(bytes, sizeof(x));
}

// function to read a pin's voltage in a more intelligent way
float readmV(AnalogIn& a) {
    float sum = 0;
    for (int i = 0; i < NUM_MEAS; ++i) {
        sum += a.read() * 3300;
    }
    return sum / NUM_MEAS;
}

// function to get and send the temperature reading
void sendTemp() {
    float v = readmV(TEMP_INPUT)/1000; // get voltage reading
    float t = linInterpTemp(v); // convert to temp
    sendFloat(v); // send it
}

void sendPDM() {
    float v = readmV(PDM_INPUT)/1000; // voltage from photodiode monitor
    sendFloat(v); // just send the voltage
}

void sendLDC() {
    float v = readmV(LDC_INPUT); // 1 mV = 1mA
    sendFloat(v); // send the reading
}

void sendTerm() {
    PC.write(TERMINATION_SEQUENCE, TERMINATION_SEQUENCE_LENGTH);
}

// main function

int main() {
    // main loop
    while(1) {
        // send all the measurements and termination charachter
        sendPDM();
        sendLDC();
        sendTemp();
        sendTerm();
        // toggle led and wait the delay
        toggleLED();
        ThisThread::sleep_for(DELAY);
    }
}
