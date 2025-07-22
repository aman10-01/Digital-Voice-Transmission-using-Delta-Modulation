**_Digital Voice Transmission using Delta Modulation_**
**Overview**
This Python project demonstrates Delta Modulation-based Digital Voice Transmission with noise simulation and reconstruction. It processes a voice signal by performing:

Delta Modulation

Noise Addition (Gaussian)

Demodulation

Low-Pass Filtering (Butterworth)

Signal Reconstruction

The output includes a reconstructed audio file and comparative signal plots.


**Features**
Converts stereo to mono for consistent processing.

Performs delta modulation with adjustable step size.

Simulates transmission noise using Gaussian distribution.

Applies demodulation and low-pass filtering for signal recovery.

Saves the reconstructed voice as a .wav file.

Generates detailed plots for each stage of processing.


**Requirements**
Python 3.x

Libraries:

numpy

scipy

matplotlib

Install dependencies using:

bash
Copy
Edit
pip install numpy scipy matplotlib

**How to Run**
Place your input audio file named voice_input.wav in the same directory.

Run the Python script:

bash
Copy
Edit
python main.py
Outputs generated:

reconstructed_voice.wav – The reconstructed voice signal.

delta_modulation_results.png – Plots of original, modulated, noisy, and reconstructed signals.

**Output Description**
Original Voice Signal – The normalized input signal.

Delta Modulated Signal – The quantized signal after delta modulation.

Noisy Delta Modulated Signal – Binary signal with simulated noise.

Reconstructed Signal – The demodulated and filtered output.

**Customization**
Step Size: Adjust step_size for quantization control.

Noise Level: Modify noise_level to simulate different transmission environments.

Filter Cutoff: Change cutoff_freq to tune the output smoothing.

**Project Applications**
Digital communication simulation

Voice signal processing experiments

Educational projects on modulation techniques

**License**
This project is for educational use. Feel free to modify and use it for learning purposes.
