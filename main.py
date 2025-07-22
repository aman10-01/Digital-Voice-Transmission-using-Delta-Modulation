import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Function to design a low-pass Butterworth filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Function to apply the low-pass filter
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Read the input WAV file
fs, voice_signal = wavfile.read('voice_input.wav')

# Convert to mono if stereo
if len(voice_signal.shape) > 1:  # Stereo check
    voice_signal = np.mean(voice_signal, axis=1)  # Average channels
voice_signal = voice_signal / np.max(np.abs(voice_signal))  # Normalize to [-1, 1]

# Delta modulation parameters
step_size = 0.1  # Step size for quantization
n_samples = len(voice_signal)

# Delta modulation
dm_signal = np.zeros(n_samples)
quantized_signal = np.zeros(n_samples)
predictor = 0  # Initial predictor value

for i in range(n_samples):
    error = voice_signal[i] - predictor
    dm_bit = 1 if error >= 0 else 0
    dm_signal[i] = dm_bit
    predictor += step_size if dm_bit == 1 else -step_size
    quantized_signal[i] = predictor

# Add Gaussian noise to the delta modulated signal
noise_level = 0.1  # Noise standard deviation
noise = np.random.normal(0, noise_level, n_samples)
noisy_dm_signal = dm_signal + noise
noisy_dm_signal = np.clip(noisy_dm_signal, 0, 1)  # Clip to binary-like values
noisy_dm_signal = (noisy_dm_signal > 0.5).astype(int)  # Threshold to binary

# Demodulation of noisy signal
reconstructed_signal = np.zeros(n_samples)
predictor = 0  # Reset predictor

for i in range(n_samples):
    predictor += step_size if noisy_dm_signal[i] == 1 else -step_size
    reconstructed_signal[i] = predictor

# Apply low-pass filter to smooth the reconstructed signal
cutoff_freq = 4000  # Cutoff frequency in Hz (adjust based on voice signal)
filtered_signal = lowpass_filter(reconstructed_signal, cutoff_freq, fs)

# Normalize the filtered signal to match WAV format
filtered_signal = filtered_signal / np.max(np.abs(filtered_signal)) * 32767
filtered_signal = filtered_signal.astype(np.int16)

# Save the reconstructed signal to a WAV file
wavfile.write('reconstructed_voice.wav', fs, filtered_signal)

# Plotting the results
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(voice_signal, label='Original Signal')
plt.title('Original Voice Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(quantized_signal, label='Delta Modulated (Quantized)')
plt.title('Delta Modulated Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(noisy_dm_signal, label='Noisy Delta Modulated')
plt.title('Noisy Delta Modulated Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(filtered_signal / 32767, label='Reconstructed Signal')
plt.title('Reconstructed Voice Signal (After Filtering)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('delta_modulation_results.png')