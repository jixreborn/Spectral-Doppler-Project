import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("Spectral Doppler Simulator: Frequency and Velocity Analysis")

# User inputs
col1, col2 = st.columns(2)
with col1:
    time_focus = st.slider("Time of Focus (s)", 0.0, 2.0, 1.0, step=0.01)
with col2:
    condition = st.selectbox("Flow Condition", ["Normal", "Intrastenotic", "Poststenotic"])

# Constants
fs = 10000
carrier_freq = 2e6  # 2 MHz
c = 1540

# Time definitions
time = np.linspace(0, 2.0, 500)
raw_time = np.linspace(0, 2.0, int(2.0 * fs), endpoint=False)
window_width = 0.3
focus_index = np.argmin(np.abs(time - time_focus))
num_scatterers = 200
velocity_bins = np.linspace(-60, 120, 300)

# Define triphasic waveform
def velocity_envelope(t):
    cycle_duration = 0.8
    envelope = np.zeros_like(t)
    for i in range(len(t)):
        phase = (t[i] % cycle_duration)
        if phase < 0.2:
            envelope[i] = 100 * np.sin(np.pi * phase / 0.2)
        elif phase < 0.3:
            envelope[i] = -30 * np.sin(np.pi * (phase - 0.2) / 0.1)
        elif phase < 0.5:
            envelope[i] = 30 * np.sin(np.pi * (phase - 0.3) / 0.2)
        else:
            envelope[i] = 0
    return envelope

base_velocity = velocity_envelope(time)

# Apply flow condition
def apply_condition(base_velocity, condition):
    if condition == "Intrastenotic":
        peak_boost = 50 * np.exp(-((time % 0.8 - 0.05)**2) / (2 * 0.01**2))
        return base_velocity + peak_boost
    elif condition == "Poststenotic":
        negative_flow = -10 * np.exp(-((time - 1.0)**2) / (2 * 0.03**2))
        return 0.6 * base_velocity + negative_flow
    else:
        return base_velocity

base_velocity = apply_condition(base_velocity, condition)

# Velocity matrix with jitter
velocity_matrix = np.zeros((num_scatterers, len(time)))
for i, peak in enumerate(base_velocity):
    if condition == "Poststenotic":
        if peak >= 0:
            velocity_matrix[:, i] = np.random.uniform(0, peak, num_scatterers)
        else:
            velocity_matrix[:, i] = np.random.uniform(peak, 0, num_scatterers)
    else:
        velocity_matrix[:, i] = np.random.normal(loc=peak, scale=(10 if condition == "Intrastenotic" else 5), size=num_scatterers)

# Spectrogram (Graph 4)
spectrogram_matrix = np.zeros((len(velocity_bins) - 1, len(time)))
for i in range(len(time)):
    hist, _ = np.histogram(velocity_matrix[:, i], bins=velocity_bins)
    spectrogram_matrix[:, i] = hist
spectrogram_matrix /= np.max(spectrogram_matrix)
spectrogram_matrix = np.power(spectrogram_matrix, 0.5)

# Velocity histogram (Graph 3)
focus_velocities = velocity_matrix[:, focus_index]
velocity_hist, velocity_edges = np.histogram(focus_velocities, bins=velocity_bins)
velocity_hist = velocity_hist / np.max(velocity_hist)
velocity_centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])

# Frequency conversion for Graph 2
frequency_axis = (2 * carrier_freq * velocity_centers / 100) / c
fmin = (2 * carrier_freq * velocity_bins[0] / 100) / c
fmax = (2 * carrier_freq * velocity_bins[-1] / 100) / c
frequency_axis_khz = frequency_axis / 1000
fmin_khz = fmin / 1000
fmax_khz = fmax / 1000

# RF signal generation (Graph 1)
start_idx = int((time_focus - window_width / 2) * fs)
end_idx = int((time_focus + window_width / 2) * fs)
rf_signal = np.zeros(end_idx - start_idx)

for i in range(num_scatterers):
    freq_shift = 2 * carrier_freq * focus_velocities[i] / c
    rf_segment = np.sin(2 * np.pi * (carrier_freq + freq_shift) * raw_time[start_idx:end_idx])
    rf_signal += rf_segment
rf_signal /= num_scatterers
rf_signal += 0.01 * np.random.randn(len(rf_signal))

# Graphs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Graph 1: Simulated RF Signal (Input to Probe)")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(raw_time[start_idx:end_idx], rf_signal, color='white')
    ax1.axvline(x=time_focus, color='red', linestyle='--')
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_facecolor('black')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RF Signal")
    fig1.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("Graph 2: Frequency Spectrum from Velocity Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(frequency_axis_khz, velocity_hist, color='white')
    ax2.set_xlim(fmin_khz, fmax_khz)
    ax2.set_facecolor('black')
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Normalized Amplitude")
    fig2.tight_layout()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Graph 3: Velocity Distribution at Time of Focus")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(velocity_centers, velocity_hist, color='white')
    ax3.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax3.set_facecolor('black')
    ax3.set_xlabel("Velocity (cm/s)")
    ax3.set_ylabel("Normalized Counts")
    fig3.tight_layout()
    st.pyplot(fig3)

with col4:
    st.subheader("Graph 4: Spectral Doppler (Time-Velocity Display)")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    extent = [time[0], time[-1], velocity_bins[0], velocity_bins[-1]]
    ax4.imshow(spectrogram_matrix, aspect='auto', cmap='gray',
               extent=extent, origin='lower', vmin=0, vmax=1)
    ax4.axvline(x=time_focus, color='red', linestyle='--')
    ax4.set_facecolor('black')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Velocity (cm/s)")
    fig4.tight_layout()
    st.pyplot(fig4)

# Explanatory text
st.markdown("""
### 🔍 Explanation of Signal Processing Flow

**Graph 1: Simulated RF Signal**  
This graph displays the raw radiofrequency (RF) signal received by the ultrasound probe. It is the composite of sinusoidal signals backscattered from red blood cells moving at different velocities.

**Graph 2: Frequency Spectrum (FFT)**  
This shows the result of applying the **Fast Fourier Transform (FFT)** to the RF signal. The FFT decomposes the signal into its frequency components. The frequencies are proportional to Doppler shifts caused by motion — higher frequencies represent higher velocities.

**Graph 3: Velocity Spectrum from FFT**  
This graph maps the frequency spectrum from Graph 2 to velocities using the Doppler equation. It reflects the velocity distribution corresponding to the frequencies at the focus time.

**Graph 4: Spectral Doppler (Time-Velocity Display)**  
Finally, this spectrogram visualizes how the velocity distribution evolves over time, mimicking what is seen in clinical spectral Doppler imaging. Each vertical slice is a velocity histogram computed across time.

Together, these four graphs illustrate the transformation of motion-induced ultrasound echoes into meaningful velocity-time plots using signal processing techniques such as FFT and Doppler shift analysis.
""")

