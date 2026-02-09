"""
Analysis utilities for Radio Lab 1: digital sampling, FFT, and data loading.
"""

import numpy as np


def sample_sine(f_signal, f_sample, duration=0.05):
    """Generate a sampled sine wave (for simulation/aliasing demos)."""
    t = np.arange(0, duration, 1 / f_sample)
    x = np.sin(2 * np.pi * f_signal * t)
    return t, x


def power_spectrum_simple(x, f_sample):
    """Power spectrum (shifted) from time series; no window or demean."""
    fft_vals = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), 1 / f_sample)
    return np.fft.fftshift(freqs), np.fft.fftshift(np.abs(fft_vals) ** 2)


def power_spectrum(x, fs, window=True):
    """Power spectrum with optional Hanning window and DC removal."""
    x = x - np.mean(x)

    if window:
        w = np.hanning(len(x))
        x = x * w

    fft_vals = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1 / fs)
    power = np.abs(fft_vals) ** 2

    freqs = np.fft.fftshift(freqs)
    power = np.fft.fftshift(power)
    return freqs, power


def load_run(npz_path, run_index=0):
    """Load one run from an .npz file (e.g. SDR capture)."""
    data = np.load(npz_path)
    x = data["arr_0"][run_index].astype(float)
    return x
