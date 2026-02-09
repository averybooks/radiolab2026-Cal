"""
Script used to capture SDR data for Radio Lab 1.
Run from repo root; requires ugradio, rtlsdr, numpy.
"""
import numpy as np

try:
    import ugradio
except ImportError:
    raise ImportError("Install ugradio to run this script (e.g. for SDR capture).")

# Example: capture noise data
LABELNAME = "noisedata1.npz"
sdr = ugradio.sdr.SDR(sample_rate=3e6)
data = sdr.capture_data(2048, nblocks=10)
print(data)
np.savez(LABELNAME, data)
print("saved ", LABELNAME)

# Load with: np.load(filename)["arr_0"][0]
