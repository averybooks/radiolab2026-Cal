import ugradio
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np

sdr = ugradio.sdr.SDR(sample_rate=2.2e6, fir_coeffs=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2047]))
data = sdr.capture_data(2048, nblocks=10)
print(sdr)
print(data)
np.savez("fir_coeff_test.npz", data)
