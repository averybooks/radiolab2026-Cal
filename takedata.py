import ugradio
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np

labelname = "noisedata1.npz"
sdr = ugradio.sdr.SDR(sample_rate=3e6)
data = sdr.capture_data(2048, nblocks=10)
print(data)
np.savez(labelname, data)
print("saved ", labelname)

### np.load(filename.txt)
#["arr_0"][0]
