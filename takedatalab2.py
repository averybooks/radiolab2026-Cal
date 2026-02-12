import ugradio
from ugradio.sdr import SDR
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np

labelname = "bighorntest3.npz"
sdr = ugradio.sdr.SDR(direct=False, center_freq=1419e6, sample_rate=2e6)
data = sdr.capture_data(2048,nblocks=5)

julian_now = ugradio.timing.julian_date()
local_now = ugradio.timing.local_time()
ut_now = ugradio.timing.utc()
# ut_now_unix = ugradio.timing.unix_time()

time_array = np.array([[[julian_now, local_now, ut_now]]])

data = np.concatenate((time_array,data))

print(sdr)
print(data)
np.savez(labelname, data)
print("saved ", labelname)

np.load(labelname)
["arr_0"][0]
