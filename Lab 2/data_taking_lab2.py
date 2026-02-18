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

local_now = ugradio.timing.local_time() # current local time as a string
ut_now = ugradio.timing.utc() # current UTC as a string
julian_now = ugradio.timing.julian_date() # current julian day (which contains the current time, too--it’s not just an integer/number.)
lst_now = ugradio.timing.lst() # current LST at NCH

time_array = np.array([[[local_now, ut_now, julian_now, lst_now]]])

data = np.concatenate((time_array,data))

print(sdr)
print(data)
np.savez(labelname, data)
print("saved ", labelname)

np.load(labelname)
["arr_0"][0]