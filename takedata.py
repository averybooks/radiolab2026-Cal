import ugradio
from ugradio.sdr import SDR
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np

labelname = "bandpassdata_1.npz"
                    sdr = ugradio.sdr.SDR(sample_rate=1e6)
data = sdr.capture_data(2048,nblocks=10)
print(sdr)
print(data)
np.savez(labelname, data)
print("saved ", labelname)

#np.load(labelname.txt)
["arr_0"][0]
