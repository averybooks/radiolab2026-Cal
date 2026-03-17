import ugradio
import ugradio.coord
import ugradio.timing
import ugradio.interf
import numpy as np
import time
from shared_state import tracking_event, stop_event , results, results_lock, duration, interval, data_array

#set a while loop for duration of event - ignore
    #writes data_array into npz -> save it
    
        
def ballin(data_array = data_array, filename = "/home/radiopi/RTFM_Lab_3/data/test.npz"):
    np.savez(filename, data_array)
    
    
