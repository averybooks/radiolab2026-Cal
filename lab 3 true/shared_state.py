#This is the shared state file that houses the state for tracking_event, stop_event and any other shared events for the threading processes
import time
import threading
tracking_event  = threading.Event()   # set when antennas are on-target
stop_event      = threading.Event()   # set to shut down both threads
results         = []                  # accumulated output spectra
results_lock    = threading.Lock()    # protect results list
duration = 600 #time in seconds of the entire tracking and data process
interval = 90 #interval[s] of before each slew
data_array = []
total_acc = []
LO = 10 #GHz - this is a hardware setup, so this param is exclusively for our metadata
SR = 0.5 #GHz