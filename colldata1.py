"""
tracksun.py
Controls both interferometer antennas to track the sun for a fixed duration.
Stows antennas and exits if the sun moves out of pointing bounds.
"""

"""
collect_data.py
Threaded SNAP data collection running parallel to sun tracking.
Reads cross-correlation data, computes quadrupled + averaged power spectra
over fixed time windows, and stores results in a list for later saving.
"""

import numpy as np
import time
import threading
from snap_spec import snap 
from shared_state import tracking_event, stop_event , results, results_lock, duration, data_array, interval
from writefile import ballin
import ugradio
import ugradio.coord
import ugradio.timing
import ugradio.interf  

# Config
AVG_WINDOW      = interval  # seconds per ONE averaged output spectrum
DELAY_TIME      = 0.1    # seconds between acc_cnt polls

# Shared state between threads
#tracking_event  = threading.Event()   # set when antennas are on-target
#stop_event      = threading.Event()   # set to shut down both threads
#results         = []                  # accumulated output spectra
#results_lock    = threading.Lock()    # protect results list

# Processing
def quad_avg_power(spectra_buffer):
    """
    Given a list of complex corr01 arrays collected over one time window,
    compute the quadrupled averaged power spectrum.

    spectra_buffer : list of np.ndarray (complex)
        Raw corr01 arrays from SNAP

    Return:
    np.ndarray (real)
        Averaged quadrupled power spectrum.
    """
    stacked = np.array(spectra_buffer)          # shape: (N_acc, N_chan)
    power   = (np.abs(stacked) ** 2) ** 2       # |corr01|^4 per accumulation
    return power.mean(axis=0)                   # average over accumulations


# Data collection thread
def collect_data(snap, avg_window=AVG_WINDOW):
    """
    Continuously reads SNAP cross-correlation data while tracking_event is set.
    Accumulates spectra into fixed-duration windows, then computes and stores
    the quadrupled averaged power spectrum for each window.

    Params
    snap       : UGRadioSnap instance, already initialized in corr mode
    avg_window : float, seconds per averaging window
    """
    print("[SNAP] Data collection thread started.")

    while not stop_event.is_set():

        print(f"SNAP DEBUG: stop_event={stop_event.is_set()}, tracking_event={tracking_event.is_set()}")
        # Wait until antennas are on-target before collecting
        #if not tracking_event.is_set():
        print("[SNAP] Waiting for antennas to be on-target...")
        tracking_event.wait(timeout=5.0)
        print(f"SNAP DEBUG: wait finished, tracking_event={tracking_event.is_set()}")
        
        if stop_event.is_set():
            print("SNAP DEBUG stop_event set, breaking")
            break

        # One averaging window 
        spectra_buffer  = []
        window_start    = time.time()
        prev_cnt        = None

        print(f"[SNAP] Starting {avg_window}s averaging window at t={window_start:.1f}")

        while (time.time() - window_start) < avg_window:

            # Stop immediately if signalled
            if stop_event.is_set():
                break

            # Also stop window early if antennas leave target (slewing)
            if not tracking_event.is_set():
                print("[SNAP] Tracking lost mid-window — discarding partial buffer.")
                spectra_buffer = []
                break

            try:
                # Block until next SNAP accumulation is ready
                data = snap.read_data(prev_cnt=prev_cnt)
            except AssertionError as e:
                # Missed an integration — log and reset acc_cnt tracking
                print(f"[SNAP] Missed integration: {e}. Resetting acc_cnt tracking.")
                prev_cnt = None
                continue

            prev_cnt = data['acc_cnt']
            spectra_buffer.append(data['corr01'])   # complex array, shape (N_chan,)

        # stores
        if len(spectra_buffer) > 0:
            result = {
                'timestamp'  : window_start,
                'n_acc'      : len(spectra_buffer),
                'power_spec' : quad_avg_power(spectra_buffer),   # shape (N_chan,)
            }
            with results_lock:
                results.append(result)
            print(f"[SNAP] Window done: {len(spectra_buffer)} accumulations averaged.")
            data_array = result
            ballin()
            print("[SNAP_DEBUG] Writing file...")

    print("[SNAP] Data collection thread exiting.")


# Int with suntrack
def run(snap, track_fn, track_kwargs=None):
    """
    Launch tracking and data collection in parallel threads.

    Params
    snap         : UGRadioSnap instance, already initialized in corr mode
    track_fn     : the track_sun function from sun_track.py, 
                   modified to set/clear tracking_event and stop_event (see note)
    track_kwargs : dict of kwargs to pass to track_fn
    """
    if track_kwargs is None:
        track_kwargs = {}

    data_thread = threading.Thread(
        target=collect_data,
        args=(snap,),
        name='SNAPCollector',
        daemon=True
    )
    data_thread.start()

    try:
        # Tracking runs on the main thread
        track_fn(**track_kwargs)
    finally:
        stop_event.set()
        data_thread.join(timeout=10)
        print(f"[MAIN] Collection complete. {len(results)} spectra windows saved.")

    return results


# Entry
if __name__ == '__main__':
    from tracksun1 import track_sun   # your tracking script

    snap = snap.UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    snap.initialize(mode='corr', sample_rate=500)

    all_spectra = run(
        snap       = snap,
        track_fn   = track_sun,
        track_kwargs = {'duration': 600, 'interval': 30}
    )

    # Handy sanity check
    print(f"\nCollected {len(all_spectra)} averaged windows.")
    for i, s in enumerate(all_spectra):
        print(f"  Window {i:02d} | t={s['timestamp']:.1f} | "
              f"n_acc={s['n_acc']} | "
              f"peak_power={s['power_spec'].max():.4e}")
    all_spectra
    
              
              
    #writes all spectra into one avg spectra array from which a file should be written from
    #I think that an another script that utilizes shared variables of -> 'sample_rate', 'duration',
    #and 'interval', and 'all spectra' that will write the spectra into an npz file on a repeated
    #loop, we could even try to write it as a 'while' loop of 'duration' length that has a 'interval' variable length
    #hot loop inside of it that writes the data such that it writes it exactly 1 second after the measurements
    #complete but shouldn't take all 29 seconds to write -> if it does then we can adjust the 'interval' param.

