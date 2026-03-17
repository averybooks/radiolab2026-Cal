"""
sun_track.py
Controls both interferometer antennas to track the sun for a fixed duration.
Stows antennas and exits if the sun moves out of pointing bounds.
"""

import ugradio
import ugradio.coord
import ugradio.timing
import ugradio.interf
import numpy as np
import time
from shared_state import tracking_event, stop_event , results, results_lock, duration, interval

# ── Configuration ──────────────────────────────────────────────────────────────
TRACK_DURATION  = duration  # seconds to track (default 10 min, change as needed)
REPOINT_INTERVAL = interval   # seconds between re-points to follow the sun

# UC Berkeley Campbell Hall coordinates
OBS_LAT  =  37.8732   # degrees N
OBS_LON  = -122.2573  # degrees E
OBS_ALT  =  30.0      # meters

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_sun_altaz():
    """Return the sun's current (alt, az) in degrees at the observatory."""
    jd = ugradio.timing.julian_date()
    ra, dec = ugradio.coord.sunpos(jd)          # sun RA/Dec at current time
    alt, az = ugradio.coord.get_altaz(         # convert to local alt/az
        ra, dec,
        jd=jd
    )
    alt = np.abs(alt)
    return alt, az

def in_bounds(alt, az):
    """Check if the position is within the antenna's hard pointing limits."""
    return (ugradio.interf.ALT_MIN < alt < ugradio.interf.ALT_MAX and
            ugradio.interf.AZ_MIN  < az  < ugradio.interf.AZ_MAX)

# ── Main tracking loop ─────────────────────────────────────────────────────────
def track_sun(duration=TRACK_DURATION, interval=REPOINT_INTERVAL):
    intf = ugradio.interf.Interferometer()

    print(f"Starting sun tracking for {duration}s, re-pointing every {interval}s.")
    print("Press Ctrl+C to stop early.\n")

    t_start = time.time()
    t_end   = t_start + duration

    try:
        while time.time() < t_end:
            alt, az = get_sun_altaz()
            print(f"[{ugradio.timing.utc()}]  Sun -> alt={alt:.2f}°  az={az:.2f}°")

            # Safety check before every slew
            if not in_bounds(alt, az):
                print("WARNING: Sun is outside pointing bounds. Stowing antennas.")
                tracking_event.clear()   # stop SNAP collection before stowing
                intf.stow()
                return

            tracking_event.clear()               # ← pause SNAP: about to slew
            print("[TRACK] Slewing...")
            intf.point(alt, az, wait=True, verbose=False)
            print("[TRACK] Slew complete, setting tracking event")
            tracking_event.set()                 # ← antennas settled: SNAP can collect
            print(f"  Antennas pointed. Next re-point in {interval}s.")

            # Sleep in small increments so Ctrl+C is responsive
            t_wake = time.time() + interval
            while time.time() < t_wake and time.time() < t_end:
                time.sleep(1)
                # tracking_event.clear()
                    
    except KeyboardInterrupt:
        print("\nTracking interrupted by user.")

    finally:
        print("Stowing antennas...")
        intf.stow(wait=True, verbose=True)
        print("Done.")

if __name__ == "__main__":
    track_sun()
