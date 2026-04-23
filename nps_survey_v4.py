'''
nps_survey_v2.py
-----------------------------------------------------------------------------
North Polar Spur HI Survey - Leuschner 4.5-m dish
Lab 4, Project 11: Mapping the North Polar Spur Expanding HI Shell

Target region:  l = 210 deg to 20 deg (wrapping through 360 deg), b = 0 to 90 deg

Frequency switching:
  Each averaging window collects two separate sub-averaged spectra:
    FREQ_LO_A = CENTER_FREQ + FREQ_SWITCH_OFFSET  (e.g. +1 MHz, "freq A")
    FREQ_LO_B = CENTER_FREQ - FREQ_SWITCH_OFFSET  (e.g. -1 MHz, "freq B")
  The SDRs are retuned between sub-windows. Each output .npz contains
  both spectra (spec_A and spec_B) and their respective frequency axes
  so they are never mixed into a single averaged array.

Calibration:
  A single noise-diode calibration spectrum is taken at startup before
  the survey begins, while the dish is still in stow. It is saved as
  nps_cal_startup.npz and is not repeated during the survey.

Output:
  One .npz per averaging window per pointing, containing:
    spec_A_pol0, spec_A_pol1  - averaged power spectra at freq A
    spec_B_pol0, spec_B_pol1  - averaged power spectra at freq B
    freq_hz_A, freq_hz_B      - corresponding frequency axes
    + full pointing metadata

Hardware assumptions (Leuschner Pi):
  ugradio installed (ugradio.leusch, ugradio.sdr, ugradio.timing, ugradio.coord)
  Two RTL-SDR dongles plugged in (pol-0 -> device_index=0, pol-1 -> device_index=1)
  ugradio.leusch.LeuschNoise available for noise-diode calibration
-----------------------------------------------------------------------------
'''

import threading
import time
import os
import numpy as np
import pandas as pd

import ugradio
import ugradio.leusch
import ugradio.sdr
import ugradio.timing
import ugradio.coord

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import Galactic
import astropy.units as u
from astropy.time import Time

import glob

# -- Observatory constants -------------------------------------------------------
OBS_LAT = ugradio.leo.lat    # degrees N
OBS_LON = ugradio.leo.lon    # degrees E
OBS_ALT = ugradio.leo.alt    # metres

# -- Telescope pointing limits ---------------------------------------------------
ALT_MIN = ugradio.leusch.ALT_MIN
ALT_MAX = ugradio.leusch.ALT_MAX
AZ_MIN  = ugradio.leusch.AZ_MIN
AZ_MAX  = ugradio.leusch.AZ_MAX

# -- SDR configuration -----------------------------------------------------------
CENTER_FREQ       = 1420.405e6   # Hz - 21-cm HI rest frequency
FREQ_SWITCH_OFFSET = 1.0e6       # Hz - offset from center for each switched freq
FREQ_A            = CENTER_FREQ + FREQ_SWITCH_OFFSET   # upper sideband
FREQ_B            = CENTER_FREQ - FREQ_SWITCH_OFFSET   # lower sideband
SAMPLE_RATE       = 2.4e6        # Hz
N_FFT             = 4096         # freq resolution ~ 586 Hz ~ 0.12 km/s
NBLOCKS_ACC       = 50           # SDR blocks per sub-integration call (~85 ms each)
SUB_WINDOW        = 30.0         # seconds to accumulate at each switched frequency
                                 # total dwell per pointing = 2 * SUB_WINDOW
SDR_GAIN          = 40.0         # dB

# -- RFI / health monitor config -----------------------------------------------
MONITOR_INTERVAL  = 20 * 60        # seconds between checks
MONITOR_N_FILES   = 3              # how many recent files to check each time
RFI_SPIKE_THRESH  = 10.0           # flag if any channel is this many sigma above median
DEAD_BAND_THRESH  = 0.01           # flag if total power is this fraction of typical
FLATLINE_STD_THRESH = 1e-6         # flag if spectrum has essentially zero variance

# -- Survey pointing list --------------------------------------------------------


# -- Survey pointing list loaded from pre-scheduled CSVs -----------------------
# Order: Hi_Alt first (afternoon/evening targets), then Mid_Alt, then Low_Alt.
# Within each tier the CSV row order is preserved as your visibility-scheduled
# sequence. The code's galactic_to_altaz still computes live alt/az at runtime
# so out-of-bounds checks remain valid regardless of when you run.

def _load_pointings():
    hi  = pd.read_csv("Lab4_Hi_Alt.csv")
    mid = pd.read_csv("Lab4_Mid_Alt.csv")
    low = pd.read_csv("Lab4_Low_Alt.csv")
    combined = pd.concat([hi, mid, low], ignore_index=True)
    return list(zip(combined["Longitude (l)"], combined["Latitude (b)"]))

POINTINGS = _load_pointings()

# -- Output directory ------------------------------------------------------------
OUTPUT_DIR = "./nps_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Shared state ----------------------------------------------------------------
stop_event       = threading.Event()
pointing_lock    = threading.Lock()
current_pointing = {"l": None, "b": None, "alt": None, "az": None,
                    "ra": None, "dec": None, "status": "init", "jd": None}


# -- Coordinate helpers ----------------------------------------------------------

def galactic_to_altaz(l_deg, b_deg):
    '''Convert Galactic (l, b) to topocentric (alt, az) at current time.'''
    jd = ugradio.timing.julian_date()
    t  = Time(jd, format='jd', scale='utc')

    obs_location = EarthLocation(lat=OBS_LAT * u.deg,
                                 lon=OBS_LON * u.deg,
                                 height=OBS_ALT * u.m)

    gal  = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic)
    icrs = gal.icrs

    altaz_frame = AltAz(obstime=t, location=obs_location)
    altaz = icrs.transform_to(altaz_frame)

    alt = float(altaz.alt.deg)
    az  = float(altaz.az.deg)
    ra  = float(icrs.ra.deg)
    dec = float(icrs.dec.deg)

    return float(np.abs(alt)), az, ra, dec, jd


def in_bounds(alt, az):
    return (ALT_MIN < alt < ALT_MAX) and (AZ_MIN < az < AZ_MAX)


# -- SDR helpers -----------------------------------------------------------------

def make_sdrs(center_freq=CENTER_FREQ):
    '''Instantiate both SDR polarisations at the given center frequency.'''
    sdr0 = ugradio.sdr.SDR(device_index=0, direct=False,
                            center_freq=center_freq,
                            sample_rate=SAMPLE_RATE,
                            gain=SDR_GAIN)
    sdr1 = ugradio.sdr.SDR(device_index=1, direct=False,
                            center_freq=center_freq,
                            sample_rate=SAMPLE_RATE,
                            gain=SDR_GAIN)
    return sdr0, sdr1


def retune_sdrs(sdr0, sdr1, new_freq):
    '''Retune both SDRs to a new center frequency in place.'''
    sdr0.center_freq = new_freq
    sdr1.center_freq = new_freq
    time.sleep(0.3)   # allow PLL to settle


def freq_axis(center_freq, sample_rate=SAMPLE_RATE, n_fft=N_FFT):
    '''Return frequency axis in Hz centered on center_freq.'''
    return center_freq + np.fft.fftshift(
        np.fft.fftfreq(n_fft, d=1.0 / sample_rate))


def accumulate_spectra(sdr0, sdr1, duration_sec, nblocks=NBLOCKS_ACC, n_fft=N_FFT):
    '''
    Repeatedly capture power spectra for duration_sec seconds.
    Returns (list_of_spec0, list_of_spec1).
    Each entry is an averaged power spectrum array of length n_fft.
    '''
    buf0, buf1 = [], []
    t_end = time.time() + duration_sec
    while time.time() < t_end:
        if stop_event.is_set():
            break
        try:
            raw0 = sdr0.capture_data(n_fft, nblocks=nblocks)
            raw1 = sdr1.capture_data(n_fft, nblocks=nblocks)
            s0 = np.mean(np.abs(np.fft.fftshift(np.fft.fft(raw0, axis=-1))) ** 2, axis=0)
            s1 = np.mean(np.abs(np.fft.fftshift(np.fft.fft(raw1, axis=-1))) ** 2, axis=0)
            buf0.append(s0)
            buf1.append(s1)
        except Exception as exc:
            print("[SDR] capture error: " + str(exc) + " - skipping block")
            time.sleep(0.2)
    return buf0, buf1


# -- File writing ----------------------------------------------------------------

def save_window(spec_A_pol0, spec_A_pol1,
                spec_B_pol0, spec_B_pol1,
                meta, jd_start, jd_end,
                label="sci"):
    '''
    Save one frequency-switched window to a .npz file.
    spec_A_* and spec_B_* are already averaged arrays.
    meta must contain: l, b, alt, az, ra, dec, jd, status.
    '''
    jd_str = "{:.5f}".format(meta['jd']) if meta['jd'] else "unknown"
    l_str  = "{:.2f}".format(meta['l'])  if meta['l']  is not None else "XX"
    b_str  = "{:.2f}".format(meta['b'])  if meta['b']  is not None else "XX"
    fname  = "nps_{}_l{}_b{}_jd{}.npz".format(label, l_str, b_str, jd_str)
    fpath  = os.path.join(OUTPUT_DIR, fname)

    np.savez(fpath,
             spec_A_pol0    = spec_A_pol0,
             spec_A_pol1    = spec_A_pol1,
             spec_B_pol0    = spec_B_pol0,
             spec_B_pol1    = spec_B_pol1,
             freq_hz_A      = freq_axis(FREQ_A),
             freq_hz_B      = freq_axis(FREQ_B),
             freq_A_hz      = FREQ_A,
             freq_B_hz      = FREQ_B,
             freq_switch_offset_hz = FREQ_SWITCH_OFFSET,
             l_deg          = meta.get('l'),
             b_deg          = meta.get('b'),
             alt_deg        = meta.get('alt'),
             az_deg         = meta.get('az'),
             ra_deg         = meta.get('ra'),
             dec_deg        = meta.get('dec'),
             jd_pointing    = meta.get('jd'),
             jd_window_start = jd_start,
             jd_window_end   = jd_end,
             pointing_status = meta.get('status', 'unknown'),
             center_freq_hz  = CENTER_FREQ,
             sample_rate_hz  = SAMPLE_RATE,
             n_fft           = N_FFT,
             nblocks_acc     = NBLOCKS_ACC,
             sub_window_sec  = SUB_WINDOW)
    print("[SAVE] " + fname + "  status=" + str(meta.get('status')))


# -- Startup calibration ---------------------------------------------------------

def take_startup_cal(sdr0, sdr1, noise):
    '''
    Take a single noise-diode calibration at both switched frequencies
    before the survey begins. Saved as nps_cal_startup_freqA.npz and
    nps_cal_startup_freqB.npz. The dish should still be in stow position.
    '''
    print("[CAL] Starting noise-diode calibration at startup ...")

    jd_cal = ugradio.timing.julian_date()
    meta_cal = {"l": None, "b": None, "alt": None, "az": None,
                "ra": None, "dec": None, "jd": jd_cal, "status": "cal_startup"}

    noise.on()
    time.sleep(0.5)

    # Cal at freq A
    print("[CAL] Calibrating at freq A ({:.3f} MHz) ...".format(FREQ_A / 1e6))
    retune_sdrs(sdr0, sdr1, FREQ_A)
    buf0_A, buf1_A = accumulate_spectra(sdr0, sdr1, duration_sec=SUB_WINDOW)

    # Cal at freq B
    print("[CAL] Calibrating at freq B ({:.3f} MHz) ...".format(FREQ_B / 1e6))
    retune_sdrs(sdr0, sdr1, FREQ_B)
    buf0_B, buf1_B = accumulate_spectra(sdr0, sdr1, duration_sec=SUB_WINDOW)

    noise.off()
    time.sleep(0.3)

    if len(buf0_A) == 0 or len(buf0_B) == 0:
        print("[CAL] WARNING: calibration buffers empty - cal file not written")
        return

    cal_A0 = np.mean(np.array(buf0_A), axis=0)
    cal_A1 = np.mean(np.array(buf1_A), axis=0)
    cal_B0 = np.mean(np.array(buf0_B), axis=0)
    cal_B1 = np.mean(np.array(buf1_B), axis=0)

    jd_end = ugradio.timing.julian_date()
    save_window(cal_A0, cal_A1, cal_B0, cal_B1,
                meta_cal, jd_cal, jd_end, label="cal_startup")
    print("[CAL] Startup calibration complete and saved.")


# -- Per-pointing frequency-switched data collection -----------------------------

def collect_switched_window(sdr0, sdr1):
    '''
    Collect one frequency-switched window at the current pointing.
    Tunes to FREQ_A for SUB_WINDOW seconds, then FREQ_B for SUB_WINDOW seconds.
    Averages each sub-window separately and saves a single .npz containing both.
    Returns True on success, False if buffers were empty.
    '''
    jd_start = ugradio.timing.julian_date()

    # --- Sub-window A ---
    print("[SDR] Tuning to freq A ({:.3f} MHz) ...".format(FREQ_A / 1e6))
    retune_sdrs(sdr0, sdr1, FREQ_A)
    buf0_A, buf1_A = accumulate_spectra(sdr0, sdr1, duration_sec=SUB_WINDOW)
    print("[SDR] Freq A done: " + str(len(buf0_A)) + " sub-integrations")

    if stop_event.is_set():
        return False

    # --- Sub-window B ---
    print("[SDR] Tuning to freq B ({:.3f} MHz) ...".format(FREQ_B / 1e6))
    retune_sdrs(sdr0, sdr1, FREQ_B)
    buf0_B, buf1_B = accumulate_spectra(sdr0, sdr1, duration_sec=SUB_WINDOW)
    print("[SDR] Freq B done: " + str(len(buf0_B)) + " sub-integrations")

    if len(buf0_A) == 0 or len(buf0_B) == 0:
        print("[SDR] WARNING: one or both sub-window buffers empty - skipping save")
        return False

    avg_A0 = np.mean(np.array(buf0_A), axis=0)
    avg_A1 = np.mean(np.array(buf1_A), axis=0)
    avg_B0 = np.mean(np.array(buf0_B), axis=0)
    avg_B1 = np.mean(np.array(buf1_B), axis=0)

    jd_end = ugradio.timing.julian_date()

    with pointing_lock:
        meta = dict(current_pointing)

    save_window(avg_A0, avg_A1, avg_B0, avg_B1, meta, jd_start, jd_end, label="sci")
    print("[SDR] Window saved for l=" + str(meta['l']) + " b=" + str(meta['b']))
    return True

# -- File Integrity Checks -------------------------------------------------------





def check_spectrum_health(spec, label=""):
    """
    Run sanity checks on a single averaged power spectrum array.
    Returns a list of warning strings, empty if all clear.
    """
    warnings = []

    if np.all(spec == 0):
        warnings.append(label + ": spectrum is all zeros - possible SDR read failure")
        return warnings

    if np.any(~np.isfinite(spec)):
        warnings.append(label + ": contains NaN or Inf - possible FFT overflow")
        return warnings

    std = np.std(spec)
    if std < FLATLINE_STD_THRESH:
        warnings.append(label + ": spectrum is flatlined (std={:.2e}) - SDR may be locked up".format(std))

    median = np.median(spec)
    if median <= 0:
        warnings.append(label + ": median power is zero or negative")
    else:
        peak_sigma = (np.max(spec) - median) / (std if std > 0 else 1.0)
        if peak_sigma > RFI_SPIKE_THRESH:
            warnings.append(label + ": strong RFI spike detected ({:.1f} sigma above median)".format(peak_sigma))

        # Check for narrowband RFI - any single channel more than thresh * median
        n_rfi_channels = np.sum(spec > RFI_SPIKE_THRESH * median)
        if n_rfi_channels > 0:
            warnings.append(label + ": " + str(n_rfi_channels) + " channels flagged as RFI (>" + str(RFI_SPIKE_THRESH) + "x median)")

    return warnings


def monitor_data(log_path="nps_monitor_log.txt"):
    """
    Background thread: every MONITOR_INTERVAL seconds, loads the most recent
    .npz files and checks each spectrum for RFI, dropouts, and hardware issues.
    Prints warnings to stdout and appends them to a plain text log file.
    """
    print("[MONITOR] Health monitor thread started. Checking every " + str(MONITOR_INTERVAL // 60) + " min.")

    def write_log(msg):
        timestamp = "[" + str(ugradio.timing.utc()) + "] "
        full_msg = timestamp + msg
        print(full_msg)
        try:
            with open(log_path, "a") as f:
                f.write(full_msg + "\n")
        except Exception as e:
            print("[MONITOR] Could not write log: " + str(e))

    seen_files = set()

    while not stop_event.is_set():
        # Sleep in small increments so stop_event is caught quickly
        for _ in range(MONITOR_INTERVAL):
            if stop_event.is_set():
                break
            time.sleep(1.0)

        if stop_event.is_set():
            break

        # Find the most recent sci files not yet checked
        all_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "nps_sci_*.npz")))
        new_files  = [f for f in all_files if f not in seen_files]
        to_check   = new_files[-MONITOR_N_FILES:] if len(new_files) >= MONITOR_N_FILES else new_files

        if not to_check:
            write_log("[MONITOR] No new files to check yet.")
            continue

        write_log("[MONITOR] Checking " + str(len(to_check)) + " recent file(s) ...")
        any_issues = False

        for fpath in to_check:
            fname = os.path.basename(fpath)
            try:
                d = np.load(fpath)
            except Exception as e:
                write_log("[MONITOR] WARNING: could not load " + fname + ": " + str(e))
                any_issues = True
                seen_files.add(fpath)
                continue

            seen_files.add(fpath)

            # Check all four spectra in the file
            checks = [
                ("spec_A_pol0", d["spec_A_pol0"]),
                ("spec_A_pol1", d["spec_A_pol1"]),
                ("spec_B_pol0", d["spec_B_pol0"]),
                ("spec_B_pol1", d["spec_B_pol1"]),
            ]
            file_warnings = []
            for key, spec in checks:
                file_warnings += check_spectrum_health(spec, label=fname + "/" + key)

            # Cross-check: freq A and B should have similar total power
            try:
                power_A = float(np.mean(d["spec_A_pol0"]) + np.mean(d["spec_A_pol1"]))
                power_B = float(np.mean(d["spec_B_pol0"]) + np.mean(d["spec_B_pol1"]))
                if power_A > 0 and power_B > 0:
                    ratio = max(power_A, power_B) / min(power_A, power_B)
                    if ratio > 5.0:
                        file_warnings.append(
                            fname + ": large power imbalance between freq A and B "
                            "(ratio={:.1f}) - possible retune failure".format(ratio))
            except Exception:
                pass

            if file_warnings:
                any_issues = True
                for w in file_warnings:
                    write_log("[MONITOR] WARNING: " + w)
            else:
                write_log("[MONITOR] OK: " + fname)

        if not any_issues:
            write_log("[MONITOR] All checked files passed.")

    print("[MONITOR] Health monitor thread exiting.")


# -- Main survey loop ------------------------------------------------------------

def run_survey(pointings=POINTINGS):
    print("=" * 60)
    print("North Polar Spur HI Survey - Leuschner 4.5-m")
    print("Total pointings planned: " + str(len(pointings)))
    print("Output directory: " + os.path.abspath(OUTPUT_DIR))
    print("Freq A: {:.3f} MHz  Freq B: {:.3f} MHz".format(FREQ_A / 1e6, FREQ_B / 1e6))
    print("Sub-window duration: " + str(SUB_WINDOW) + "s each, total dwell ~" + str(int(2 * SUB_WINDOW)) + "s per pointing")
    print("=" * 60)

    # -- Hardware init -------------------------------------------------------------
    telescope = ugradio.leusch.LeuschTelescope()
    noise     = ugradio.leusch.LeuschNoise()
    sdr0, sdr1 = make_sdrs(center_freq=CENTER_FREQ)

    # -- Single startup calibration ------------------------------------------------
    take_startup_cal(sdr0, sdr1, noise)

    # Retune back to center before survey starts
    retune_sdrs(sdr0, sdr1, CENTER_FREQ)

    # -- Start health monitor thread -----------------------------------------------  # <-- ADD HERE
    monitor_thread = threading.Thread(                                                  # <-- ADD HERE
        target=monitor_data,                                                            # <-- ADD HERE
        args=("nps_monitor_log.txt",),                                                 # <-- ADD HERE
        name="HealthMonitor",                                                           # <-- ADD HERE
        daemon=True                                                                     # <-- ADD HERE
    )                                                                                   # <-- ADD HERE
    monitor_thread.start()                                                              # <-- ADD HERE

    skipped   = []
    completed = []

    try:
        for idx, (l_deg, b_deg) in enumerate(pointings):
            if stop_event.is_set():
                break

            print("\n[SURVEY] Pointing " + str(idx + 1) + "/" + str(len(pointings)) +
                  " - l=" + "{:.2f}".format(l_deg) + " b=" + "{:.2f}".format(b_deg))

            try:
                alt, az, ra, dec, jd = galactic_to_altaz(l_deg, b_deg)
            except Exception as exc:
                print("[SURVEY] Coord conversion failed: " + str(exc) + " - skipping.")
                skipped.append((l_deg, b_deg, "coord_error"))
                continue

            if not in_bounds(alt, az):
                print("[SURVEY] alt=" + "{:.1f}".format(alt) +
                      " az=" + "{:.1f}".format(az) + " - out of bounds, skipping.")
                skipped.append((l_deg, b_deg, "out_of_bounds"))
                continue

            with pointing_lock:
                current_pointing.update({"l": l_deg, "b": b_deg,
                                         "alt": alt, "az": az,
                                         "ra": ra, "dec": dec,
                                         "jd": jd, "status": "slewing"})

            print("[SURVEY] Slewing to alt=" + "{:.2f}".format(alt) +
                  " az=" + "{:.2f}".format(az) + " ...")
            try:
                telescope.point(alt, az)
            except Exception as exc:
                print("[SURVEY] Slew failed: " + str(exc) + " - skipping pointing.")
                skipped.append((l_deg, b_deg, "slew_error"))
                continue

            alt, az, ra, dec, jd = galactic_to_altaz(l_deg, b_deg)

            with pointing_lock:
                current_pointing.update({"alt": alt, "az": az,
                                         "ra": ra, "dec": dec,
                                         "jd": jd, "status": "on_target"})

            print("[SURVEY] On target. alt=" + "{:.2f}".format(alt) +
                  " az=" + "{:.2f}".format(az))

            success = collect_switched_window(sdr0, sdr1)
            if success:
                completed.append((l_deg, b_deg))
            else:
                print("[SURVEY] Window collection failed for this pointing.")
                skipped.append((l_deg, b_deg, "collection_failed"))

    except KeyboardInterrupt:
        print("\n[SURVEY] Interrupted by user.")

    finally:
        print("\n[SURVEY] Finishing up ...")
        stop_event.set()
        monitor_thread.join(timeout=10)                                                 # <-- ADD HERE

        with pointing_lock:
            current_pointing["status"] = "stowing"

        print("[SURVEY] Stowing telescope ...")
        try:
            telescope.stow()
        except Exception as exc:
            print("[SURVEY] Stow error: " + str(exc))

        print("\n" + "=" * 60)
        print("Survey complete.")
        print("  Completed pointings : " + str(len(completed)))
        print("  Skipped pointings   : " + str(len(skipped)))
        if skipped:
            print("  Skipped list:")
            for s in skipped:
                print("    l=" + "{:.2f}".format(s[0]) +
                      "  b=" + "{:.2f}".format(s[1]) +
                      "  reason=" + s[2])
        print("  Data saved to       : " + os.path.abspath(OUTPUT_DIR))
        print("=" * 60)

    return completed, skipped


# -- Entry point -----------------------------------------------------------------

if __name__ == "__main__":
    run_survey()
