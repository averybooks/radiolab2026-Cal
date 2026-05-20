"""
Build an interpolated NPS heatmap from many survey .npz files.

Compatible with files that contain keys like:
spec_A_pol0, spec_A_pol1, spec_B_pol0, spec_B_pol1, freq_hz_A, freq_hz_B,
l_deg, b_deg, sample_rate_hz, n_fft, ...
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

try:
    from scipy.interpolate import RBFInterpolator
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
except ImportError:
    RBFInterpolator = None
    griddata = None
    gaussian_filter = None
    cKDTree = None

try:
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
except ImportError:
    u = None
    EarthLocation = None
    SkyCoord = None
    Time = None


@dataclass
class SurveyPoint:
    l_deg: float
    b_deg: float
    bandpower: float
    source_file: str


@dataclass
class RfiConfig:
    sigma_clip: float = 6.0
    edge_trim: int = 8
    patch_masked: bool = True
    manual_ranges_hz: tuple[tuple[float, float], ...] = ()


def _spectrum_to_1d(arr: np.ndarray, n_freq: int | None = None) -> np.ndarray:
    """Collapse a stored spectrum array to shape (n_chan,).

    pol0 and pol1 are two polarizations of the same switch position and
    should be averaged together. The frequency-switch reduction
    (spec_A signal vs spec_B reference) is handled separately in
    _freq_switch_reduce, exactly as in make_sky_projections.py.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if n_freq is not None and n_freq in arr.shape:
            fax = list(arr.shape).index(n_freq)
            import numpy as _np
            return _np.nanmean(_np.moveaxis(arr, fax, 0).reshape(n_freq, -1), axis=1)
        if arr.shape[1] <= 4:
            return np.nanmean(arr, axis=1)
        if arr.shape[0] <= 4:
            return np.nanmean(arr, axis=0)
        return np.nanmean(arr, axis=0)
    raise ValueError(f"Unsupported spectrum shape {arr.shape}")


def _freq_switch_reduce(spec_A: np.ndarray, spec_B: np.ndarray) -> np.ndarray:
    """Frequency-switch reduction: (sA - sB) / sB.

    spec_A is the signal position (switched ON), spec_B is the reference
    (switched OFF). Dividing by sB normalises out the bandpass shape and
    yields a dimensionless fractional brightness excess — identical to
    make_sky_projections.py's  fs = (sA - sB) / sB.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        fs = (spec_A - spec_B) / spec_B
    fs[~np.isfinite(fs)] = np.nan
    return fs


def _select_band(freq_hz: np.ndarray, spec: np.ndarray, fmin_hz: float | None, fmax_hz: float | None) -> np.ndarray:
    if fmin_hz is None and fmax_hz is None:
        return spec
    mask = np.ones_like(freq_hz, dtype=bool)
    if fmin_hz is not None:
        mask &= freq_hz >= fmin_hz
    if fmax_hz is not None:
        mask &= freq_hz <= fmax_hz
    if not np.any(mask):
        raise ValueError("Requested frequency band has no channels in this file.")
    return spec[mask]


def _running_median(x: np.ndarray, width: int = 31) -> np.ndarray:
    """Fast running median using scipy.ndimage — replaces slow Python loop."""
    from scipy.ndimage import median_filter
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    return median_filter(x.astype(float), size=width, mode="nearest")


def _patch_masked_channels(freq_hz: np.ndarray, spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linearly fill masked channels using nearest valid neighbors."""
    out = spec.copy()
    good = (~mask) & np.isfinite(spec)
    if np.count_nonzero(good) < 2:
        return out
    out[mask] = np.interp(freq_hz[mask], freq_hz[good], spec[good])
    return out


def _rfi_clean_spectrum(freq_hz: np.ndarray, spec: np.ndarray, cfg: RfiConfig) -> np.ndarray:
    """
    Remove narrow RFI outliers and optionally patch removed channels.
    - Detects spike channels via robust MAD on residuals from running median.
    - Masks user-provided manual frequency ranges.
    - Trims edge channels where FFT leakage is often unstable.
    """
    s = np.asarray(spec, dtype=float).copy()
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    n = min(s.shape[0], f.shape[0])
    s, f = s[:n], f[:n]

    baseline = _running_median(s, width=31)
    resid = s - baseline
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    robust_sigma = 1.4826 * max(mad, 1e-20)
    auto_mask = np.abs(resid - med) > (cfg.sigma_clip * robust_sigma)

    manual_mask = np.zeros_like(auto_mask, dtype=bool)
    for lo_hz, hi_hz in cfg.manual_ranges_hz:
        lo, hi = (lo_hz, hi_hz) if lo_hz <= hi_hz else (hi_hz, lo_hz)
        manual_mask |= (f >= lo) & (f <= hi)

    edge_mask = np.zeros_like(auto_mask, dtype=bool)
    if cfg.edge_trim > 0 and 2 * cfg.edge_trim < n:
        edge_mask[: cfg.edge_trim] = True
        edge_mask[-cfg.edge_trim :] = True

    mask = auto_mask | manual_mask | edge_mask | (~np.isfinite(s))
    if not np.any(mask):
        return s
    if cfg.patch_masked:
        return _patch_masked_channels(f, s, mask)
    s[mask] = np.nan
    return s


def _bandpower_from_file(
    data: np.lib.npyio.NpzFile,
    fmin_hz: float | None,
    fmax_hz: float | None,
    spectrum_mode: str = "A+B",
    rfi_config: RfiConfig | None = None,
    metric: str = "mean_power",
    baseline_width: int = 31,
) -> float:
    """
    Return scalar power used as heatmap value.

    spectrum_mode:
      - "A+B": average all available A/B, pol0/pol1 spectra
      - "A": use only A-side spectra
      - "B": use only B-side spectra
    """
    modes = {"A+B", "A", "B"}
    if spectrum_mode not in modes:
        raise ValueError(f"spectrum_mode must be one of {modes}")

    # --- Determine frequency axis length for shape-aware averaging ---
    freq_A = np.asarray(data["freq_hz_A"], dtype=float).ravel() if "freq_hz_A" in data else (
             np.asarray(data["freq_A_hz"], dtype=float).ravel() if "freq_A_hz" in data else None)
    freq_B = np.asarray(data["freq_hz_B"], dtype=float).ravel() if "freq_hz_B" in data else (
             np.asarray(data["freq_B_hz"], dtype=float).ravel() if "freq_B_hz" in data else freq_A)
    if freq_A is None:
        raise ValueError("No frequency axis found in this file.")
    n_freq_A = freq_A.size
    n_freq_B = freq_B.size if freq_B is not None else n_freq_A

    def _get_spec(key, n_freq):
        if key not in data:
            return None
        return _spectrum_to_1d(np.asarray(data[key], dtype=float), n_freq=n_freq)

    # Average polarizations within each switch position (A=signal, B=reference)
    chunks_A, chunks_B = [], []
    if spectrum_mode in ("A+B", "A"):
        for k in ("spec_A_pol0", "spec_A_pol1"):
            s = _get_spec(k, n_freq_A)
            if s is not None:
                chunks_A.append(s[:n_freq_A])
    if spectrum_mode in ("A+B", "B"):
        for k in ("spec_B_pol0", "spec_B_pol1"):
            s = _get_spec(k, n_freq_B)
            if s is not None:
                chunks_B.append(s[:n_freq_B])

    if not chunks_A and not chunks_B:
        raise ValueError("No valid spectra found in this file.")

    # Proper frequency-switch reduction: fs = (sA - sB) / sB
    if chunks_A and chunks_B:
        sA = np.nanmean(np.stack(chunks_A), axis=0)
        sB = np.nanmean(np.stack(chunks_B), axis=0)
        n = min(sA.size, sB.size)
        fs = _freq_switch_reduce(sA[:n], sB[:n])
        freq_1d = freq_A[:n]
    elif chunks_A:
        fs = np.nanmean(np.stack(chunks_A), axis=0)
        freq_1d = freq_A[:fs.size]
    else:
        fs = np.nanmean(np.stack(chunks_B), axis=0)
        freq_1d = freq_B[:fs.size]

    if rfi_config is not None:
        fs = _rfi_clean_spectrum(freq_1d, fs, rfi_config)

    stacked = _select_band(freq_1d, fs, fmin_hz, fmax_hz)

    if metric == "mean_power":
        return float(np.nanmean(stacked))

    # fs is already baseline-corrected by the switch reduction;
    # apply a light running-median only to remove any residual slope.
    baseline = _running_median(stacked, width=baseline_width)
    line = stacked - baseline

    if metric == "line_integral":
        return float(np.nansum(line))
    if metric == "peak_excess":
        # nanmax catches single-channel HI lines that 99.5th pct misses.
        # Clamp to zero: negative nanmax means baseline overshot — no real
        # emission at this pointing, treat as zero rather than propagating
        # negative values into the interpolated map.
        return float(max(0.0, float(np.nanmax(line))))
    if metric == "rms_excess":
        return float(np.sqrt(np.nanmean(np.square(line))))
    raise ValueError("metric must be one of: mean_power, line_integral, peak_excess, rms_excess")


def _reduced_spectrum_from_file(
    data: np.lib.npyio.NpzFile,
    spectrum_mode: str = "A+B",
    rfi_config: RfiConfig | None = None,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq_hz, reduced_spectrum) after polarization averaging and switch reduction.

    RFI cleaning is applied to raw sA/sB BEFORE switch reduction, with the
    science band [fmin_hz, fmax_hz] excluded from auto-masking so the HI line
    is never flagged as a spike.
    """
    freq_A = np.asarray(data["freq_hz_A"], dtype=float).ravel() if "freq_hz_A" in data else (
             np.asarray(data["freq_A_hz"], dtype=float).ravel() if "freq_A_hz" in data else None)
    freq_B = np.asarray(data["freq_hz_B"], dtype=float).ravel() if "freq_hz_B" in data else (
             np.asarray(data["freq_B_hz"], dtype=float).ravel() if "freq_B_hz" in data else freq_A)
    if freq_A is None:
        raise ValueError("No frequency axis found in this file.")
    n_freq_A = freq_A.size
    n_freq_B = freq_B.size if freq_B is not None else n_freq_A

    def _get_spec(key, n_freq):
        if key not in data:
            return None
        return _spectrum_to_1d(np.asarray(data[key], dtype=float), n_freq=n_freq)

    chunks_A, chunks_B = [], []
    if spectrum_mode in ("A+B", "A"):
        for k in ("spec_A_pol0", "spec_A_pol1"):
            s = _get_spec(k, n_freq_A)
            if s is not None:
                chunks_A.append(s[:n_freq_A])
    if spectrum_mode in ("A+B", "B"):
        for k in ("spec_B_pol0", "spec_B_pol1"):
            s = _get_spec(k, n_freq_B)
            if s is not None:
                chunks_B.append(s[:n_freq_B])

    if not chunks_A and not chunks_B:
        raise ValueError("No valid spectra found in this file.")

    def _clean_raw(spec, freq, n_freq):
        """RFI-clean a raw spectrum protecting [fmin_hz, fmax_hz] from auto-masking."""
        if rfi_config is None:
            return spec
        f = freq[:n_freq]
        hi_lo = fmin_hz if fmin_hz is not None else (f[len(f) // 2] - 2e6)
        hi_hi = fmax_hz if fmax_hz is not None else (f[len(f) // 2] + 2e6)
        out = spec.copy()
        out_band = (f < hi_lo) | (f > hi_hi)
        if out_band.any():
            out[out_band] = _rfi_clean_spectrum(f[out_band], spec[out_band], rfi_config)
        return out

    if chunks_A and chunks_B:
        sA = np.nanmean(np.stack([_clean_raw(s, freq_A, n_freq_A) for s in chunks_A]), axis=0)
        sB = np.nanmean(np.stack([_clean_raw(s, freq_B, n_freq_B) for s in chunks_B]), axis=0)
        n = min(sA.size, sB.size)
        fs = _freq_switch_reduce(sA[:n], sB[:n])
        freq_1d = freq_A[:n]
    elif chunks_A:
        sA = np.nanmean(np.stack([_clean_raw(s, freq_A, n_freq_A) for s in chunks_A]), axis=0)
        fs = sA
        freq_1d = freq_A[:fs.size]
    else:
        sB = np.nanmean(np.stack([_clean_raw(s, freq_B, n_freq_B) for s in chunks_B]), axis=0)
        fs = sB
        freq_1d = freq_B[:fs.size]

    return freq_1d, fs


def _velocity_spectrum_from_file(
    data: np.lib.npyio.NpzFile,
    spectrum_mode: str = "A+B",
    rfi_config: RfiConfig | None = None,
    velocity_source: str = "fs",
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (freq_hz, spec) for velocity estimation.

    velocity_source:
      - "fs": use frequency-switched reduced spectrum (A-B)/B  [DEFAULT — recommended]
              The switch ratio removes the bandpass, leaving a clean line for
              centroid estimation. RFI cleaning protects the science band.
      - "A": use switched-A spectrum only (raw power — bandpass not removed)
      - "B": use switched-B spectrum only
      - "AplusB": average A and B after interpolating B onto A frequency axis
    """
    if velocity_source == "fs":
        return _reduced_spectrum_from_file(
            data, spectrum_mode=spectrum_mode, rfi_config=rfi_config,
            fmin_hz=fmin_hz, fmax_hz=fmax_hz,
        )

    freq_A = np.asarray(data["freq_hz_A"], dtype=float).ravel() if "freq_hz_A" in data else (
             np.asarray(data["freq_A_hz"], dtype=float).ravel() if "freq_A_hz" in data else None)
    freq_B = np.asarray(data["freq_hz_B"], dtype=float).ravel() if "freq_hz_B" in data else (
             np.asarray(data["freq_B_hz"], dtype=float).ravel() if "freq_B_hz" in data else None)
    if freq_A is None and freq_B is None:
        raise ValueError("No frequency axis found in file.")

    def _avg_pol(prefix: str, n_freq: int | None) -> np.ndarray | None:
        arrs = []
        for k in (f"spec_{prefix}_pol0", f"spec_{prefix}_pol1"):
            if k in data:
                arrs.append(_spectrum_to_1d(np.asarray(data[k], dtype=float), n_freq=n_freq))
        if not arrs:
            return None
        n = min(a.size for a in arrs)
        return np.nanmean(np.stack([a[:n] for a in arrs]), axis=0)

    sA = _avg_pol("A", freq_A.size if freq_A is not None else None)
    sB = _avg_pol("B", freq_B.size if freq_B is not None else None)

    if velocity_source == "A":
        if sA is None or freq_A is None:
            raise ValueError("velocity_source='A' requires A spectra.")
        freq_1d, spec = freq_A[: sA.size], sA
    elif velocity_source == "B":
        if sB is None or freq_B is None:
            raise ValueError("velocity_source='B' requires B spectra.")
        freq_1d, spec = freq_B[: sB.size], sB
    elif velocity_source == "AplusB":
        if sA is None and sB is None:
            raise ValueError("No A/B spectra available.")
        if sA is None:
            freq_1d, spec = freq_B[: sB.size], sB
        elif sB is None:
            freq_1d, spec = freq_A[: sA.size], sA
        else:
            nA = min(freq_A.size, sA.size)
            nB = min(freq_B.size, sB.size)
            fA = freq_A[:nA]
            A = sA[:nA]
            fB = freq_B[:nB]
            B = sB[:nB]
            B_on_A = np.interp(fA, fB, B, left=np.nan, right=np.nan)
            spec = np.nanmean(np.vstack([A, B_on_A]), axis=0)
            freq_1d = fA
    else:
        raise ValueError("velocity_source must be one of: A, B, AplusB, fs")

    # Protect the science band from RFI auto-masking so the HI line survives
    if rfi_config is not None:
        f = freq_1d
        hi_lo = fmin_hz if fmin_hz is not None else (f[len(f) // 2] - 2e6)
        hi_hi = fmax_hz if fmax_hz is not None else (f[len(f) // 2] + 2e6)
        out_band = (f < hi_lo) | (f > hi_hi)
        if out_band.any():
            spec[out_band] = _rfi_clean_spectrum(f[out_band], spec[out_band], rfi_config)
    return freq_1d, spec


def _lsr_correction_kms(
    l_deg: float,
    b_deg: float,
    ra_deg: float | None,
    dec_deg: float | None,
    jd_utc: float | None,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_alt_m: float,
) -> float:
    """Return v_corr to add to topocentric velocity to get LSR velocity.

    Implemented as:
      v_lsr = v_topo + v_helio + v_solar_proj

    where:
      - v_helio is topocentric->heliocentric correction from astropy
      - v_solar_proj projects the Sun's peculiar velocity wrt LSR
        onto the target line of sight.
    """
    if SkyCoord is None or Time is None or EarthLocation is None or u is None:
        raise ImportError("astropy is required for LSR correction. Install astropy.")
    if jd_utc is None or not np.isfinite(jd_utc):
        raise ValueError("jd_pointing is required for topocentric correction.")

    if ra_deg is not None and dec_deg is not None and np.isfinite(ra_deg) and np.isfinite(dec_deg):
        sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    else:
        sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")

    obstime = Time(jd_utc, format="jd", scale="utc")
    location = EarthLocation(lat=obs_lat_deg * u.deg, lon=obs_lon_deg * u.deg, height=obs_alt_m * u.m)

    # Topocentric -> heliocentric (barycentric would also be close; helio is common in radio workflows).
    v_helio = sc.radial_velocity_correction(kind="heliocentric", obstime=obstime, location=location).to(u.km / u.s).value

    # Solar peculiar motion with respect to LSR (Schonrich et al. 2010).
    U, V, W = 11.1, 12.24, 7.25
    l_rad = np.deg2rad(l_deg)
    b_rad = np.deg2rad(b_deg)
    v_pec = U * np.cos(l_rad) * np.cos(b_rad) + V * np.sin(l_rad) * np.cos(b_rad) + W * np.sin(b_rad)
    return float(v_helio + v_pec)


def _velocity_from_file(
    data: np.lib.npyio.NpzFile,
    fmin_hz: float | None,
    fmax_hz: float | None,
    spectrum_mode: str,
    rfi_config: RfiConfig | None,
    baseline_width: int,
    rest_freq_hz: float,
    apply_lsr: bool,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_alt_m: float,
    estimator: str = "centroid",
    velocity_source: str = "fs",
    v_search_kms: float = 80.0,
) -> float:
    """Estimate line peak Doppler velocity in km/s (optionally LSR corrected).

    Mirrors make_sky_projections.py: converts all channels to velocity, then
    restricts peak/centroid search to ±v_search_kms. This excludes the
    frequency-switch echo sidelobe (at ±c*Δf/f_rest km/s) and out-of-window
    RFI that would otherwise bias the centroid to large negative velocities.
    """
    c_kms = 299792.458
    freq_hz, fs = _velocity_spectrum_from_file(
        data,
        spectrum_mode=spectrum_mode,
        rfi_config=rfi_config,
        velocity_source=velocity_source,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )
    fs = _select_band(freq_hz, fs, fmin_hz, fmax_hz)
    freq_hz = _select_band(freq_hz, freq_hz, fmin_hz, fmax_hz)

    # Convert all channels to topocentric velocity — mirrors make_sky_projections.py
    v_topo_all = c_kms * (rest_freq_hz - freq_hz) / rest_freq_hz

    # Apply velocity search window BEFORE baseline/peak estimation.
    # Excludes the freq-switch echo sidelobe and out-of-window RFI.
    v_mask = (v_topo_all >= -v_search_kms) & (v_topo_all <= v_search_kms) & np.isfinite(fs)
    if not np.any(v_mask):
        return float("nan")
    freq_win = freq_hz[v_mask]
    fs_win   = fs[v_mask]

    baseline = _running_median(fs_win, width=baseline_width)
    line = fs_win - baseline
    if not np.any(np.isfinite(line)):
        return float("nan")

    if estimator not in {"peak", "centroid"}:
        raise ValueError("estimator must be 'peak' or 'centroid'")

    if estimator == "peak":
        i_pk = int(np.nanargmax(line))
        f_pk = float(freq_win[i_pk])
    else:
        w = np.clip(line, 0.0, None)
        wsum = float(np.nansum(w))
        if not np.isfinite(wsum) or wsum <= 0:
            i_pk = int(np.nanargmax(line))
            f_pk = float(freq_win[i_pk])
        else:
            f_pk = float(np.nansum(freq_win * w) / wsum)

    v_topo = c_kms * (rest_freq_hz - f_pk) / rest_freq_hz
    if not apply_lsr:
        return float(v_topo)

    l_deg = float(np.asarray(data["l_deg"]).reshape(-1)[0]) if "l_deg" in data else np.nan
    b_deg = float(np.asarray(data["b_deg"]).reshape(-1)[0]) if "b_deg" in data else np.nan
    ra_deg = float(np.asarray(data["ra_deg"]).reshape(-1)[0]) if "ra_deg" in data else np.nan
    dec_deg = float(np.asarray(data["dec_deg"]).reshape(-1)[0]) if "dec_deg" in data else np.nan
    jd = float(np.asarray(data["jd_pointing"]).reshape(-1)[0]) if "jd_pointing" in data else np.nan
    v_corr = _lsr_correction_kms(l_deg, b_deg, ra_deg, dec_deg, jd, obs_lat_deg, obs_lon_deg, obs_alt_m)
    return float(v_topo + v_corr)


def load_survey_points(
    npz_glob: str,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
    spectrum_mode: str = "A+B",
    to_db: bool = True,
    rfi_config: RfiConfig | None = None,
    metric: str = "mean_power",
    baseline_width: int = 31,
) -> list[SurveyPoint]:
    """Load all files and compute one scalar value per (l, b) sample."""
    paths = sorted(glob.glob(npz_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {npz_glob}")

    points: list[SurveyPoint] = []
    for p in paths:
        try:
            with np.load(p) as d:
                if "l_deg" not in d or "b_deg" not in d:
                    continue
                l = float(np.asarray(d["l_deg"]).reshape(-1)[0])
                b = float(np.asarray(d["b_deg"]).reshape(-1)[0])
                if not np.isfinite(l) or not np.isfinite(b):
                    continue
                power = _bandpower_from_file(
                    d,
                    fmin_hz=fmin_hz,
                    fmax_hz=fmax_hz,
                    spectrum_mode=spectrum_mode,
                    rfi_config=rfi_config,
                    metric=metric,
                    baseline_width=baseline_width,
                )
                if to_db and metric == "mean_power":
                    power = 10.0 * np.log10(np.maximum(power, 1e-20))
                points.append(SurveyPoint(l_deg=l, b_deg=b, bandpower=power, source_file=p))
        except Exception:
            # Skip damaged or partial files while keeping the workflow robust.
            continue
    if not points:
        raise RuntimeError("No valid survey points could be extracted from matched files.")
    return points


def load_velocity_points(
    npz_glob: str,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
    spectrum_mode: str = "A+B",
    rfi_config: RfiConfig | None = None,
    baseline_width: int = 31,
    rest_freq_hz: float = 1420.40575177e6,
    apply_lsr: bool = True,
    obs_lat_deg: float = 37.9183,
    obs_lon_deg: float = -122.1537,
    obs_alt_m: float = 304.0,
    estimator: str = "centroid",
    velocity_source: str = "fs",
    v_search_kms: float = 80.0,
) -> list[SurveyPoint]:
    """Load files and compute one LSR-corrected Doppler velocity per pointing."""
    paths = sorted(glob.glob(npz_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {npz_glob}")

    points: list[SurveyPoint] = []
    for p in paths:
        try:
            with np.load(p) as d:
                if "l_deg" not in d or "b_deg" not in d:
                    continue
                l = float(np.asarray(d["l_deg"]).reshape(-1)[0])
                b = float(np.asarray(d["b_deg"]).reshape(-1)[0])
                if not np.isfinite(l) or not np.isfinite(b):
                    continue
                vel = _velocity_from_file(
                    d,
                    fmin_hz=fmin_hz,
                    fmax_hz=fmax_hz,
                    spectrum_mode=spectrum_mode,
                    rfi_config=rfi_config,
                    baseline_width=baseline_width,
                    rest_freq_hz=rest_freq_hz,
                    apply_lsr=apply_lsr,
                    obs_lat_deg=obs_lat_deg,
                    obs_lon_deg=obs_lon_deg,
                    obs_alt_m=obs_alt_m,
                    estimator=estimator,
                    velocity_source=velocity_source,
                    v_search_kms=v_search_kms,
                )
                if np.isfinite(vel):
                    points.append(SurveyPoint(l_deg=l, b_deg=b, bandpower=float(vel), source_file=p))
        except Exception:
            continue
    if not points:
        raise RuntimeError("No valid velocity points could be extracted from matched files.")
    return points


def _unwrap_longitudes(l_deg: np.ndarray, wrap_center_deg: float = 180.0) -> np.ndarray:
    """Wrap to [center-180, center+180) for stable interpolation near l=0 seam."""
    return ((l_deg - wrap_center_deg + 180.0) % 360.0) + wrap_center_deg - 180.0


def _make_grid(lmin: float, lmax: float, bmin: float, bmax: float, grid_step_deg: float) -> tuple[np.ndarray, np.ndarray]:
    l_grid = np.arange(lmin, lmax + grid_step_deg, grid_step_deg, dtype=float)
    b_grid = np.arange(bmin, bmax + grid_step_deg, grid_step_deg, dtype=float)
    ll, bb = np.meshgrid(l_grid, b_grid)
    return ll, bb


def _aggregate_duplicate_points(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    z: np.ndarray,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse duplicate (l,b) locations by taking median bandpower.
    This prevents singular RBF systems when repeated sky pointings exist.
    """
    key_l = np.round(l_deg, round_decimals)
    key_b = np.round(b_deg, round_decimals)
    uniq = {}
    for i in range(len(z)):
        key = (float(key_l[i]), float(key_b[i]))
        uniq.setdefault(key, []).append(float(z[i]))

    l_out, b_out, z_out = [], [], []
    for (l_i, b_i), vals in uniq.items():
        l_out.append(l_i)
        b_out.append(b_i)
        z_out.append(float(np.nanmedian(vals)))
    return np.asarray(l_out), np.asarray(b_out), np.asarray(z_out)


def interpolate_map(
    points: Sequence[SurveyPoint],
    lmin: float = 210.0,
    lmax: float = 380.0,
    bmin: float = 0.0,
    bmax: float = 90.0,
    grid_step_deg: float = 1.0,
    epsilon_deg: float = 0.25,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate sparse 4-degree samples onto a dense map grid.

    epsilon_deg controls interpolation smoothness:
      - lower -> sharper / more local structure
      - higher -> smoother / broader structure
    """
    if RBFInterpolator is None and griddata is None:
        raise ImportError("scipy is required for interpolation: pip install scipy")
    if epsilon_deg <= 0:
        raise ValueError("epsilon_deg must be > 0")

    l_obs = np.array([p.l_deg for p in points], dtype=float)
    b_obs = np.array([p.b_deg for p in points], dtype=float)
    z_obs = np.array([p.bandpower for p in points], dtype=float)
    l_obs, b_obs, z_obs = _aggregate_duplicate_points(l_obs, b_obs, z_obs)
    if l_obs.size < 4:
        raise ValueError(
            "Not enough unique sky points for RBF interpolation. "
            "Need at least 4 unique (l,b) samples."
        )

    l_obs_unwrapped = _unwrap_longitudes(l_obs, wrap_center_deg=295.0)
    ll, bb = _make_grid(lmin, lmax, bmin, bmax, grid_step_deg)
    ll_unwrapped = _unwrap_longitudes(ll, wrap_center_deg=295.0)

    xy_obs = np.column_stack([l_obs_unwrapped, b_obs])
    xy_grid = np.column_stack([ll_unwrapped.ravel(), bb.ravel()])

    if method == "linear" and griddata is not None:
        zz_linear = griddata(xy_obs, z_obs, xy_grid, method="linear")
        zz_nearest = griddata(xy_obs, z_obs, xy_grid, method="nearest")
        zz = np.where(np.isfinite(zz_linear), zz_linear, zz_nearest).reshape(ll.shape)
        return ll, bb, zz

    # Robust interpolation chain for repeated/degenerate pointings:
    # 1) Global RBF (exact), 2) Regularized global RBF,
    # 3) Local-neighbor RBF, 4) griddata linear + nearest fill.
    n_obs = xy_obs.shape[0]
    trial_settings = [
        {"smoothing": 0.0, "neighbors": None},
        {"smoothing": 1e-6, "neighbors": None},
        {"smoothing": 1e-3, "neighbors": None},
        {"smoothing": 1e-6, "neighbors": max(8, min(32, n_obs - 1))},
        {"smoothing": 1e-3, "neighbors": max(8, min(32, n_obs - 1))},
    ]
    for cfg in trial_settings:
        try:
            rbf = RBFInterpolator(
                y=xy_obs,
                d=z_obs,
                kernel="gaussian",
                epsilon=epsilon_deg,
                smoothing=cfg["smoothing"],
                neighbors=cfg["neighbors"],
            )
            zz = rbf(xy_grid).reshape(ll.shape)
            if np.any(np.isfinite(zz)):
                return ll, bb, zz
        except (np.linalg.LinAlgError, ValueError):
            continue

    if griddata is None:
        raise np.linalg.LinAlgError(
            "Interpolation failed: RBF system remained singular and griddata fallback unavailable."
        )

    zz_linear = griddata(xy_obs, z_obs, xy_grid, method="linear")
    zz_nearest = griddata(xy_obs, z_obs, xy_grid, method="nearest")
    zz = np.where(np.isfinite(zz_linear), zz_linear, zz_nearest).reshape(ll.shape)
    return ll, bb, zz


def _smooth_nan_grid(zz: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Gaussian smooth while preserving NaN mask."""
    if gaussian_filter is None or sigma_pix <= 0:
        return zz
    val = np.nan_to_num(zz, nan=0.0)
    wgt = np.isfinite(zz).astype(float)
    val_s = gaussian_filter(val, sigma=sigma_pix, mode="nearest")
    wgt_s = gaussian_filter(wgt, sigma=sigma_pix, mode="nearest")
    out = np.full_like(zz, np.nan, dtype=float)
    ok = wgt_s > 1e-6
    out[ok] = val_s[ok] / wgt_s[ok]
    return out


def _apply_footprint_mask(
    ll_deg: np.ndarray,
    bb_deg: np.ndarray,
    points: Sequence[SurveyPoint],
    radius_deg: float,
) -> np.ndarray:
    """Mask interpolated map outside nearest-neighbor footprint radius."""
    if cKDTree is None or radius_deg <= 0:
        return np.ones_like(ll_deg, dtype=bool)
    l_obs = np.array([p.l_deg for p in points], dtype=float)
    b_obs = np.array([p.b_deg for p in points], dtype=float)
    l_obs_u = _unwrap_longitudes(l_obs, wrap_center_deg=295.0)
    ll_u = _unwrap_longitudes(ll_deg, wrap_center_deg=295.0)
    tree = cKDTree(np.column_stack([l_obs_u, b_obs]))
    d, _ = tree.query(np.column_stack([ll_u.ravel(), bb_deg.ravel()]), k=1)
    return (d.reshape(ll_deg.shape) <= radius_deg)


def plot_constrained_nps_map(
    ll_deg: np.ndarray,
    bb_deg: np.ndarray,
    zz: np.ndarray,
    points: Sequence[SurveyPoint],
    title: str = "NPS Supershell: Interpolated Bandpower Map",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    show_samples: bool = True,
    colorbar_label: str = "Bandpower (dB)",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11, 6))
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(ll_deg, bb_deg, zz, shading="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label)

    if show_samples:
        ax.scatter([p.l_deg for p in points], [p.b_deg for p in points], s=8, c="white", alpha=0.4, label="4 deg samples")
        ax.legend(loc="upper right")

    ax.set_xlabel("Galactic Longitude l (deg)")
    ax.set_ylabel("Galactic Latitude b (deg)")
    ax.set_title(title)
    ax.set_xlim(float(np.nanmin(ll_deg)), float(np.nanmax(ll_deg)))
    ax.set_ylim(float(np.nanmin(bb_deg)), float(np.nanmax(bb_deg)))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax



def plot_mollweide_nps_map(
    ll_deg: np.ndarray,
    bb_deg: np.ndarray,
    zz: np.ndarray,
    points: Sequence[SurveyPoint],
    title: str = "NPS Supershell",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    center_l_deg: float = 300.0,
    colorbar_label: str = "peak_excess (arb. units)",
    show_samples: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Mollweide projection centered on center_l_deg (default 300°, NPS region).

    Longitude increases to the LEFT following standard astronomy convention.
    """
    # Shift longitudes so center_l_deg maps to 0 in the projection
    ll_shift = ((ll_deg - center_l_deg + 180.0) % 360.0) - 180.0
    ll_rad   = np.deg2rad(ll_shift)
    bb_rad   = np.deg2rad(bb_deg)

    # Sort columns so longitude increases monotonically for pcolormesh
    col_order = np.argsort(ll_shift[0])
    ll_rad_s  = ll_rad[:, col_order]
    bb_rad_s  = bb_rad[:, col_order]
    zz_s      = zz[:, col_order]

    fig = plt.figure(figsize=(13, 7))
    ax  = fig.add_subplot(111, projection="mollweide")

    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(ll_rad_s, bb_rad_s, zz_s,
                       cmap=cmap, norm=norm, shading="auto")

    # Longitude tick labels: convert back to galactic l
    tick_offsets_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    tick_l_gal = (tick_offsets_deg + center_l_deg) % 360
    ax.set_xticklabels([f"{int(l)}°" for l in tick_l_gal], fontsize=8)

    # Sample positions
    if show_samples and points:
        sl = np.array([p.l_deg for p in points])
        sb = np.array([p.b_deg for p in points])
        sl_shift = np.deg2rad(((sl - center_l_deg + 180.0) % 360.0) - 180.0)
        ax.scatter(sl_shift, np.deg2rad(sb),
                   s=6, c="white", alpha=0.4, zorder=5, label="4 deg samples")
        ax.legend(loc="lower right", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.07, shrink=0.75)
    cbar.set_label(colorbar_label)
    ax.set_title(title, pad=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def run(
    npz_glob: str,
    output_png: str | Path,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
    spectrum_mode: str = "A+B",
    epsilon_deg: float = 0.25,
    grid_step_deg: float = 1.0,
    lmin: float = 210.0,
    lmax: float = 380.0,
    bmin: float = 0.0,
    bmax: float = 90.0,
    rfi_sigma_clip: float = 6.0,
    rfi_edge_trim: int = 8,
    rfi_patch: bool = True,
    rfi_ranges_hz: Sequence[tuple[float, float]] = (),
    metric: str = "mean_power",
    baseline_width: int = 31,
    interpolation_method: str = "linear",
    smooth_sigma_deg: float = 1.5,
    footprint_radius_deg: float = 6.0,
    show_samples: bool = False,
) -> Path:
    rfi_cfg = RfiConfig(
        sigma_clip=rfi_sigma_clip,
        edge_trim=rfi_edge_trim,
        patch_masked=rfi_patch,
        manual_ranges_hz=tuple(rfi_ranges_hz),
    )
    points = load_survey_points(
        npz_glob=npz_glob,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        spectrum_mode=spectrum_mode,
        to_db=False,
        rfi_config=rfi_cfg,
        metric=metric,
        baseline_width=baseline_width,
    )
    ll, bb, zz = interpolate_map(
        points,
        lmin=lmin,
        lmax=lmax,
        bmin=bmin,
        bmax=bmax,
        grid_step_deg=grid_step_deg,
        epsilon_deg=epsilon_deg,
        method=interpolation_method,
    )
    sigma_pix = max(0.0, smooth_sigma_deg / max(grid_step_deg, 1e-6))
    zz = _smooth_nan_grid(zz, sigma_pix=sigma_pix)
    keep = _apply_footprint_mask(ll, bb, points, radius_deg=footprint_radius_deg)
    zz = np.where(keep, zz, np.nan)

    title = f"NPS Supershell: Interpolated Map (metric={metric}, epsilon={epsilon_deg})"
    colorbar_label = "Bandpower (dB)" if metric == "mean_power" else f"{metric} (arb. units)"
    finite = zz[np.isfinite(zz)]
    vmin = 0.0 if metric == "peak_excess" else float(np.nanpercentile(zz, 2))
    vmax = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
    fig, _ = plot_constrained_nps_map(
        ll,
        bb,
        zz,
        points,
        title=title,
        colorbar_label=colorbar_label,
        vmin=vmin,
        vmax=vmax,
        show_samples=show_samples,
    )

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    # Also save a Mollweide projection centered on the NPS (l=300°)
    moll_png = output_png.with_name(output_png.stem + "_mollweide" + output_png.suffix)
    fig_m, _ = plot_mollweide_nps_map(
        ll, bb, zz, points,
        title=title,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        center_l_deg=300.0,
        colorbar_label=colorbar_label,
        show_samples=show_samples,
    )
    fig_m.savefig(moll_png, dpi=180)
    plt.close(fig_m)

    return output_png


def run_velocity_map(
    npz_glob: str,
    output_png: str | Path,
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
    spectrum_mode: str = "A+B",
    grid_step_deg: float = 1.0,
    lmin: float = 210.0,
    lmax: float = 380.0,
    bmin: float = 0.0,
    bmax: float = 90.0,
    rfi_sigma_clip: float = 6.0,
    rfi_edge_trim: int = 8,
    rfi_patch: bool = True,
    rfi_ranges_hz: Sequence[tuple[float, float]] = (),
    baseline_width: int = 31,
    rest_freq_hz: float = 1420.40575177e6,
    apply_lsr: bool = True,
    obs_lat_deg: float = 37.9183,
    obs_lon_deg: float = -122.1537,
    obs_alt_m: float = 304.0,
    interpolation_method: str = "linear",
    epsilon_deg: float = 0.3,
    smooth_sigma_deg: float = 0.8,
    footprint_radius_deg: float = 0.0,
    show_samples: bool = False,
    velocity_estimator: str = "centroid",
    velocity_source: str = "fs",
    v_search_kms: float = 80.0,
) -> Path:
    """Create Doppler velocity map (km/s), with optional LSR correction."""
    rfi_cfg = RfiConfig(
        sigma_clip=rfi_sigma_clip,
        edge_trim=rfi_edge_trim,
        patch_masked=rfi_patch,
        manual_ranges_hz=tuple(rfi_ranges_hz),
    )
    points = load_velocity_points(
        npz_glob=npz_glob,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        spectrum_mode=spectrum_mode,
        rfi_config=rfi_cfg,
        baseline_width=baseline_width,
        rest_freq_hz=rest_freq_hz,
        apply_lsr=apply_lsr,
        obs_lat_deg=obs_lat_deg,
        obs_lon_deg=obs_lon_deg,
        obs_alt_m=obs_alt_m,
        estimator=velocity_estimator,
        velocity_source=velocity_source,
        v_search_kms=v_search_kms,
    )
    ll, bb, zz = interpolate_map(
        points,
        lmin=lmin,
        lmax=lmax,
        bmin=bmin,
        bmax=bmax,
        grid_step_deg=grid_step_deg,
        epsilon_deg=epsilon_deg,
        method=interpolation_method,
    )
    sigma_pix = max(0.0, smooth_sigma_deg / max(grid_step_deg, 1e-6))
    zz = _smooth_nan_grid(zz, sigma_pix=sigma_pix)
    if footprint_radius_deg > 0:
        keep = _apply_footprint_mask(ll, bb, points, radius_deg=footprint_radius_deg)
        zz = np.where(keep, zz, np.nan)

    finite = zz[np.isfinite(zz)]
    if finite.size:
        vmax = float(np.nanpercentile(np.abs(finite), 95))
        vmin = -vmax
    else:
        vmin, vmax = -20.0, 20.0
    title = "NPS Supershell: LSR Doppler Velocity Map"
    label = "v_lsr (km/s)"
    fig, _ = plot_constrained_nps_map(
        ll,
        bb,
        zz,
        points,
        title=title,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        show_samples=show_samples,
        colorbar_label=label,
    )
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    moll_png = output_png.with_name(output_png.stem + "_mollweide" + output_png.suffix)
    fig_m, _ = plot_mollweide_nps_map(
        ll,
        bb,
        zz,
        points,
        title=title,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        center_l_deg=300.0,
        colorbar_label=label,
        show_samples=show_samples,
    )
    fig_m.savefig(moll_png, dpi=180)
    plt.close(fig_m)
    return output_png


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interpolated NPS heatmap from .npz survey files.")
    parser.add_argument("--npz-glob", default="nps_data/*.npz", help="Glob pattern for survey files.")
    parser.add_argument("--out", default="nps_heatmap.png", help="Output image path.")
    parser.add_argument("--fmin-hz", type=float, default=None, help="Lower frequency bound for bandpower.")
    parser.add_argument("--fmax-hz", type=float, default=None, help="Upper frequency bound for bandpower.")
    parser.add_argument("--mode", choices=["A+B", "A", "B"], default="A+B", help="Which switched spectra to use.")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Gaussian interpolation width (deg).")
    parser.add_argument("--grid-step", type=float, default=1.0, help="Output map grid spacing (deg).")
    parser.add_argument("--lmin", type=float, default=210.0)
    parser.add_argument("--lmax", type=float, default=380.0, help="Use >360 to include wrap region (e.g., 380 == 20 deg).")
    parser.add_argument("--bmin", type=float, default=0.0)
    parser.add_argument("--bmax", type=float, default=90.0)
    parser.add_argument("--rfi-sigma", type=float, default=6.0, help="Robust sigma threshold for auto RFI masking.")
    parser.add_argument("--rfi-edge-trim", type=int, default=8, help="Channels trimmed from each band edge before averaging.")
    parser.add_argument("--no-rfi-patch", action="store_true", help="Mask RFI channels without interpolation patching.")
    parser.add_argument(
        "--metric",
        choices=["mean_power", "line_integral", "peak_excess", "rms_excess"],
        default="mean_power",
        help="Scalar extracted from spectra for map intensity.",
    )
    parser.add_argument(
        "--baseline-width",
        type=int,
        default=31,
        help="Running-median window for line-based metrics.",
    )
    parser.add_argument(
        "--interp-method",
        choices=["linear", "rbf"],
        default="linear",
        help="Spatial interpolation method.",
    )
    parser.add_argument(
        "--smooth-sigma-deg",
        type=float,
        default=1.5,
        help="Post-interpolation Gaussian smoothing scale (deg).",
    )
    parser.add_argument(
        "--footprint-radius-deg",
        type=float,
        default=6.0,
        help="Mask map farther than this from nearest sample point.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Overlay sample/grid point markers on the plots.",
    )
    parser.add_argument(
        "--rfi-range-hz",
        nargs=2,
        action="append",
        metavar=("FREQ_LOW_HZ", "FREQ_HIGH_HZ"),
        default=[],
        help="Manual RFI mask range; can be repeated.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    manual_ranges = [(float(lo), float(hi)) for lo, hi in args.rfi_range_hz]
    out = run(
        npz_glob=args.npz_glob,
        output_png=args.out,
        fmin_hz=args.fmin_hz,
        fmax_hz=args.fmax_hz,
        spectrum_mode=args.mode,
        epsilon_deg=args.epsilon,
        grid_step_deg=args.grid_step,
        lmin=args.lmin,
        lmax=args.lmax,
        bmin=args.bmin,
        bmax=args.bmax,
        rfi_sigma_clip=args.rfi_sigma,
        rfi_edge_trim=args.rfi_edge_trim,
        rfi_patch=not args.no_rfi_patch,
        rfi_ranges_hz=manual_ranges,
        metric=args.metric,
        baseline_width=args.baseline_width,
        interpolation_method=args.interp_method,
        smooth_sigma_deg=args.smooth_sigma_deg,
        footprint_radius_deg=args.footprint_radius_deg,
        show_samples=args.show_samples,
    )
    print(f"Saved heatmap: {out}")


if __name__ == "__main__":
    main()
