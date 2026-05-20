"""
plot_bessel_fit.py
──────────────────────────────────────────────────────────────────────────────
Plot the observed fringe modulator (MF_obs) vs local fringe frequency, with
the best-fit theoretical modulator (MF_theory) for a uniform circular disk
overlaid — i.e. the Bessel-function fit used to measure the Sun's diameter.

Public API
----------
plot_visibility_with_bessel_fit(
    h_s, F_obs, nonlin_result,
    delta, b_ew, b_ns, lat, lam,
    *,
    n_bins=40,
    R_range_rad=None,
    n_grid=2000,
    show_residuals=True,
    show_zero_crossings=True,
    title="Sun Diameter: Fringe Modulator Fit",
    ax=None,
)  →  fig, axes, results_dict

Quick usage
-----------
from plot_bessel_fit import plot_visibility_with_bessel_fit

fig, axes, res = plot_visibility_with_bessel_fit(
    h_s, band_real_filt, nl,
    delta=delta, b_ew=b_ew_nl, b_ns=b_ns_nl,
    lat=p.lat, lam=p.lam,
)
print(f"Sun diameter: {res['diameter_deg']:.3f} deg  "
      f"(true ≈ 0.53 deg)")
fig.savefig("sun_bessel_fit.png", dpi=150)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

# ── rafpy imports ────────────────────────────────────────────────────────────
from rafpy.modulator import mf_theory, mf_observed, fit_diameter
from rafpy.fringe     import local_fringe_freq


# ── Aesthetics ───────────────────────────────────────────────────────────────

_PALETTE = {
    "bg"      : "#0f1117",
    "panel"   : "#181c27",
    "grid"    : "#2a2f3e",
    "accent1" : "#4fc3f7",   # ice blue  – MF_obs data
    "accent2" : "#ff6b6b",   # coral red – MF_theory fit
    "accent3" : "#ffd166",   # amber     – zero crossings
    "text"    : "#e0e6f0",
    "subtext" : "#7b8cac",
}

def _apply_style(ax):
    ax.set_facecolor(_PALETTE["panel"])
    ax.tick_params(colors=_PALETTE["subtext"], which="both", labelsize=9)
    ax.xaxis.label.set_color(_PALETTE["text"])
    ax.yaxis.label.set_color(_PALETTE["text"])
    ax.title.set_color(_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE["grid"])
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.6, linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# ── Zero-crossing finder ─────────────────────────────────────────────────────

def _find_zero_crossings(x, y, n_max=5):
    """Return x-values where y passes through zero (sign changes)."""
    crossings = []
    for i in range(len(y) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] * y[i + 1] < 0:
            # Linear interpolation
            x_zero = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            crossings.append(x_zero)
            if len(crossings) == n_max:
                break
    return np.array(crossings)


# ── Main function ────────────────────────────────────────────────────────────

def plot_visibility_with_bessel_fit(
    h_s,
    F_obs,
    nonlin_result,
    delta,
    b_ew,
    b_ns,
    lat,
    lam,
    *,
    n_bins=40,
    R_range_rad=None,
    n_grid=2000,
    show_residuals=True,
    show_zero_crossings=True,
    title="Sun Diameter: Fringe Modulator Fit",
    figsize=None,
):
    """
    Compute and plot MF_obs vs local fringe frequency, overlaid with the
    best-fit MF_theory for a uniformly-bright circular disk.

    Parameters
    ----------
    h_s            : (N,) array  Hour angles, radians
    F_obs          : (N,) array  Observed real fringe (band-averaged)
    nonlin_result  : dict        Output of rafpy.fitting.nonlinear_fit()
    delta          : float       Source declination, radians
    b_ew, b_ns     : float       Fitted baseline components, metres
    lat            : float       Observatory latitude, radians
    lam            : float       Wavelength, metres
    n_bins         : int         Number of ff bins for MF_obs (default 40)
    R_range_rad    : (lo, hi)    Search range for radius in radians.
                                  Default: 0.1° – 1.0°
    n_grid         : int         Grid points for chi-squared fit (default 2000)
    show_residuals : bool        Show residual panel below main plot
    show_zero_crossings : bool   Annotate zero crossings of theory curve
    title          : str         Figure title
    figsize        : tuple       Override figure size

    Returns
    -------
    fig            : matplotlib Figure
    axes           : list of Axes (main [, residual])
    results        : dict with keys
        'mf_obs_result'   – output of mf_observed()
        'diameter_result' – output of fit_diameter()
        'diameter_deg'    – best-fit diameter in degrees
        'R_arcmin'        – best-fit radius in arcminutes
        'zero_crossings_obs'    – ff values of MF_obs zero crossings
        'zero_crossings_theory' – ff*R values of MF_theory zero crossings
    """
    # ── 1. Compute MF_obs ────────────────────────────────────────────────────
    mf_obs_r = mf_observed(
        h_s, F_obs, nonlin_result,
        delta=delta, b_ew=b_ew, b_ns=b_ns,
        lat=lat, lam=lam,
        n_bins=n_bins,
    )

    # ── 2. Fit diameter ──────────────────────────────────────────────────────
    if R_range_rad is None:
        R_range_rad = (np.radians(0.1), np.radians(1.0))

    diam_r = fit_diameter(mf_obs_r, R_range_rad=R_range_rad, n_grid=n_grid)
    R_best = diam_r["R_rad"]

    # ── 3. Build theory curve ────────────────────────────────────────────────
    ff_rad   = mf_obs_r["ff_rad_bins"]
    mf_obs   = mf_obs_r["mf_obs"]
    mf_err   = mf_obs_r["mf_err"]
    n_in_bin = mf_obs_r["n_in_bin"]

    ff_fine    = np.linspace(0, ff_rad.max() * 1.25, 2000)
    mf_th_fine = mf_theory(ff_fine * R_best)

    # Normalise theory to data at lowest ff where data is finite
    good = np.isfinite(mf_obs)
    if good.any():
        i0 = np.where(good)[0][0]
        th_at_i0 = mf_theory(np.array([ff_rad[i0] * R_best]))[0]
        norm = np.nanmean(mf_obs[good][:3]) if mf_obs[good][0] != 0 else 1.0
    else:
        norm = 1.0
    mf_th_fine_normed = mf_th_fine / norm

    # Zero crossings
    zc_theory_ffR = _find_zero_crossings(ff_fine * R_best, mf_th_fine)  # in ff*R units
    zc_theory_ff  = zc_theory_ffR / R_best                              # in ff units
    zc_obs        = _find_zero_crossings(ff_rad[good], mf_obs[good])

    # ── 4. Layout ────────────────────────────────────────────────────────────
    n_rows   = 3 if show_residuals else 2
    row_rats = [4, 1.5, 1] if show_residuals else [4, 1]
    if figsize is None:
        figsize = (11, 8) if show_residuals else (11, 6.5)

    fig = plt.figure(figsize=figsize, facecolor=_PALETTE["bg"])
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig,
                            height_ratios=row_rats,
                            hspace=0.08)

    ax_main = fig.add_subplot(gs[0])
    ax_chi2 = fig.add_subplot(gs[1], sharex=None)
    axes    = [ax_main, ax_chi2]
    if show_residuals:
        ax_res = fig.add_subplot(gs[2], sharex=ax_main)
        axes.append(ax_res)

    for ax in axes:
        _apply_style(ax)

    # ── 5. Main panel: MF_obs + MF_theory ───────────────────────────────────
    # Error band
    ax_main.fill_between(
        ff_rad[good],
        (mf_obs - mf_err)[good],
        (mf_obs + mf_err)[good],
        color=_PALETTE["accent1"], alpha=0.18, label="_nolegend_",
    )
    # Data points coloured by bin count
    sc = ax_main.scatter(
        ff_rad[good], mf_obs[good],
        c=n_in_bin[good], cmap="Blues",
        s=55, zorder=4, edgecolors=_PALETTE["accent1"],
        linewidths=0.7, label="MF$_{obs}$",
    )
    ax_main.errorbar(
        ff_rad[good], mf_obs[good], yerr=mf_err[good],
        fmt="none", ecolor=_PALETTE["accent1"], elinewidth=0.8,
        capsize=2.5, alpha=0.6, zorder=3,
    )

    # Theory fit line
    ax_main.plot(
        ff_fine, mf_th_fine_normed,
        color=_PALETTE["accent2"], lw=2.2, zorder=5,
        label=(
            f"MF$_{{theory}}$ — uniform disk\n"
            f"  R = {diam_r['R_arcmin']:.2f}′ "
            f"  d = {diam_r['diameter_deg']:.3f}°"
        ),
    )

    # Zero-crossing markers on theory curve
    if show_zero_crossings:
        for i, ff_zc in enumerate(zc_theory_ff):
            if ff_zc <= ff_fine.max():
                ax_main.axvline(
                    ff_zc, color=_PALETTE["accent3"],
                    lw=0.9, ls=":", alpha=0.7,
                    label="Theory zero-crossings" if i == 0 else "_nolegend_",
                )
        # Mark obs zero crossings
        for i, ff_zc in enumerate(zc_obs):
            ax_main.axvline(
                ff_zc, color="#b0f2b6",
                lw=0.9, ls="--", alpha=0.55,
                label="Obs zero-crossings" if i == 0 else "_nolegend_",
            )

    ax_main.axhline(0, color=_PALETTE["subtext"], lw=0.8, ls="-")
    ax_main.set_ylabel("Fringe Modulator  MF", color=_PALETTE["text"], fontsize=11)
    ax_main.set_title(title, color=_PALETTE["text"], fontsize=13, pad=10,
                      fontweight="bold")
    legend = ax_main.legend(
        loc="upper right", fontsize=9,
        facecolor=_PALETTE["panel"], edgecolor=_PALETTE["grid"],
        labelcolor=_PALETTE["text"],
    )
    plt.colorbar(sc, ax=ax_main, label="Samples per bin",
                 pad=0.01).ax.yaxis.label.set_color(_PALETTE["subtext"])
    ax_main.tick_params(labelbottom=False)

    # ── 6. Chi-squared panel ─────────────────────────────────────────────────
    R_arr  = diam_r["R_grid"]
    chi2   = diam_r["chi2_grid"]
    R_degs = np.degrees(R_arr) * 60   # arcminutes

    ax_chi2.plot(R_degs, chi2, color=_PALETTE["accent1"], lw=1.4)
    ax_chi2.axvline(
        diam_r["R_arcmin"], color=_PALETTE["accent2"],
        lw=1.6, ls="--",
        label=f"Best R = {diam_r['R_arcmin']:.2f}′",
    )
    ax_chi2.set_ylabel("χ²", color=_PALETTE["text"], fontsize=10)
    ax_chi2.legend(fontsize=8, facecolor=_PALETTE["panel"],
                   edgecolor=_PALETTE["grid"], labelcolor=_PALETTE["text"])
    if not show_residuals:
        ax_chi2.set_xlabel("Angular Radius R  (arcmin)", color=_PALETTE["text"])
    else:
        ax_chi2.tick_params(labelbottom=False)

    # ── 7. Residuals panel ───────────────────────────────────────────────────
    if show_residuals:
        # Theory evaluated at data ff positions
        mf_th_at_data = mf_theory(ff_rad[good] * R_best) / norm
        resid = mf_obs[good] - mf_th_at_data

        ax_res.axhline(0, color=_PALETTE["subtext"], lw=0.8)
        ax_res.fill_between(
            ff_rad[good], resid, 0,
            color=_PALETTE["accent2"], alpha=0.35,
        )
        ax_res.scatter(ff_rad[good], resid, s=20,
                       color=_PALETTE["accent2"], zorder=3)
        ax_res.set_ylabel("Resid.", color=_PALETTE["text"], fontsize=9)
        ax_res.set_xlabel(
            "Local Fringe Frequency  $f_f$  (cycles rad$^{-1}$)",
            color=_PALETTE["text"], fontsize=11,
        )

    # Shared x-axis label when no residuals
    if not show_residuals:
        ax_chi2.set_xlabel(
            "Local Fringe Frequency  $f_f$  (cycles rad$^{-1}$)",
            color=_PALETTE["text"], fontsize=11,
        )

    # ── 8. Annotation box ────────────────────────────────────────────────────
    true_diam = 0.5307    # deg
    meas_diam = diam_r["diameter_deg"]
    err_pct   = abs(meas_diam - true_diam) / true_diam * 100

    box_txt = (
        f"Best-fit radius  R = {diam_r['R_arcmin']:.2f}′\n"
        f"Diameter         d = {meas_diam:.3f}°\n"
        f"True Sun diam  ≈ {true_diam:.3f}°\n"
        f"Residual error    {err_pct:.1f}%"
    )
    ax_main.text(
        0.02, 0.04, box_txt,
        transform=ax_main.transAxes,
        fontsize=8.5, family="monospace",
        color=_PALETTE["accent3"],
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=_PALETTE["bg"],
            edgecolor=_PALETTE["accent3"],
            alpha=0.85,
        ),
        verticalalignment="bottom",
        zorder=10,
    )

    fig.tight_layout()

    results = {
        "mf_obs_result"        : mf_obs_r,
        "diameter_result"      : diam_r,
        "diameter_deg"         : meas_diam,
        "R_arcmin"             : diam_r["R_arcmin"],
        "zero_crossings_obs"   : zc_obs,
        "zero_crossings_theory": zc_theory_ffR,
    }

    return fig, axes, results


# ── CLI demo with synthetic data ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke-test: synthesise a fringe from a uniform disk of known radius
    and verify the fit recovers it.
    """
    import sys, textwrap
    sys.path.insert(0, ".")

    print("Generating synthetic fringe for R = 0.265 deg (Sun-like)...")

    from rafpy.params import InterfParams, C
    from rafpy.fringe  import fringe_model, local_fringe_freq, ha_from_lst_ra
    from rafpy.fitting import brute_force_fit, nonlinear_fit
    from rafpy.modulator import mf_theory

    # System params
    p = InterfParams(b_ew=20.4, b_ns=0.2, freq_rf=10.674e9)

    # Source: Sun at dec ~ +10 deg
    delta    = np.radians(10.0)
    R_true   = np.radians(0.265)   # true radius
    n_t      = 800
    h_s      = np.linspace(np.radians(-60), np.radians(60), n_t)

    # Point-source fringe
    F_pt = fringe_model(h_s, A=1.0, B=0.3,
                        delta=delta, b_ew=p.b_ew, b_ns=p.b_ns,
                        lat=p.lat, lam=p.lam)

    # Fringe modulated by uniform disk
    ff_rad = local_fringe_freq(h_s, delta, p.b_ew, p.b_ns, p.lat, p.lam, in_hz=False)
    MF     = mf_theory(ff_rad * R_true)
    F_obs  = F_pt * MF + np.random.normal(0, 0.05, n_t)   # add noise

    # Brute-force + nonlinear fit
    bf = brute_force_fit(h_s, F_obs, n_ew=500, b_ew_approx=p.b_ew, lam=p.lam)
    nl = nonlinear_fit(h_s, F_obs, p0_dict=bf)

    print(f"Nonlinear fit: Q_ew={nl['Q_ew']:.4f}, A={nl['A']:.4f}, B={nl['B']:.4f}")

    fig, axes, res = plot_visibility_with_bessel_fit(
        h_s, F_obs, nl,
        delta=delta, b_ew=p.b_ew, b_ns=p.b_ns,
        lat=p.lat, lam=p.lam,
        n_bins=35,
        R_range_rad=(np.radians(0.1), np.radians(0.8)),
        title="Synthetic Sun — Fringe Modulator Bessel Fit",
    )

    print(textwrap.dedent(f"""
    ── Results ───────────────────────────────────
    True   radius : {np.degrees(R_true)*60:.2f} arcmin
    Fitted radius : {res['R_arcmin']:.2f} arcmin
    True   diam   : {np.degrees(R_true)*2:.3f} deg
    Fitted diam   : {res['diameter_deg']:.3f} deg
    ──────────────────────────────────────────────
    """))

    fig.savefig("sun_bessel_fit_demo.png", dpi=150)
    print("Saved → sun_bessel_fit_demo.png")
    plt.show()
