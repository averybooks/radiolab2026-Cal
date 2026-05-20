"""
baseline_heatmap.py
Interactive chi-squared heatmap over (b_ew, b_ns) baseline parameter space.

Shows the S / chi-squared surface from brute_force_fit() or nonlinear_fit()
with 1-sigma and 2-sigma confidence contours and a slider for the correlation rho.

Usage (standalone with synthetic data):
    python baseline_heatmap.py

Usage (with your real fit results):
    from baseline_heatmap import plot_baseline_heatmap
    plot_baseline_heatmap(
        b_ew      = nl_result['b_ew'],
        b_ns      = nl_result['b_ns'],
        sigma_ew  = nl_result['sigma_b_ew'],
        sigma_ns  = nl_result['sigma_b_ns'],
        rho       = nl_result['rho'],        # from cov matrix
        S_grid    = bf_result['S_grid'],     # optional: real brute-force surface
        bew_grid  = bf_result['Q_ew_grid'],  # optional
        bns_grid  = bf_result['Q_ns_grid'],  # optional
        lam       = p.lam,
        delta     = delta,
        lat       = p.lat,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec


# ── Chi-squared surface generation ───────────────────────────────────────────

def make_chi2_surface(b_ew, b_ns, sigma_ew, sigma_ns, rho,
                      n_span=4.5, nx=200, ny=200):
    """
    Generate a Gaussian chi-squared surface over (b_ew, b_ns).

    For use when no real brute-force S_grid is available (or to overlay
    on top of one). The 2-parameter chi-squared level for n-sigma is:
        Delta_chi2 = 2.30  (1 sigma, 68.3%)
        Delta_chi2 = 6.18  (2 sigma, 95.4%)

    Parameters
    ----------
    b_ew, b_ns    : float  Best-fit baseline components, metres
    sigma_ew      : float  1-sigma uncertainty on b_ew, metres
    sigma_ns      : float  1-sigma uncertainty on b_ns, metres
    rho           : float  Correlation coefficient [-1, 1]
    n_span        : float  Number of sigma to span in each direction
    nx, ny        : int    Grid resolution

    Returns
    -------
    BEW, BNS : (ny, nx) ndarrays  Meshgrid of parameter values
    chi2     : (ny, nx) ndarray   Chi-squared surface
    """
    bew_arr = np.linspace(b_ew - n_span * sigma_ew, b_ew + n_span * sigma_ew, nx)
    bns_arr = np.linspace(b_ns - n_span * sigma_ns, b_ns + n_span * sigma_ns, ny)
    BEW, BNS = np.meshgrid(bew_arr, bns_arr)

    u = (BEW - b_ew) / sigma_ew
    v = (BNS - b_ns) / sigma_ns
    rho2 = rho ** 2
    chi2 = (u**2 - 2 * rho * u * v + v**2) / (2 * (1 - rho2))

    return BEW, BNS, chi2


def chi2_from_S_grid(S_grid, Q_ew_grid, Q_ns_grid, lam, delta, lat):
    """
    Convert a brute-force S_grid (in Q space) to chi-squared in physical
    baseline space (metres).

    chi2 = S - S_min   (relative to minimum)

    Parameters
    ----------
    S_grid   : (n_ew, n_ns) ndarray  Sum-of-squares from brute_force_fit
    Q_ew_grid: (n_ew,) ndarray       Q_ew values
    Q_ns_grid: (n_ns,) ndarray       Q_ns values
    lam      : float  Wavelength, metres
    delta    : float  Source declination, radians
    lat      : float  Observatory latitude, radians

    Returns
    -------
    BEW, BNS : (n_ns, n_ew) meshgrid in metres
    chi2     : (n_ns, n_ew) chi-squared surface
    """
    cos_d = np.cos(delta)
    sin_l = np.sin(lat)

    bew_arr = Q_ew_grid * lam / cos_d
    bns_arr = Q_ns_grid * lam / (sin_l * cos_d) if len(Q_ns_grid) > 1 \
              else np.array([0.0])

    BEW, BNS = np.meshgrid(bew_arr, bns_arr)
    S_2d = S_grid.T if S_grid.shape[0] == len(Q_ew_grid) else S_grid
    chi2 = S_2d - S_2d.min()
    return BEW, BNS, chi2


# ── Main plotting function ────────────────────────────────────────────────────

def plot_baseline_heatmap(b_ew, b_ns, sigma_ew, sigma_ns, rho=0.0,
                          S_grid=None, bew_grid=None, bns_grid=None,
                          lam=None, delta=None, lat=None,
                          nx=200, ny=200, cmap='Blues_r'):
    """
    Interactive baseline chi-squared heatmap with 1σ / 2σ contours
    and a slider for the correlation rho.

    Parameters
    ----------
    b_ew, b_ns   : float  Best-fit baseline components, metres
    sigma_ew     : float  1-sigma uncertainty on b_ew, metres
    sigma_ns     : float  1-sigma uncertainty on b_ns, metres
    rho          : float  Initial correlation coefficient
    S_grid       : optional (n_ew, n_ns) ndarray  Real brute-force surface
    bew_grid     : optional (n_ew,) ndarray        Q_ew grid values
    bns_grid     : optional (n_ns,) ndarray        Q_ns grid values
    lam          : float  Wavelength in metres (required if S_grid provided)
    delta        : float  Declination in radians (required if S_grid provided)
    lat          : float  Observatory latitude in radians (required if S_grid provided)
    nx, ny       : int    Resolution of synthetic chi2 surface
    cmap         : str    Matplotlib colormap name
    """

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor('#0f0f0f')

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, 0.05],
        height_ratios=[1, 0.12],
        hspace=0.35,
        wspace=0.08,
        left=0.12, right=0.88, top=0.90, bottom=0.18,
    )

    ax      = fig.add_subplot(gs[0, 0])
    cax     = fig.add_subplot(gs[0, 1])
    ax_rho  = fig.add_subplot(gs[1, 0])

    for a in [ax, cax, ax_rho]:
        a.set_facecolor('#0f0f0f')

    # ── Colormap ─────────────────────────────────────────────────────────────
    cm = plt.get_cmap(cmap)

    # ── Real S_grid surface (background) ─────────────────────────────────────
    if S_grid is not None and lam is not None and delta is not None and lat is not None:
        BEW_r, BNS_r, chi2_r = chi2_from_S_grid(S_grid, bew_grid, bns_grid,
                                                  lam, delta, lat)
        im_real = ax.pcolormesh(BEW_r, BNS_r, chi2_r,
                                cmap=cmap, shading='auto',
                                vmin=0, vmax=15, alpha=0.6, zorder=1)

    # ── Synthetic Gaussian surface ────────────────────────────────────────────
    BEW, BNS, chi2 = make_chi2_surface(b_ew, b_ns, sigma_ew, sigma_ns, rho,
                                        nx=nx, ny=ny)
    im = ax.pcolormesh(BEW, BNS, chi2,
                       cmap=cmap, shading='auto',
                       vmin=0, vmax=12, zorder=2, alpha=0.85)

    # ── Sigma contours ────────────────────────────────────────────────────────
    # 2-parameter confidence levels: Delta_chi2 = 2.30 (1sig), 6.18 (2sig)
    LEVEL_1SIG = 2.30
    LEVEL_2SIG = 6.18

    cs1 = ax.contour(BEW, BNS, chi2, levels=[LEVEL_1SIG],
                     colors=['white'], linewidths=1.8, linestyles='-', zorder=4)
    cs2 = ax.contour(BEW, BNS, chi2, levels=[LEVEL_2SIG],
                     colors=['white'], linewidths=1.4, linestyles='--', zorder=4)

    # ── Best-fit marker ───────────────────────────────────────────────────────
    best_pt, = ax.plot(b_ew, b_ns, 'o',
                       color='#ff4444', ms=8, mew=1.5,
                       markeredgecolor='white', zorder=5,
                       label=f'Best fit  ({b_ew:.2f}, {b_ns:.2f}) m')

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Δχ²', color='#cccccc', fontsize=11)
    cb.ax.yaxis.set_tick_params(color='#cccccc')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#cccccc', fontsize=9)
    cb.ax.set_facecolor('#0f0f0f')

    # Horizontal lines on colorbar at sigma levels
    cb.ax.axhline(LEVEL_1SIG, color='white', lw=1.5, ls='-')
    cb.ax.axhline(LEVEL_2SIG, color='white', lw=1.2, ls='--')
    cb.ax.text(1.6, LEVEL_1SIG, '1σ', color='white', fontsize=9,
               va='center', transform=cb.ax.get_yaxis_transform())
    cb.ax.text(1.6, LEVEL_2SIG, '2σ', color='white', fontsize=9,
               va='center', transform=cb.ax.get_yaxis_transform())

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel('b_ew  (m)', color='#cccccc', fontsize=12)
    ax.set_ylabel('b_ns  (m)', color='#cccccc', fontsize=12)
    ax.tick_params(colors='#cccccc', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='white', lw=1.8, ls='-',  label='1σ  (Δχ² = 2.30)'),
        Line2D([0], [0], color='white', lw=1.4, ls='--', label='2σ  (Δχ² = 6.18)'),
        Line2D([0], [0], marker='o', color='#ff4444', ms=7, mew=1.2,
               markeredgecolor='white', lw=0,
               label=f'best fit  ({b_ew:.2f}, {b_ns:.2f}) m'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper right',
                    fontsize=9, framealpha=0.25,
                    facecolor='#1a1a1a', edgecolor='#444444',
                    labelcolor='#cccccc')

    # Title with fit summary
    ax.set_title(
        f'Baseline χ² surface\n'
        f'b_ew = {b_ew:.3f} ± {sigma_ew:.3f} m      '
        f'b_ns = {b_ns:.3f} ± {sigma_ns:.3f} m',
        color='#eeeeee', fontsize=11, pad=10
    )

    # ── rho slider ────────────────────────────────────────────────────────────
    ax_rho.set_facecolor('#1a1a1a')
    sl_rho = Slider(
        ax=ax_rho,
        label='ρ (correlation)',
        valmin=-0.95,
        valmax=0.95,
        valinit=rho,
        valstep=0.01,
        color='#378ADD',
    )
    sl_rho.label.set_color('#cccccc')
    sl_rho.valtext.set_color('#cccccc')
    ax_rho.set_facecolor('#1a1a1a')

    # ── Update function ───────────────────────────────────────────────────────
    def update(val):
        rho_new = sl_rho.val
        _, _, chi2_new = make_chi2_surface(b_ew, b_ns, sigma_ew, sigma_ns,
                                            rho_new, nx=nx, ny=ny)
        im.set_array(chi2_new.ravel())

        # Redraw contours
        for coll in cs1.collections + cs2.collections:
            coll.remove()
        cs1_new = ax.contour(BEW, BNS, chi2_new, levels=[LEVEL_1SIG],
                             colors=['white'], linewidths=1.8, linestyles='-', zorder=4)
        cs2_new = ax.contour(BEW, BNS, chi2_new, levels=[LEVEL_2SIG],
                             colors=['white'], linewidths=1.4, linestyles='--', zorder=4)
        cs1.collections[:] = cs1_new.collections
        cs2.collections[:] = cs2_new.collections

        fig.canvas.draw_idle()

    sl_rho.on_changed(update)

    plt.show()
    return fig, ax, sl_rho


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Plug in your real values from nonlinear_fit() ────────────────────────
    # Example using approximate UCB interferometer values:
    #
    #   from interf_analysis import InterfParams, nonlinear_fit
    #   from interf_analysis.fitting import recover_baseline
    #   import numpy as np
    #
    #   p = InterfParams()
    #   nl = nonlinear_fit(h_s, band_real_filt, p0_dict=bf)
    #   b_ew, b_ns = recover_baseline(nl['Q_ew'], nl['Q_ns'], delta, p.lat, p.lam)
    #   sigma_ew   = nl['sigma_Q_ew'] * p.lam / np.cos(delta)
    #   sigma_ns   = nl['sigma_Q_ns'] * p.lam / (np.sin(p.lat) * np.cos(delta))
    #   rho        = nl['cov'][2,3] / (nl['sigma_Q_ew'] * nl['sigma_Q_ns'])
    #
    #   plot_baseline_heatmap(b_ew, b_ns, sigma_ew, sigma_ns, rho=rho)

    # Synthetic demonstration values:
    plot_baseline_heatmap(
        b_ew     = 20.14,
        b_ns     =  0.31,
        sigma_ew =  0.38,
        sigma_ns =  0.62,
        rho      =  0.15,
    )
