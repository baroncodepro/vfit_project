"""
vfit/visualization/__init__.py
──────────────────────────────
Publication-quality plots for Vector Fitting results.

All functions accept an optional `ax` (or `axes`) argument.
If None, a new figure is created.  All return (fig, ax) tuples.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..core.rational_function import RationalModel


# ── Colour palette (consistent across all plots) ─────────────────────────────
_C_MEAS  = "#2c7bb6"   # measured data  — blue
_C_FIT   = "#d7191c"   # fitted model   — red
_C_POLE  = "#e66101"   # poles          — orange
_C_ZERO  = "#1a9641"   # zeros          — green
_C_GRID  = "#cccccc"


# ─────────────────────────────────────────────────────────────────────────────
# Bode Plot
# ─────────────────────────────────────────────────────────────────────────────

def bode_plot(
    freq: np.ndarray,
    H_meas: np.ndarray,
    model: Optional[RationalModel] = None,
    *,
    axes: Optional[Tuple[Axes, Axes]] = None,
    freq_unit: str = "Hz",
    mag_unit: str = "dB",
    title: str = "Bode Plot",
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Bode magnitude and phase plot.

    Parameters
    ----------
    freq : ndarray
        Frequency array in Hz.
    H_meas : ndarray complex
        Measured (or reference) complex response.
    model : RationalModel, optional
        Fitted rational model to overlay.
    axes : (ax_mag, ax_phase), optional
        Pre-existing axes pair.  Created if None.
    freq_unit : str
        Label unit for frequency axis (default 'Hz').
    mag_unit : str
        'dB' or 'linear'.
    title : str
        Figure title.

    Returns
    -------
    fig, (ax_mag, ax_phase)
    """
    if axes is None:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True,
            gridspec_kw={"hspace": 0.08}
        )
    else:
        ax_mag, ax_phase = axes
        fig = ax_mag.figure

    # Scale frequency for display
    f_scale, f_label = _freq_scale(freq, freq_unit)
    f_disp = freq * f_scale

    # Magnitude
    if mag_unit == "dB":
        mag_meas = 20 * np.log10(np.abs(H_meas) + 1e-300)
        y_label_mag = "Magnitude (dB)"
    else:
        mag_meas = np.abs(H_meas)
        y_label_mag = "Magnitude"

    phase_meas = np.degrees(np.unwrap(np.angle(H_meas)))

    ax_mag.semilogx(f_disp, mag_meas, color=_C_MEAS,
                    lw=1.5, label="Measured", alpha=0.85)
    ax_phase.semilogx(f_disp, phase_meas, color=_C_MEAS,
                      lw=1.5, label="Measured", alpha=0.85)

    if model is not None:
        H_fit = model.evaluate(freq)
        if mag_unit == "dB":
            mag_fit = 20 * np.log10(np.abs(H_fit) + 1e-300)
        else:
            mag_fit = np.abs(H_fit)
        phase_fit = np.degrees(np.unwrap(np.angle(H_fit)))

        ax_mag.semilogx(f_disp, mag_fit, color=_C_FIT,
                        lw=2.0, ls="--", label=f"VF fit  (RMS={model.rms_error:.2e})")
        ax_phase.semilogx(f_disp, phase_fit, color=_C_FIT,
                          lw=2.0, ls="--", label="VF fit")

    _style_ax(ax_mag, ylabel=y_label_mag, grid=True)
    _style_ax(ax_phase, xlabel=f_label, ylabel="Phase (°)", grid=True)

    ax_mag.legend(fontsize=9)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, (ax_mag, ax_phase)


# ─────────────────────────────────────────────────────────────────────────────
# Nyquist Plot
# ─────────────────────────────────────────────────────────────────────────────

def nyquist_plot(
    freq: np.ndarray,
    H_meas: np.ndarray,
    model: Optional[RationalModel] = None,
    *,
    ax: Optional[Axes] = None,
    title: str = "Nyquist Plot",
) -> Tuple[Figure, Axes]:
    """
    Nyquist diagram — Im(H) vs Re(H).

    Parameters
    ----------
    freq, H_meas, model
        Same as bode_plot.
    ax : Axes, optional
    title : str

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    ax.plot(H_meas.real, H_meas.imag,
            color=_C_MEAS, lw=1.5, label="Measured", alpha=0.85)
    ax.scatter(H_meas.real[0], H_meas.imag[0],
               color=_C_MEAS, zorder=5, s=50, label=f"f={freq[0]:.2g} Hz")
    ax.scatter(H_meas.real[-1], H_meas.imag[-1],
               color=_C_MEAS, marker="s", zorder=5, s=50,
               label=f"f={freq[-1]:.2g} Hz")

    if model is not None:
        H_fit = model.evaluate(freq)
        ax.plot(H_fit.real, H_fit.imag,
                color=_C_FIT, lw=2.0, ls="--", label="VF fit")

    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_aspect("equal")
    _style_ax(ax, xlabel="Re(H)", ylabel="Im(H)", grid=True)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Pole-Zero Map
# ─────────────────────────────────────────────────────────────────────────────

def pole_zero_map(
    model: RationalModel,
    *,
    ax: Optional[Axes] = None,
    freq_unit: str = "rad/s",
    title: str = "Pole-Zero Map (S-Plane)",
    show_unit_circle: bool = False,
) -> Tuple[Figure, Axes]:
    """
    S-plane pole-zero plot.

    Parameters
    ----------
    model : RationalModel
    ax : Axes, optional
    freq_unit : 'rad/s' or 'Hz'
    title : str
    show_unit_circle : bool
        Draw the unit circle (relevant for z-domain; optional here).

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    poles = model.poles
    try:
        zeros = model.zeros
    except Exception:
        zeros = np.array([])

    scale = 1.0 if freq_unit == "rad/s" else 1 / (2 * np.pi)

    # Poles
    ax.scatter(poles.real * scale, poles.imag * scale,
               marker="x", s=120, color=_C_POLE, linewidths=2,
               zorder=5, label=f"Poles ({len(poles)})")

    # Zeros
    if len(zeros) > 0:
        ax.scatter(zeros.real * scale, zeros.imag * scale,
                   marker="o", s=80, facecolors="none",
                   edgecolors=_C_ZERO, linewidths=2,
                   zorder=5, label=f"Zeros ({len(zeros)})")

    # Stability boundary
    ymin = min(poles.imag.min(), -1) * scale * 1.2
    ymax = max(poles.imag.max(), +1) * scale * 1.2
    ax.axvline(0, color="k", lw=1.2, ls="--", alpha=0.5, label="jω axis")
    ax.fill_betweenx([ymin, ymax], 0,
                     ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1,
                     alpha=0.05, color="red", label="RHP (unstable)")

    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(np.cos(theta), np.sin(theta),
                color="gray", lw=0.8, ls=":", label="Unit circle")

    ax.set_xlim(left=min(poles.real.min() * scale * 1.3, -0.1 * scale))
    ax.set_ylim(ymin, ymax)

    unit_str = " (Hz)" if freq_unit == "Hz" else " (rad/s)"
    _style_ax(ax, xlabel="σ" + unit_str, ylabel="jω" + unit_str, grid=True)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Convergence Plot
# ─────────────────────────────────────────────────────────────────────────────

def convergence_plot(
    model: RationalModel,
    *,
    ax: Optional[Axes] = None,
    title: str = "VF Convergence",
) -> Tuple[Figure, Axes]:
    """
    Plot RMS fitting error vs iteration number.

    Parameters
    ----------
    model : RationalModel
    ax : Axes, optional
    title : str

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    hist = model.rms_error_history
    iters = np.arange(1, len(hist) + 1)

    ax.semilogy(iters, hist, "o-", color=_C_MEAS, lw=2, ms=6)
    ax.axhline(hist[-1], color=_C_FIT, ls="--", lw=1,
               label=f"Final RMS = {hist[-1]:.3e}")

    ax.set_xticks(iters)
    _style_ax(ax, xlabel="Iteration", ylabel="RMS Error", grid=True)
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(
    ax: Axes,
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
) -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, which="both", color=_C_GRID, lw=0.6, alpha=0.7)
        ax.grid(True, which="minor", color=_C_GRID, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _freq_scale(freq: np.ndarray, unit: str) -> Tuple[float, str]:
    """Return (scale_factor, axis_label) for frequency display."""
    fmax = freq.max()
    if unit != "Hz":
        return 1.0, f"Frequency ({unit})"
    if fmax >= 1e9:
        return 1e-9, "Frequency (GHz)"
    if fmax >= 1e6:
        return 1e-6, "Frequency (MHz)"
    if fmax >= 1e3:
        return 1e-3, "Frequency (kHz)"
    return 1.0, "Frequency (Hz)"
