"""
vfit/core/pole_zero.py
──────────────────────
Utilities for pole/zero manipulation and enforcement.
"""

from __future__ import annotations

import warnings
import numpy as np


def enforce_conjugate_pairs(poles: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Ensure poles appear as conjugate pairs for real-valued systems.

    For each pole p = α + jβ with β ≠ 0, ensure conj(p) = α - jβ is present.
    Real poles (|Im(p)| < tol) are kept as-is.

    Parameters
    ----------
    poles : np.ndarray, shape (N,) complex
    tol : float
        Imaginary part threshold below which a pole is considered real.

    Returns
    -------
    np.ndarray
        Poles with conjugate symmetry enforced.
    """
    result = []
    poles = np.asarray(poles, dtype=complex)
    used = np.zeros(len(poles), dtype=bool)

    for i, p in enumerate(poles):
        if used[i]:
            continue
        if abs(p.imag) < tol:
            result.append(p.real + 0j)
            used[i] = True
        else:
            # Search for conjugate
            dists = np.abs(poles - np.conj(p))
            dists[used] = np.inf
            j = np.argmin(dists)
            if dists[j] < tol:
                result.append(p)
                result.append(poles[j])
                used[i] = True
                used[j] = True
            else:
                # No conjugate found — add one
                result.append(p)
                result.append(np.conj(p))
                used[i] = True

    return np.array(result, dtype=complex)


def stabilize_poles(poles: np.ndarray) -> np.ndarray:
    """
    Flip any unstable (RHP) poles to the left half-plane.

    Unstable pole: Re(p) > 0  →  stabilized: -Re(p) + j*Im(p)

    Parameters
    ----------
    poles : np.ndarray, complex

    Returns
    -------
    np.ndarray
        Poles with Re(p) ≤ 0 guaranteed.
    """
    poles = np.array(poles, dtype=complex)
    n_unstable = np.sum(poles.real > 0)
    if n_unstable > 0:
        warnings.warn(
            f"{n_unstable} unstable pole(s) detected and reflected to LHP.",
            RuntimeWarning,
            stacklevel=2,
        )
    poles[poles.real > 0] = -poles[poles.real > 0].real + 1j * poles[poles.real > 0].imag
    return poles


def sort_by_frequency(poles: np.ndarray) -> np.ndarray:
    """Sort poles by ascending imaginary part (natural frequency)."""
    return poles[np.argsort(np.abs(poles.imag))]


def pole_resonant_frequency(pole: complex) -> float:
    """
    Compute the resonant frequency of a complex pole in Hz.

    f₀ = |Im(p)| / (2π)
    """
    return abs(pole.imag) / (2 * np.pi)


def pole_quality_factor(pole: complex) -> float:
    """
    Compute the quality factor Q of a complex pole.

    Q = |p| / (2 · |Re(p)|)

    For a real pole (Im(p) ≈ 0) the Q is infinite.
    """
    if abs(pole.real) < 1e-30:
        return float("inf")
    return abs(pole) / (2.0 * abs(pole.real))
