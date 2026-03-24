"""
vfit/core/rational_function.py
──────────────────────────────
RationalModel — stores a fitted rational transfer function.

    H(s) = e·s  +  d  +  Σᵢ cᵢ / (s − pᵢ)

    e   : proportional (s) coefficient — needed for biproper systems
          e.g. series-RLC impedance Z = R + sL + 1/(sC)  → e = L
    d   : direct (constant) term
    pᵢ  : poles  (complex, LHP for stable systems)
    cᵢ  : residues (complex; conjugate pairs for real-valued systems)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RationalModel:
    """
    Rational transfer function in partial-fraction + direct-term form.

    Attributes
    ----------
    poles : ndarray complex (n,)
        Poles of the transfer function.
    residues : ndarray complex (n,)
        Residues corresponding to each pole.
    d : float
        Direct (constant) term.
    e : float
        Proportional term (coefficient of s = jω).
        Zero for strictly-proper systems; nonzero for biproper systems
        such as series-RLC impedance Z = R + sL + 1/(sC).
    freq_fit : ndarray (N,), optional
        Frequency array in Hz used during fitting.
    H_meas : ndarray complex (N,), optional
        Original measured frequency response.
    rms_error_history : list of float
        Absolute RMS error at each VF iteration.
    """

    poles:             np.ndarray
    residues:          np.ndarray
    d:                 float = 0.0
    e:                 float = 0.0
    freq_fit:          Optional[np.ndarray] = None
    H_meas:            Optional[np.ndarray] = None
    rms_error_history: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Evaluation                                                            #
    # ------------------------------------------------------------------ #

    def evaluate(self, freq: np.ndarray) -> np.ndarray:
        """
        Evaluate H(jω) at given frequencies.

        Parameters
        ----------
        freq : ndarray
            Frequency in Hz.

        Returns
        -------
        ndarray complex  —  H(j·2π·f)
        """
        s   = 1j * 2.0 * np.pi * np.asarray(freq, dtype=float)
        pfe = np.zeros(s.shape, dtype=complex)
        for ci, pi in zip(self.residues, self.poles):
            pfe += ci / (s - pi)
        return self.e * s + self.d + pfe

    # ------------------------------------------------------------------ #
    # Poles / zeros                                                         #
    # ------------------------------------------------------------------ #

    @property
    def zeros(self) -> np.ndarray:
        """
        Compute zeros numerically from poles, residues, d, and e.

        H(s) = e·s + d + Σᵢ cᵢ/(s−pᵢ)
             = [e·s·∏(s−pᵢ) + d·∏(s−pᵢ) + Σᵢ cᵢ·∏_{j≠i}(s−pⱼ)] / ∏(s−pᵢ)

        The numerator polynomial is built and its roots returned.
        """
        poles    = self.poles
        residues = self.residues
        n        = len(poles)

        def _poly(roots: np.ndarray) -> np.ndarray:
            """np.poly wrapper that handles empty arrays → returns [1.0]."""
            arr = np.asarray(roots, dtype=complex)
            return np.array([1.0], dtype=complex) if arr.size == 0 else np.poly(arr).astype(complex)

        # Denominator poly: length n+1, descending powers [s^n … s^0]
        den = _poly(poles)   # (n+1,)

        # Numerator: up to degree n+1 (e·s raises degree by 1)
        # Layout: index 0 = s^(n+1), index 1 = s^n, …, index n+1 = s^0
        numer = np.zeros(n + 2, dtype=complex)

        # e·s · den  → coefficients of e·[s^(n+1) … s^1 0]
        if self.e != 0.0:
            numer[0:n + 1] += self.e * den          # s^(n+1) … s^1

        # d · den
        numer[1:n + 2] += self.d * den              # s^n … s^0

        # Σᵢ cᵢ · ∏_{j≠i}(s−pⱼ) : degree n-1 poly → fills positions [2 … n+1]
        for i in range(n):
            partial = _poly(np.delete(poles, i))    # length n  (degree n-1)
            numer[2:n + 2] += residues[i] * partial  # s^(n-1) … s^0

        # Strip negligible leading coefficients before root-finding
        scale = np.abs(numer).max() + 1e-300
        while len(numer) > 1 and abs(numer[0]) < 1e-14 * scale:
            numer = numer[1:]

        return np.roots(numer)

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def n_poles(self) -> int:
        """Number of poles."""
        return len(self.poles)

    @property
    def rms_error(self) -> float:
        """Final absolute RMS fitting error (last iteration)."""
        return self.rms_error_history[-1] if self.rms_error_history else float("nan")

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            f"RationalModel — {self.n_poles} poles",
            f"  RMS error        = {self.rms_error:.4e}",
            f"  d (direct term)  = {self.d:.4e}",
            f"  e (s-coeff)      = {self.e:.4e}",
            "  Poles (rad/s):",
        ]
        for i, p in enumerate(self.poles):
            lines.append(f"    p[{i}] = {p.real:+.4e}  {p.imag:+.4e}j")
        return "\n".join(lines)

    def __repr__(self) -> str:
        e_str = f", e={self.e:.3e}" if self.e != 0.0 else ""
        return (
            f"RationalModel(n_poles={self.n_poles}, "
            f"rms={self.rms_error:.3e}{e_str})"
        )
