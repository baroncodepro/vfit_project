"""
vfit/core/vector_fitting.py
───────────────────────────
Main Vector Fitting (VF) implementation.

Algorithm reference:
    B. Gustavsen and A. Semlyen,
    "Rational approximation of frequency domain responses by vector fitting,"
    IEEE Trans. Power Delivery, vol. 14, no. 3, pp. 1052-1061, July 1999.

    B. Gustavsen,
    "Improving the pole relocating properties of vector fitting,"
    IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592, July 2006.

Model form
----------
    H(s) = e·s  +  d  +  Σᵢ cᵢ / (s − pᵢ)

    e   — proportional term (default 0; set include_e_term=True for RLC impedance)
    d   — direct (constant) term
    pᵢ  — poles   (complex, LHP after stabilisation)
    cᵢ  — residues (complex)

Algorithm steps per iteration
------------------------------
1. Solve the weighted real-split LS system with d_sigma=1 fixed,
   using column scaling for numerical stability.
2. Relocate poles: eigenvalues of  diag(poles) − ones · c_sig^T.
3. Reflect any RHP poles to LHP (optional, default True).
4. Residue fit: solve LS for (c_H, d, e) with poles fixed.
5. Repeat until |ΔRMS| < tol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rational_function import RationalModel
from .pole_zero import stabilize_poles


@dataclass
class VFOptions:
    """Configuration options for the Vector Fitting algorithm."""
    n_poles:           int   = 10
    n_iter_max:        int   = 20
    tol:               float = 1e-10
    init_poles:        str   = "log"
    weight:            str   = "inverse"
    enforce_stability: bool  = True
    include_e_term:    bool  = True
    verbose:           bool  = False


class VectorFitter:
    """
    Fit a rational transfer function to complex frequency-domain data.

        H(s) = e·s + d + Σᵢ cᵢ / (s − pᵢ)

    Parameters
    ----------
    n_poles : int
        Number of poles (default 10).
    n_iter_max : int
        Maximum iterations (default 20).
    tol : float
        Convergence threshold on |ΔRMS| (default 1e-10).
    init_poles : str
        'log' or 'linear' spacing for starting poles.
    weight : str
        'inverse' (recommended, 1/|H|) or 'uniform'.
    enforce_stability : bool
        Reflect RHP poles to LHP (default True).
    include_e_term : bool
        Fit an e·s direct term (default True, needed for RLC impedance).
    verbose : bool
        Print RMS per iteration (default False).

    Examples
    --------
    >>> import numpy as np
    >>> from vfit import VectorFitter
    >>> freq = np.logspace(7, 11, 300)
    >>> s = 1j * 2 * np.pi * freq
    >>> Z = 50 + s * 10e-9 + 1 / (s * 1e-12)   # series RLC
    >>> model = VectorFitter(n_poles=6).fit(freq, Z)
    >>> print(model)
    RationalModel(n_poles=6, rms=..., e=1.000e-08)
    """

    def __init__(self, **options):
        self.opts = VFOptions(**options)

    # ------------------------------------------------------------------ #
    # Public                                                                #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        freq: np.ndarray,
        H: np.ndarray,
        poles_init: Optional[np.ndarray] = None,
    ) -> RationalModel:
        """
        Fit a rational model to frequency-domain data.

        Parameters
        ----------
        freq : ndarray (N,)   Frequency in Hz (must be > 0).
        H    : ndarray (N,) complex   Response H(j·2π·f).
        poles_init : ndarray complex, optional   Override starting poles.

        Returns
        -------
        RationalModel with poles, residues, d, e, rms_error_history.
        """
        freq = np.asarray(freq, dtype=float)
        H    = np.asarray(H,    dtype=complex)
        self._validate(freq, H)

        s = 1j * 2.0 * np.pi * freq

        poles = (
            np.asarray(poles_init, dtype=complex)
            if poles_init is not None
            else self._init_poles(s)
        )

        c_H, d_H, e_H = np.zeros(len(poles), dtype=complex), 0.0, 0.0
        rms_history: list[float] = []

        for iteration in range(self.opts.n_iter_max):

            # Step 1 — sigma-H LS
            c_H, d_H, e_H, c_sig = self._vf_ls_step(s, H, poles)

            # Step 2 — pole relocation
            poles = self._relocate(poles, c_sig)

            # Step 3 — stabilise
            if self.opts.enforce_stability:
                poles = stabilize_poles(poles)

            # Step 4 — residue fit
            c_H, d_H, e_H = self._residue_fit(s, H, poles)

            # Convergence
            H_fit = self._eval(s, poles, c_H, d_H, e_H)
            rms   = float(np.sqrt(np.mean(np.abs(H - H_fit) ** 2)))
            rms_history.append(rms)

            if self.opts.verbose:
                print(f"  Iter {iteration + 1:2d}: RMS = {rms:.4e}")

            if len(rms_history) > 1:
                delta_rel = abs(rms_history[-2] - rms_history[-1]) / (rms_history[-2] + 1e-300)
                if delta_rel < self.opts.tol or rms_history[-1] < 1e-13:
                    break
        else:
            warnings.warn(
                f"VectorFitter did not converge in {self.opts.n_iter_max} iterations. "
                f"Final RMS = {rms_history[-1]:.4e}",
                RuntimeWarning, stacklevel=2,
            )

        return RationalModel(
            poles=poles, residues=c_H, d=d_H, e=e_H,
            freq_fit=freq, H_meas=H, rms_error_history=rms_history,
        )

    # ------------------------------------------------------------------ #
    # Private — algorithm                                                   #
    # ------------------------------------------------------------------ #

    def _init_poles(self, s: np.ndarray) -> np.ndarray:
        """Lightly-damped conjugate pairs across the frequency band."""
        n       = self.opts.n_poles
        n_pairs = n // 2
        omega   = np.abs(s.imag)
        w_min, w_max = omega.min(), omega.max()

        beta = (
            np.logspace(np.log10(w_min), np.log10(w_max), max(n_pairs, 1))
            if self.opts.init_poles == "log"
            else np.linspace(w_min, w_max, max(n_pairs, 1))
        )
        alpha = beta / 100.0

        # Build from lists to avoid the pre-allocation/append length mismatch
        # that occurs with odd n_poles (np.append would yield n+1 poles).
        parts: list[complex] = []
        for a, b in zip(alpha[:n_pairs], beta[:n_pairs]):
            parts.append(-a + 1j * b)   # upper half-plane
            parts.append(-a - 1j * b)   # conjugate (lower)

        if n % 2 == 1:
            # One real pole at the geometric-mean frequency
            parts.append(-(np.sqrt(w_min * w_max) / 100.0) + 0j)

        return np.array(parts, dtype=complex)

    def _vf_ls_step(self, s, H, poles):
        """
        Solve the sigma-H LS system (d_sigma = 1 fixed, column-scaled).

        Unknown vector x (all real):
            [ Re(c_H_0), Im(c_H_0), …,   <- 2n values
              d_H,                        <- 1 value
              e_H,                        <- 1 value
              Re(c_sig_0), Im(c_sig_0), … <- 2n values ]
        Total: 4n + 2.
        """
        N, n  = len(s), len(poles)
        w     = self._weights(H)
        wr    = np.concatenate([w, w])

        Phi   = 1.0 / (s[:, None] - poles[None, :])   # (N,n) complex
        mHPhi = -H[:, None] * Phi                       # (N,n) complex

        off   = 2 * n + 2       # start of sigma block
        n_unk = 4 * n + 2

        A = np.zeros((2 * N, n_unk))

        # c_H columns (complex residues, real-split)
        for i in range(n):
            A[:N, 2*i]     =  Phi[:, i].real
            A[:N, 2*i + 1] = -Phi[:, i].imag
            A[N:, 2*i]     =  Phi[:, i].imag
            A[N:, 2*i + 1] =  Phi[:, i].real

        # d_H column (real constant — only Re rows)
        A[:N, 2*n] = 1.0

        # e_H column (e·jω — only Im rows: Im(e·jω) = e·ω)
        if self.opts.include_e_term:
            A[N:, 2*n + 1] = s.imag   # ω = Im(jω)

        # c_sig columns (sigma residues)
        for i in range(n):
            A[:N, off + 2*i]     =  mHPhi[:, i].real
            A[:N, off + 2*i + 1] = -mHPhi[:, i].imag
            A[N:, off + 2*i]     =  mHPhi[:, i].imag
            A[N:, off + 2*i + 1] =  mHPhi[:, i].real

        b = np.concatenate([H.real, H.imag])
        A = A * wr[:, None]
        b = b * wr

        # Column scaling — critical for conditioning
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms < 1e-30] = 1.0
        x_sc, *_ = np.linalg.lstsq(A / col_norms, b, rcond=None)
        x = x_sc / col_norms

        c_H   = x[0:2*n:2]    + 1j * x[1:2*n:2]
        d_H   = float(x[2*n])
        e_H   = float(x[2*n + 1]) if self.opts.include_e_term else 0.0
        c_sig = x[off::2][:n] + 1j * x[off + 1::2][:n]

        return c_H, d_H, e_H, c_sig

    def _relocate(self, poles: np.ndarray, c_sig: np.ndarray) -> np.ndarray:
        """New poles = eigenvalues of  diag(a) − ones·c_sigᵀ."""
        n = len(poles)
        A = np.diag(poles.astype(complex))
        b = np.ones((n, 1), dtype=complex)
        c = c_sig[:n].reshape(1, n)
        return np.linalg.eigvals(A - b @ c)

    def _residue_fit(self, s, H, poles):
        """Solve for c_H, d_H, e_H with poles fixed (column-scaled LS)."""
        N, n  = len(s), len(poles)
        w     = self._weights(H)
        wr    = np.concatenate([w, w])

        Phi   = 1.0 / (s[:, None] - poles[None, :])
        n_unk = 2 * n + 2

        A = np.zeros((2 * N, n_unk))
        for i in range(n):
            A[:N, 2*i]     =  Phi[:, i].real
            A[:N, 2*i + 1] = -Phi[:, i].imag
            A[N:, 2*i]     =  Phi[:, i].imag
            A[N:, 2*i + 1] =  Phi[:, i].real
        A[:N, 2*n] = 1.0
        if self.opts.include_e_term:
            A[N:, 2*n + 1] = s.imag

        b = np.concatenate([H.real, H.imag])
        A = A * wr[:, None]
        b = b * wr

        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms < 1e-30] = 1.0
        x_sc, *_ = np.linalg.lstsq(A / col_norms, b, rcond=None)
        x = x_sc / col_norms

        c_H = x[0:2*n:2] + 1j * x[1:2*n:2]
        d_H = float(x[2*n])
        e_H = float(x[2*n + 1]) if self.opts.include_e_term else 0.0
        return c_H, d_H, e_H

    # ------------------------------------------------------------------ #
    # Private — utilities                                                   #
    # ------------------------------------------------------------------ #

    def _weights(self, H: np.ndarray) -> np.ndarray:
        w = 1.0 / (np.abs(H) + 1e-30) if self.opts.weight == "inverse" else np.ones(len(H))
        return w / w.max()

    @staticmethod
    def _eval(s, poles, residues, d, e=0.0):
        """H(s) = e·s + d + Σᵢ cᵢ/(s−pᵢ)."""
        pfe = np.sum(residues[None, :] / (s[:, None] - poles[None, :]), axis=1)
        return e * s + d + pfe

    @staticmethod
    def _validate(freq, H):
        if freq.shape != H.shape:
            raise ValueError(f"freq and H must have same shape: {freq.shape} vs {H.shape}.")
        if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(H)):
            raise ValueError("freq and H must not contain NaN or Inf.")
        if np.any(freq <= 0):
            raise ValueError("All frequency values must be strictly positive (Hz).")
