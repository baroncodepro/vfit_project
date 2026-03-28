"""
vfit/rlc/rlc_synthesis.py
─────────────────────────
Convert a RationalModel (impedance Z(s)) to Foster-form RLC elements.

Foster-I synthesis (series arms in parallel):
    For each conjugate pair  p = -α ± jβ  with residue  c = a + jb:

        Parallel RLC branch:
            R  = 1 / (2a)            [Ω]
            L  = 1 / (2a)            [H]   — depends on port convention
            C  = 2a / |p|²           [F]

    For each real pole  p = -α  (negative real, α > 0):
        Series RL:
            R  = Re(c) / α           [Ω]   (or R = c if d has units)
            L  = Re(c) / α²          [H]   — reworked per physical units

    Direct term  d  (real scalar):
        Pure resistance  R = d       [Ω]

Reference:
    Gustavsen, "Computer Code for Rational Approximation …", SINTEF, 2009.
    Balabanian, "Network Synthesis", Prentice-Hall, 1958.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..core.rational_function import RationalModel


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RLCBranch:
    """
    One branch (term) of a Foster-I parallel RLC ladder.

    Topology   R [Ω]   L [H]   C [F]   Circuit                  Notes
    ---------  ------  ------  ------  -----------------------  ----------------------------------
    R_only     R       –       –       R                        d direct term
    L_only     –       L       –       sL                       e proportional term
    C_only     –       –       C       1/(sC)                   near-DC pole
    RC         R       –       C       R ∥ C = R/(1+sRC)        real LHP pole: R=c/α, C=1/c
    RLC        R       L       C       R ∥ L ∥ C                complex conjugate pair
    """

    branch_type: Literal["RL", "RC", "RLC", "R_only", "C_only", "L_only"]
    R: float | None = None   # Ohm
    L: float | None = None   # Henry
    C: float | None = None   # Farad
    pole: complex | None = None   # originating pole (for reference)

    def __repr__(self) -> str:
        parts = [f"type={self.branch_type}"]
        if self.R is not None:
            parts.append(f"R={self.R:.4g} Ω")
        if self.L is not None:
            parts.append(f"L={self.L:.4g} H")
        if self.C is not None:
            parts.append(f"C={self.C:.4g} F")
        return "RLCBranch(" + ", ".join(parts) + ")"


@dataclass
class FosterNetwork:
    """
    Foster-I RLC network synthesised from a rational impedance model.

    The network is a set of parallel branches, all in series,
    representing Z(s) = d + Σᵢ branchᵢ(s).
    """

    branches: list[RLCBranch] = field(default_factory=list)
    model: RationalModel | None = None    # originating model

    def __repr__(self) -> str:
        lines = [f"FosterNetwork — {len(self.branches)} branches:"]
        for i, b in enumerate(self.branches):
            lines.append(f"  [{i}] {b}")
        return "\n".join(lines)

    def impedance(self, freq: np.ndarray) -> np.ndarray:
        """
        Re-simulate the network impedance at given frequencies.

        Parameters
        ----------
        freq : np.ndarray
            Frequency in Hz.

        Returns
        -------
        np.ndarray complex
            Z(jω) from synthesised elements.
        """
        omega = 2 * np.pi * np.asarray(freq, dtype=float)
        s = 1j * omega
        Z = np.zeros_like(s, dtype=complex)
        for branch in self.branches:
            Z += _branch_impedance(branch, s)
        return Z


# ── Core synthesis ────────────────────────────────────────────────────────────

def _clamp_positive(val: float, name: str) -> float:
    """Return max(val, 0), warning if val is significantly negative."""
    if val < -1e-10:
        warnings.warn(
            f"Non-physical {name}={val:.3e} clamped to 0 (numerical noise).",
            UserWarning, stacklevel=3,
        )
    return max(val, 0.0)


def foster_synthesis(model: RationalModel) -> FosterNetwork:
    """
    Synthesise a Foster-I parallel RLC network from a rational impedance model.

    The model is assumed to represent Z(s) (impedance) in partial fraction form:

        Z(s) = d + Σᵢ cᵢ / (s - pᵢ)

    Parameters
    ----------
    model : RationalModel
        Fitted rational model.  Poles must be in LHP (stable).

    Returns
    -------
    FosterNetwork
        Synthesised RLC network with element values.

    Raises
    ------
    ValueError
        If any pole has positive real part (unstable).
    """
    if len(model.poles) > 0 and np.any(model.poles.real > 0):
        raise ValueError(
            "Foster synthesis requires stable (LHP) poles. "
            "Call stabilize_poles() before synthesis."
        )

    poles    = model.poles
    residues = model.residues
    branches: list[RLCBranch] = []

    # ── h term → series inductance  L = h  ───────────────────────────────
    e_val = getattr(model, "e", 0.0)
    if abs(e_val) > 1e-30:
        branches.append(RLCBranch(branch_type="L_only", L=float(abs(e_val))))

    # ── Direct term d → series resistance ────────────────────────────────
    if abs(model.d) > 1e-20:
        branches.append(RLCBranch(branch_type="R_only", R=_clamp_positive(float(model.d), "d")))

    # ── Process poles ─────────────────────────────────────────────────────
    used = np.zeros(len(poles), dtype=bool)
    _near_dc = 1.0   # rad/s — poles closer to origin model 1/(sC) capacitors

    for i, (p, c) in enumerate(zip(poles, residues)):
        if used[i]:
            continue

        # Near-DC pole: |p| ≈ 0  →  Z_branch ≈ c/s  →  C = 1/Re(c)
        if abs(p) < _near_dc and abs(c.real) > 1e-30:
            C_val = 1.0 / abs(c.real)
            branches.append(RLCBranch(branch_type="C_only", C=C_val, pole=p))
            used[i] = True
            j_c = _find_conjugate(poles, p, used, i)
            if j_c is not None:
                used[j_c] = True
            continue

        is_real_pole = abs(p.imag) < 1e-6 * (abs(p) + 1e-30)

        if is_real_pole:
            alpha  = -p.real    # > 0 for LHP pole
            c_real = abs(c.real)   # always use magnitude (negative already warned about)

            if c_real > 1e-30:
                # Z_branch = c/(s+alpha)
                # Y = (s+alpha)/c = s*(1/c) + alpha/c
                # => parallel RC: R = c/alpha,  C = 1/c
                R_val = _clamp_positive(c_real / alpha, "R")
                C_val = _clamp_positive(1.0 / c_real, "C")
                branches.append(RLCBranch(branch_type="RC", R=R_val, C=C_val, pole=p))
            used[i] = True

        else:
            # ── Complex conjugate pair ────────────────────────────────────
            j_partner = _find_conjugate(poles, p, used, i)

            if p.imag < 0:
                if j_partner is not None:
                    used[i] = True
                continue

            alpha      = -p.real
            beta       = abs(p.imag)
            omega_0_sq = alpha**2 + beta**2

            a = c.real
            if a <= 0:
                warnings.warn(
                    f"Negative residue real part for pole {p:.3e}: "
                    f"Re(c)={a:.3e}. Using |Re(c)|.",
                    UserWarning, stacklevel=2,
                )
                a = abs(a)

            C_val = _clamp_positive(1.0 / (2.0 * a), "C")
            L_val = _clamp_positive(1.0 / (C_val * omega_0_sq) if C_val > 1e-30 else 0.0, "L")
            R_val = _clamp_positive(1.0 / (2.0 * alpha * C_val) if C_val > 1e-30 else 0.0, "R")

            branches.append(RLCBranch(branch_type="RLC", R=R_val, L=L_val, C=C_val, pole=p))
            used[i] = True
            if j_partner is not None:
                used[j_partner] = True

    return FosterNetwork(branches=branches, model=model)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_conjugate(
    poles: np.ndarray,
    p: complex,
    used: np.ndarray,
    self_idx: int,
) -> int | None:
    """Return index of the conjugate of pole p, or None."""
    target = np.conj(p)
    for j, q in enumerate(poles):
        if j == self_idx or used[j]:
            continue
        if abs(q - target) < 1e-6 * (abs(p) + 1e-30):
            return j
    return None


def _branch_impedance(branch: RLCBranch, s: np.ndarray) -> np.ndarray:
    """Compute Z(s) for a single RLC branch.

    Topology reference (Foster-I impedance synthesis):

    R_only   Z = R
    L_only   Z = sL                         (e-term: high-freq inductance)
    C_only   Z = 1/(sC)                     (near-DC pole)
    RC       Z = 1/(1/R + sC) = R/(1+sRC)   (real LHP pole: p=-alpha, R=c/alpha, C=1/c)
    RLC      Z = 1/(1/R + 1/(sL) + sC)      (complex conjugate pole pair)
    """
    bt = branch.branch_type

    if bt == "R_only":
        return np.full_like(s, branch.R, dtype=complex)

    if bt == "L_only":
        return s * branch.L

    if bt == "C_only":
        return 1.0 / (s * branch.C)

    if bt == "RC":
        # Parallel RC: Z = R / (1 + s*R*C)
        return branch.R / (1.0 + s * branch.R * branch.C)

    if bt == "RLC":
        # Parallel RLC: Z = 1 / (1/R + 1/(sL) + sC)
        admittance = (1.0 / branch.R
                      + 1.0 / (s * branch.L)
                      + s * branch.C)
        return 1.0 / admittance

    # Legacy RL branch (should no longer appear in new synthesis)
    if bt == "RL":
        # Parallel RL: Z = sL*R/(sL+R)
        sL = s * branch.L
        return sL * branch.R / (sL + branch.R)

    raise ValueError(f"Unknown branch type: {bt}")
