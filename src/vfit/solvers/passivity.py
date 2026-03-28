"""
vfit/solvers/passivity.py
──────────────────────────
Passivity check and enforcement for scalar rational impedance models.

Background
──────────
A one-port impedance Z(s) is passive if and only if it is a
*positive-real* (PR) function:

  1. Z(s) is analytic in Re(s) > 0  (no RHP poles)
  2. Re[Z(jω)] ≥ 0  for all ω ∈ ℝ  (non-negative real part on jω-axis)

Condition 1 is already guaranteed by `stabilize_poles()` in the VF loop.
This module handles condition 2.

Why VF models can violate passivity
─────────────────────────────────────
Even with all poles in the LHP, the fitted residues can produce
Re[Z(jω)] < 0 at some frequencies — typically near band edges or where
the SNR is low.  A non-passive model will cause SPICE/time-domain
simulators to produce unbounded (diverging) responses.

Algorithm  (scalar PR enforcement)
────────────────────────────────────
Given a rational model H(s) = d + Σᵢ cᵢ/(s−pᵢ):

  check():   Evaluate Re[H(jω)] on a dense grid; find all intervals
             where it drops below zero.

  enforce(): Iteratively correct residues by the minimum-norm perturbation
             that lifts Re[H(jω)] to zero at each violation point:

               δRe(cᵢ) ∝ Re[1/(jω* − pᵢ)]    (sensitivity of Re[H] to Re(cᵢ))

             Scale factor k chosen so the correction exactly repairs the
             violation depth at ω* with a small positive margin.
             Repeat until Re[H(jω)] ≥ 0 everywhere or max_iter reached.

  The perturbation is minimum-norm: only the real parts of residues
  closest to the violation frequency are significantly affected.
  The imaginary parts of residues (which control the reactive/inductive
  character) are left untouched.

Reference
──────────
Triverio, Grivet-Talocia, Nakhla, Canavero, Achar,
"Stability, Causality, and Passivity in Electrical Interconnect Models,"
IEEE Trans. Advanced Packaging, 30(4), 795–804, 2007.

Gustavsen,
"Fast passivity enforcement for pole-residue models by perturbation
of residue matrix eigenvalues,"
IEEE Trans. Power Delivery, 23(4), 2278–2285, 2008.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.rational_function import RationalModel


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PassivityViolation:
    """
    One contiguous frequency interval where Re[Z(jω)] < 0.

    Attributes
    ----------
    f_lo, f_hi : float
        Band edges in Hz.
    f_worst : float
        Frequency of deepest violation (Hz).
    depth : float
        Magnitude of deepest violation: −min(Re[Z]) > 0.
    """
    f_lo    : float
    f_hi    : float
    f_worst : float
    depth   : float

    def __repr__(self) -> str:
        return (f"PassivityViolation("
                f"f={self.f_lo:.3e}–{self.f_hi:.3e} Hz, "
                f"worst={self.f_worst:.3e} Hz, depth={self.depth:.4e} Ω)")


@dataclass
class PassivityReport:
    """
    Full passivity assessment of a rational model.

    Attributes
    ----------
    is_passive : bool
    min_re_Z : float
        Global minimum of Re[Z(jω)] across the check grid.
    violations : list[PassivityViolation]
        All intervals where Re[Z(jω)] < 0.  Empty if passive.
    freq_check : ndarray
        Frequency grid used for the check (Hz).
    re_Z_check : ndarray
        Re[Z(jω)] evaluated on freq_check.
    """
    is_passive  : bool
    min_re_Z    : float
    violations  : list[PassivityViolation] = field(default_factory=list)
    freq_check  : Optional[np.ndarray] = field(default=None, repr=False)
    re_Z_check  : Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"Passivity report",
            f"  Passive       : {self.is_passive}",
            f"  min Re[Z(jω)] : {self.min_re_Z:.6e} Ω",
            f"  Violations    : {len(self.violations)}",
        ]
        for v in self.violations:
            lines.append(f"    {v}")
        return "\n".join(lines)


@dataclass
class EnforcementResult:
    """
    Result of passivity enforcement.

    Attributes
    ----------
    model : RationalModel
        Corrected model (passive).
    converged : bool
        True if Re[Z(jω)] ≥ 0 everywhere after correction.
    n_iterations : int
        Number of correction iterations applied.
    rms_before : float
        RMS fitting error before enforcement.
    rms_after : float
        RMS fitting error after enforcement.
    rms_increase_pct : float
        Relative RMS degradation as a percentage.
    report_before : PassivityReport
        Passivity state before enforcement.
    report_after : PassivityReport
        Passivity state after enforcement.
    """
    model             : RationalModel
    converged         : bool
    n_iterations      : int
    rms_before        : float
    rms_after         : float
    rms_increase_pct  : float
    report_before     : PassivityReport
    report_after      : PassivityReport

    def summary(self) -> str:
        lines = [
            f"Passivity enforcement result",
            f"  Converged     : {self.converged}",
            f"  Iterations    : {self.n_iterations}",
            f"  RMS before    : {self.rms_before:.4e}",
            f"  RMS after     : {self.rms_after:.4e}  "
            f"(+{self.rms_increase_pct:.2f}%)",
        ]
        if not self.converged:
            lines.append(
                f"  WARNING: enforcement did not fully converge. "
                f"min Re[Z] = {self.report_after.min_re_Z:.4e}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_passivity(
    model: RationalModel,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    n_points: int = 20_000,
) -> PassivityReport:
    """
    Check whether a rational impedance model is passive.

    Re[Z(jω)] is evaluated on a dense logarithmic grid.  Any interval
    where it drops below zero is reported as a PassivityViolation.

    Parameters
    ----------
    model : RationalModel
        Rational impedance Z(s).
    freq_min : float, optional
        Lower bound of check grid in Hz.
        Defaults to 1/100 of the lowest fitting frequency, or 1 Hz.
    freq_max : float, optional
        Upper bound in Hz.
        Defaults to 100× the highest fitting frequency, or 1 THz.
    n_points : int
        Number of grid points (default 20 000 — resolves narrow violations).

    Returns
    -------
    PassivityReport
    """
    freq_check = _make_check_grid(model, freq_min, freq_max, n_points)
    re_Z       = model.evaluate(freq_check).real

    is_passive = bool(re_Z.min() >= 0)
    violations = _find_violations(freq_check, re_Z)

    return PassivityReport(
        is_passive = is_passive,
        min_re_Z   = float(re_Z.min()),
        violations = violations,
        freq_check = freq_check,
        re_Z_check = re_Z,
    )


def enforce_passivity(
    model: RationalModel,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    n_points: int = 20_000,
    max_iter: int = 300,
    margin: float = 1e-6,
    max_step_frac: float = 0.10,
    tol_rel: float = 1e-6,
    verbose: bool = False,
) -> EnforcementResult:
    """
    Enforce passivity of a rational impedance model by iterative
    minimum-norm residue perturbation.

    At each iteration the algorithm:
      1. Finds the frequency ω* of deepest Re[Z] violation.
      2. Computes the sensitivity ∂Re[Z(jω*)]/∂Re(cᵢ) = Re[1/(jω*−pᵢ)]
         for every pole pᵢ.
      3. Applies the minimum-norm real-residue correction that lifts
         Re[Z(jω*)] to exactly `margin` above zero, clamped so the
         relative change in any residue stays within `max_step_frac`.
      4. Repeats until Re[Z(jω)] ≥ 0 everywhere.

    Only the real parts of the residues are modified; imaginary parts
    (which control the reactive character) are preserved.

    Parameters
    ----------
    model : RationalModel
        Rational impedance to enforce passivity on.
        Poles must already be in the LHP (call stabilize_poles() first).
    freq_min, freq_max : float, optional
        Frequency bounds for the passivity check grid.
    n_points : int
        Grid density (default 20 000).
    max_iter : int
        Maximum correction iterations (default 300).
    margin : float
        Target minimum Re[Z] after enforcement (default 1e-6 Ω).
    max_step_frac : float
        Maximum allowed relative change per residue per iteration
        (default 0.10 = 10%).  Prevents runaway corrections when the
        violation frequency is far from all poles.
    tol_rel : float
        Relative violation threshold below which the model is treated
        as passive (default 1e-6).  Violations smaller than
        ``tol_rel * max|Z(jω)|`` are numerical noise and ignored.
    verbose : bool
        Print progress per iteration.

    Returns
    -------
    EnforcementResult

    Raises
    ------
    ValueError
        If model has RHP poles (must stabilise first).
    """
    if np.any(model.poles.real > 0):
        raise ValueError(
            "Model has RHP poles — call stabilize_poles() before "
            "enforce_passivity()."
        )

    freq_check = _make_check_grid(model, freq_min, freq_max, n_points)
    w_check    = 2.0 * np.pi * freq_check

    report_before = check_passivity(model, freq_min, freq_max, n_points)

    # Compute RMS before (on fitting frequencies if available)
    rms_before = _model_rms(model)

    if report_before.is_passive:
        if verbose:
            print("Model is already passive — no correction needed.")
        return EnforcementResult(
            model            = model,
            converged        = True,
            n_iterations     = 0,
            rms_before       = rms_before,
            rms_after        = rms_before,
            rms_increase_pct = 0.0,
            report_before    = report_before,
            report_after     = report_before,
        )

    # Work on a mutable copy of residues
    poles    = model.poles.copy()
    residues = model.residues.copy()
    d        = model.d
    e        = model.e

    converged    = False
    n_iter_done  = 0

    for iteration in range(max_iter):
        # Evaluate Re[Z] on check grid
        s_check = 1j * w_check
        Phi     = 1.0 / (s_check[:, None] - poles[None, :])   # (M, n)
        Z_vals  = (e * s_check + d
                   + np.sum(residues[None, :] * Phi, axis=1))
        re_Z    = Z_vals.real

        viol_depth = -re_Z.min()
        max_Z      = np.abs(Z_vals).max() + 1e-30

        # Converge if violation is zero OR negligibly small relative to |Z|
        if viol_depth <= 0 or viol_depth < tol_rel * max_Z:
            converged   = True
            n_iter_done = iteration
            break

        # Frequency of deepest violation
        idx_worst = int(np.argmin(re_Z))
        w_star    = w_check[idx_worst]
        s_star    = 1j * w_star

        # Sensitivity: ∂Re[Z(jω*)]/∂Re(cᵢ) = Re[1/(jω*−pᵢ)]
        sens = np.real(1.0 / (s_star - poles))   # (n,)
        ss   = float(np.dot(sens, sens))

        if ss < 1e-60:
            warnings.warn(
                f"Passivity enforcement: sensitivity near zero at "
                f"f={w_star/(2*np.pi):.3e} Hz — stopping early.",
                RuntimeWarning, stacklevel=2,
            )
            break

        # Minimum-norm correction
        k     = (viol_depth + margin) / ss
        delta = (k * sens).astype(complex)        # purely real increment

        # Clamp step size: limit each |delta_i| to max_step_frac * |c_i|
        # This prevents runaway when violation is far from all poles
        # (sens → 0 → k → ∞).
        abs_res = np.abs(residues) + 1e-30
        max_allowed = max_step_frac * abs_res
        clamp_mask  = np.abs(delta.real) > max_allowed
        if np.any(clamp_mask):
            delta.real[clamp_mask] = (
                np.sign(delta.real[clamp_mask]) * max_allowed[clamp_mask]
            )

        residues = residues + delta

        if verbose:
            f_star = w_star / (2.0 * np.pi)
            print(f"  Iter {iteration+1:3d}: depth={viol_depth:.4e} Ω "
                  f"@ f={f_star:.3e} Hz  k={k:.3e}")

    else:
        # Loop exhausted
        n_iter_done = max_iter

    # Build corrected model
    corrected = RationalModel(
        poles             = poles,
        residues          = residues,
        d                 = d,
        e                 = e,
        freq_fit          = model.freq_fit,
        H_meas            = model.H_meas,
        rms_error_history = model.rms_error_history,
    )

    report_after = check_passivity(corrected, freq_min, freq_max, n_points)
    rms_after    = _model_rms(corrected)
    rms_increase = (rms_after / (rms_before + 1e-300) - 1.0) * 100.0

    if not converged:
        warnings.warn(
            f"enforce_passivity() did not fully converge in {max_iter} "
            f"iterations.  Remaining violation: "
            f"{report_after.min_re_Z:.4e} Ω.  "
            f"Try increasing max_iter or n_points.",
            RuntimeWarning, stacklevel=2,
        )

    return EnforcementResult(
        model            = corrected,
        converged        = converged,
        n_iterations     = n_iter_done,
        rms_before       = rms_before,
        rms_after        = rms_after,
        rms_increase_pct = rms_increase,
        report_before    = report_before,
        report_after     = report_after,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_check_grid(
    model: RationalModel,
    freq_min: Optional[float],
    freq_max: Optional[float],
    n_points: int,
) -> np.ndarray:
    """Build a logarithmic frequency grid for the passivity check."""
    if model.freq_fit is not None and len(model.freq_fit) > 0:
        default_lo = model.freq_fit.min() / 100.0
        default_hi = model.freq_fit.max() * 100.0
    else:
        default_lo = 1.0
        default_hi = 1e12

    f_lo = max(freq_min or default_lo, 1e-3)
    f_hi = freq_max or default_hi

    return np.logspace(np.log10(f_lo), np.log10(f_hi), n_points)


def _find_violations(
    freq: np.ndarray,
    re_Z: np.ndarray,
) -> list[PassivityViolation]:
    """
    Identify contiguous intervals where re_Z < 0 and characterise each.
    """
    below    = re_Z < 0
    changes  = np.diff(below.astype(int))
    starts   = np.where(changes == +1)[0] + 1
    ends     = np.where(changes == -1)[0] + 1

    # Handle edge cases where violation starts or ends at array boundary
    if below[0]:
        starts = np.concatenate([[0], starts])
    if below[-1]:
        ends = np.concatenate([ends, [len(freq) - 1]])

    violations = []
    for lo, hi in zip(starts, ends):
        segment  = re_Z[lo:hi]
        idx_w    = int(np.argmin(segment))
        violations.append(PassivityViolation(
            f_lo    = float(freq[lo]),
            f_hi    = float(freq[hi]),
            f_worst = float(freq[lo + idx_w]),
            depth   = float(-segment.min()),
        ))

    return violations


def _model_rms(model: RationalModel) -> float:
    """RMS error of model vs its stored H_meas, or rms_error if no data."""
    if model.freq_fit is not None and model.H_meas is not None:
        H_fit = model.evaluate(model.freq_fit)
        return float(np.sqrt(np.mean(np.abs(H_fit - model.H_meas) ** 2)))
    return model.rms_error
