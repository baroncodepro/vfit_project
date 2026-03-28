"""
vfit/solvers/auto_order.py
───────────────────────────
Automatic model-order selection for Vector Fitting.

Problem
────────
Choosing ``n_poles`` by hand requires knowing how many resonances the
data contains.  Too few poles → high RMS; too many → over-fitting (extra
poles absorb noise instead of signal).

Strategy
─────────
Sweep ``n_poles`` from ``n_min`` to ``n_max`` in steps of 2 (conjugate
pairs), fit a model at each order, and select the best order by one of:

  ``'elbow'``  (default)
      The first order where the absolute RMS stops improving by more than
      ``rel_improvement`` relative to the previous step.  This is the
      "elbow" in the RMS-vs-order curve — adding more poles after this
      point only fits noise.

  ``'aic'``
      Akaike Information Criterion.  Penalises model complexity:
          AIC = 2k + N · ln(RSS/N)
      where k = number of free parameters (4·n_poles + 2 for d and e),
      N = number of frequency points, RSS = sum of squared residuals.
      Selects the order that minimises AIC.

  ``'bic'``
      Bayesian Information Criterion.  Stronger complexity penalty:
          BIC = k·ln(N) + N · ln(RSS/N)
      Usually selects a sparser model than AIC.

  ``'rms_threshold'``
      Select the smallest order whose RMS is below ``rms_threshold``.
      Requires ``rms_threshold`` to be set.

All strategies also expose the full sweep results so you can inspect the
trade-off curve yourself.

Reference
──────────
Akaike, H. "A new look at the statistical model identification."
IEEE Trans. Automatic Control, 19(6), 716–723, 1974.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from ..core.vector_fitting import VectorFitter, VFOptions
from ..core.rational_function import RationalModel


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderSweepResult:
    """
    Full results of an automatic order-selection sweep.

    Attributes
    ----------
    best_order : int
        Selected n_poles.
    best_model : RationalModel
        Fitted model at the selected order.
    criterion : str
        Selection criterion used.
    orders : list[int]
        All n_poles values that were tried.
    rms_values : list[float]
        Absolute RMS at each order.
    aic_values : list[float]
        AIC at each order (nan if not computed).
    bic_values : list[float]
        BIC at each order (nan if not computed).
    models : list[RationalModel]
        All fitted models (one per order).
    """
    best_order  : int
    best_model  : RationalModel
    criterion   : str
    orders      : list[int]          = field(default_factory=list)
    rms_values  : list[float]        = field(default_factory=list)
    aic_values  : list[float]        = field(default_factory=list)
    bic_values  : list[float]        = field(default_factory=list)
    models      : list[RationalModel] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        lines = [
            f"Auto order selection  (criterion='{self.criterion}')",
            f"  Best order : {self.best_order}",
            f"  Best RMS   : {self.best_model.rms_error:.4e}",
            f"",
            f"  {'n_poles':>8}  {'RMS':>12}  {'AIC':>14}  {'BIC':>14}",
            f"  {'─'*8}  {'─'*12}  {'─'*14}  {'─'*14}",
        ]
        for n, rms, aic, bic in zip(
            self.orders, self.rms_values, self.aic_values, self.bic_values
        ):
            marker = "  ◄" if n == self.best_order else ""
            aic_s  = f"{aic:>14.2f}" if not np.isnan(aic) else f"{'—':>14}"
            bic_s  = f"{bic:>14.2f}" if not np.isnan(bic) else f"{'—':>14}"
            lines.append(
                f"  {n:>8}  {rms:>12.4e}  {aic_s}  {bic_s}{marker}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def auto_order(
    freq: np.ndarray,
    H: np.ndarray,
    n_min: int = 2,
    n_max: int = 20,
    criterion: Literal["elbow", "aic", "bic", "rms_threshold"] = "elbow",
    rel_improvement: float = 0.05,
    rms_threshold: Optional[float] = None,
    vf_options: Optional[dict] = None,
    verbose: bool = False,
) -> OrderSweepResult:
    """
    Automatically select the best model order (n_poles) for a VF fit.

    Parameters
    ----------
    freq : ndarray (N,)
        Frequency array in Hz.
    H : ndarray complex (N,)
        Complex frequency response.
    n_min : int
        Minimum n_poles to try (default 2, rounded up to nearest even).
    n_max : int
        Maximum n_poles to try (default 20).
    criterion : str
        One of ``'elbow'``, ``'aic'``, ``'bic'``, ``'rms_threshold'``.
    rel_improvement : float
        For ``criterion='elbow'``: stop when the RMS improves by less
        than this fraction relative to the previous step (default 0.05 = 5%).
    rms_threshold : float, optional
        For ``criterion='rms_threshold'``: target absolute RMS.
    vf_options : dict, optional
        Extra keyword arguments passed to ``VectorFitter`` at every order.
        Common choices: ``weight='uniform'``, ``n_iter_max=50``.
    verbose : bool
        Print sweep progress.

    Returns
    -------
    OrderSweepResult

    Examples
    --------
    >>> result = auto_order(freq, Z, n_min=2, n_max=16, criterion='elbow')
    >>> print(result.summary())
    >>> model = result.best_model

    >>> result = auto_order(freq, H, criterion='aic', n_max=12)
    >>> print(result.summary())
    """
    freq = np.asarray(freq, dtype=float)
    H    = np.asarray(H,    dtype=complex)
    N    = len(freq)

    if rms_threshold is None and criterion == "rms_threshold":
        raise ValueError(
            "rms_threshold must be set when criterion='rms_threshold'."
        )

    vf_kw = vf_options or {}

    # Ensure n_min is even (conjugate pairs)
    if n_min % 2 != 0:
        n_min += 1
    if n_max % 2 != 0:
        n_max -= 1

    orders     : list[int]           = []
    rms_vals   : list[float]         = []
    aic_vals   : list[float]         = []
    bic_vals   : list[float]         = []
    models     : list[RationalModel] = []

    if verbose:
        print(f"  Auto order sweep  n={n_min}..{n_max}  criterion='{criterion}'")
        print(f"  {'n':>6}  {'RMS':>12}  {'AIC':>12}  {'BIC':>12}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*12}")

    for n in range(n_min, n_max + 1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VectorFitter(n_poles=n, **vf_kw).fit(freq, H)

        rms  = model.rms_error
        k    = 4 * n + 2          # free parameters: 2 per complex residue × n poles + d + e
        rss  = rms ** 2 * N       # residual sum of squares (≈ N·RMS²)
        aic  = 2 * k + N * np.log(rss / N + 1e-300)
        bic  = k * np.log(N) + N * np.log(rss / N + 1e-300)

        orders.append(n)
        rms_vals.append(rms)
        aic_vals.append(aic)
        bic_vals.append(bic)
        models.append(model)

        if verbose:
            print(f"  {n:>6}  {rms:>12.4e}  {aic:>12.2f}  {bic:>12.2f}")

        # Early stopping for elbow and rms_threshold
        if criterion == "rms_threshold" and rms_threshold is not None:
            if rms <= rms_threshold:
                if verbose:
                    print(f"  → RMS threshold reached at n={n}")
                break

        if criterion == "elbow" and len(rms_vals) >= 2:
            prev, curr = rms_vals[-2], rms_vals[-1]
            improvement = (prev - curr) / (prev + 1e-300)
            if improvement < rel_improvement:
                if verbose:
                    print(f"  → Elbow detected at n={n-2} "
                          f"(improvement={improvement*100:.1f}% < "
                          f"{rel_improvement*100:.0f}%)")
                break

    # ── Select best order ──────────────────────────────────────────────────
    if criterion == "elbow":
        # Use the order just before the elbow (last point where improvement
        # was still significant), or the second-to-last if we broke out early
        if len(orders) >= 2:
            best_idx = len(orders) - 2      # one step before stopping point
        else:
            best_idx = 0

    elif criterion == "aic":
        best_idx = int(np.argmin(aic_vals))

    elif criterion == "bic":
        best_idx = int(np.argmin(bic_vals))

    elif criterion == "rms_threshold":
        # Smallest order that meets the threshold
        under = [i for i, r in enumerate(rms_vals)
                 if rms_threshold is not None and r <= rms_threshold]
        best_idx = under[0] if under else int(np.argmin(rms_vals))

    else:
        raise ValueError(f"Unknown criterion '{criterion}'. "
                         f"Choose from: elbow, aic, bic, rms_threshold.")

    # Pad aic/bic with nan if elbow stopped early before computing them all
    pad = len(orders) - len(aic_vals)
    aic_vals += [float("nan")] * pad
    bic_vals += [float("nan")] * pad

    return OrderSweepResult(
        best_order  = orders[best_idx],
        best_model  = models[best_idx],
        criterion   = criterion,
        orders      = orders,
        rms_values  = rms_vals,
        aic_values  = aic_vals,
        bic_values  = bic_vals,
        models      = models,
    )
