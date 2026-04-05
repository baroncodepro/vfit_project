"""
examples/06_passivity.py
─────────────────────────────────────────────────────────────────────────────
Passivity check and enforcement for rational impedance models.

What is passivity and why it matters
──────────────────────────────────────
A one-port impedance Z(s) is *passive* if it cannot generate energy —
it can only absorb or store it.  For a rational model this means:

    Re[Z(jω)] ≥ 0   for all ω ∈ ℝ

If this condition is violated, a SPICE transient simulation will see
the model as an active source and the solution can diverge or oscillate
even though the original physical component is perfectly passive.

Common causes of passivity violations in VF models
────────────────────────────────────────────────────
  • Low SNR near band edges — noise pulls Re[Z] negative
  • Under-sampled resonances — the fit overshoots between data points
  • n_poles too high — extra poles fit noise with unphysical residues

This example shows three scenarios:
  1. Healthy model (already passive) — no action needed
  2. Mildly non-passive model (low noise) — small correction
  3. Severely non-passive model (high noise) — larger correction,
     with full before/after comparison
"""

from __future__ import annotations

import sys
import io
import os
import warnings
from pathlib import Path

# Ensure Unicode output works on Windows terminals with non-UTF-8 code pages
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import re as _re

from vfit import (
    VectorFitter,
    check_passivity,
    enforce_passivity,
    foster_synthesis,
    export_spice_foster,
    export_spice_behavioral,
    export_spice_test_foster,
    export_spice_test_behavioral,
)

HERE = Path(__file__).parent


def _header(t): print(f"\n{'═'*68}\n  {t}\n{'═'*68}")
def _section(t): print(f"\n  ── {t} {'─'*(60-len(t))}")


# ── True two-branch RLC network (same as example 05) ─────────────────────────
def design_rlc(f0_hz, Q, L):
    omega_0 = 2 * np.pi * f0_hz
    C = 1 / (omega_0 ** 2 * L)
    R = Q / (omega_0 * C)
    return R, L, C

R1, L1, C1 = design_rlc(150e6, 10, 100e-9)
R2, L2, C2 = design_rlc(600e6,  8,   5e-9)

freq = np.logspace(6, 10, 600)
s    = 1j * 2 * np.pi * freq

def Zp(R, L, C, s):
    return 1.0 / (1/R + 1/(s*L) + s*C)

Z_true = Zp(R1, L1, C1, s) + Zp(R2, L2, C2, s)

rng = np.random.default_rng(seed=42)


def fit_model(noise_amp: float, n_poles: int = 4) -> object:
    """Fit a VF model to the two-RLC network with given noise level."""
    noise = noise_amp * (rng.standard_normal(len(s)) +
                         1j * rng.standard_normal(len(s)))
    Z_noisy = Z_true + noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VectorFitter(
            n_poles    = n_poles,
            n_iter_max = 50,
            weight     = "uniform",
        ).fit(freq, Z_noisy)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Already passive (very low noise)
# ─────────────────────────────────────────────────────────────────────────────

_header("Scenario 1 — Clean data (noise=0.001) — already passive?")

model1 = fit_model(noise_amp=0.001)
report1 = check_passivity(model1)
print(report1.summary())

if report1.is_passive:
    print("\n  Model is passive — no enforcement needed.")
else:
    result1 = enforce_passivity(model1, verbose=True)
    print("\n" + result1.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Mild violation (moderate noise)
# ─────────────────────────────────────────────────────────────────────────────

_header("Scenario 2 — Moderate noise (noise=0.1) — mild violation")

model2  = fit_model(noise_amp=0.1)
report2 = check_passivity(model2)
print(report2.summary())

_section("Enforcing passivity")
result2 = enforce_passivity(model2, verbose=True)
print("\n" + result2.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Severe violation (high noise + over-fitted model)
# ─────────────────────────────────────────────────────────────────────────────

_header("Scenario 3 — High noise (noise=1.0) + n_poles=8 — severe violation")

model3  = fit_model(noise_amp=1.0, n_poles=8)
report3 = check_passivity(model3)
print(report3.summary())

_section("Enforcing passivity")
result3 = enforce_passivity(model3, verbose=True)
print("\n" + result3.summary())

_section("Impact on fit quality")
rms_vs_true_before = float(np.sqrt(np.mean(
    np.abs(model3.evaluate(freq) - Z_true) ** 2
)))
rms_vs_true_after = float(np.sqrt(np.mean(
    np.abs(result3.model.evaluate(freq) - Z_true) ** 2
)))
print(f"  RMS vs true Z  before enforcement: {rms_vs_true_before:.4e} Ω")
print(f"  RMS vs true Z  after  enforcement: {rms_vs_true_after:.4e} Ω")
print(f"  RMS vs noisy data change: +{result3.rms_increase_pct:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_C_TRUE   = "#444444"
_C_BEFORE = "#d7191c"   # red  — non-passive model
_C_AFTER  = "#1a9641"   # green — enforced model
_C_ZERO   = "#aaaaaa"   # zero-line
_C_SHADE  = "#ffdddd"   # violation shading

freq_plot = np.logspace(5, 11, 5000)
f_mhz     = freq_plot / 1e6


def plot_re_Z(ax, model_before, model_after, report, title, label_b, label_a):
    """Plot Re[Z(jω)] before and after enforcement, shading violations."""
    Z_b = model_before.evaluate(freq_plot).real
    Z_a = model_after.evaluate(freq_plot).real if model_after is not model_before else None

    ax.semilogx(f_mhz, Z_b, color=_C_BEFORE, lw=2.0,
                label=label_b, zorder=3)
    if Z_a is not None:
        ax.semilogx(f_mhz, Z_a, color=_C_AFTER, lw=2.0, ls="--",
                    label=label_a, zorder=4)

    ax.axhline(0, color=_C_ZERO, lw=1.0, ls=":", zorder=2)

    # Shade violation regions
    ax.fill_between(f_mhz, Z_b, 0,
                    where=Z_b < 0,
                    alpha=0.25, color=_C_BEFORE,
                    label="Violation region")

    ax.set_xlabel("Frequency (MHz)", fontsize=9)
    ax.set_ylabel("Re[Z(jω)]  (Ω)", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", color="#cccccc", lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    "Passivity enforcement — Re[Z(jω)] before and after correction\n"
    "Passive condition: Re[Z(jω)] ≥ 0  for all ω",
    fontsize=12, fontweight="bold"
)

# ── Scenario 1 ────────────────────────────────────────────────────────────────
ax_re1, ax_bode1 = axes[0]

model1_after = result1.model if not report1.is_passive else model1
plot_re_Z(ax_re1, model1, model1_after, report1,
          f"Scenario 1 (noise=0.001)  —  passive={report1.is_passive}",
          "Before", "After")
ax_re1.set_ylim(bottom=min(-1, report1.min_re_Z * 1.5))

Z_true_plot = (Zp(R1,L1,C1,1j*2*np.pi*freq_plot) +
               Zp(R2,L2,C2,1j*2*np.pi*freq_plot))
ax_bode1.semilogx(f_mhz, 20*np.log10(np.abs(Z_true_plot)+1e-300),
                  color=_C_TRUE, lw=1.5, alpha=0.6, label="True Z")
ax_bode1.semilogx(f_mhz, 20*np.log10(np.abs(model1.evaluate(freq_plot))+1e-300),
                  color=_C_BEFORE, lw=2.0, ls="--", label="Fitted")
ax_bode1.set_xlabel("Frequency (MHz)", fontsize=9)
ax_bode1.set_ylabel("|Z| (dB Ω)", fontsize=9)
ax_bode1.set_title("Bode — Scenario 1", fontsize=9)
ax_bode1.legend(fontsize=8)
ax_bode1.grid(True, which="both", color="#cccccc", lw=0.4)
ax_bode1.spines["top"].set_visible(False)
ax_bode1.spines["right"].set_visible(False)
ax_bode1.tick_params(labelsize=8)


# ── Scenario 2 ────────────────────────────────────────────────────────────────
ax_re2, ax_bode2 = axes[1]

plot_re_Z(ax_re2, model2, result2.model, report2,
          f"Scenario 2 (noise=0.1) — violations={len(report2.violations)}  "
          f"depth={report2.min_re_Z:.3f} Ω",
          f"Before (min={report2.min_re_Z:.3f} Ω)",
          f"After (min={result2.report_after.min_re_Z:.2e} Ω)")
ax_re2.set_ylim(bottom=min(-2, report2.min_re_Z * 1.5))

ax_bode2.semilogx(f_mhz, 20*np.log10(np.abs(Z_true_plot)+1e-300),
                  color=_C_TRUE, lw=1.5, alpha=0.6, label="True Z")
ax_bode2.semilogx(f_mhz, 20*np.log10(np.abs(model2.evaluate(freq_plot))+1e-300),
                  color=_C_BEFORE, lw=2.0, ls="--", label="Before")
ax_bode2.semilogx(f_mhz, 20*np.log10(np.abs(result2.model.evaluate(freq_plot))+1e-300),
                  color=_C_AFTER, lw=2.0, ls=":", label="After")
ax_bode2.set_xlabel("Frequency (MHz)", fontsize=9)
ax_bode2.set_ylabel("|Z| (dB Ω)", fontsize=9)
ax_bode2.set_title(f"Bode — Scenario 2  (RMS +{result2.rms_increase_pct:.1f}%)", fontsize=9)
ax_bode2.legend(fontsize=8)
ax_bode2.grid(True, which="both", color="#cccccc", lw=0.4)
ax_bode2.spines["top"].set_visible(False)
ax_bode2.spines["right"].set_visible(False)
ax_bode2.tick_params(labelsize=8)


# ── Scenario 3 ────────────────────────────────────────────────────────────────
ax_re3, ax_bode3 = axes[2]

plot_re_Z(ax_re3, model3, result3.model, report3,
          f"Scenario 3 (noise=1.0, n=8) — violations={len(report3.violations)}  "
          f"depth={report3.min_re_Z:.3f} Ω",
          f"Before (min={report3.min_re_Z:.3f} Ω)",
          f"After  (min={result3.report_after.min_re_Z:.2e} Ω)")
ax_re3.set_ylim(bottom=min(-5, report3.min_re_Z * 1.5))

ax_bode3.semilogx(f_mhz, 20*np.log10(np.abs(Z_true_plot)+1e-300),
                  color=_C_TRUE, lw=1.5, alpha=0.6, label="True Z")
ax_bode3.semilogx(f_mhz, 20*np.log10(np.abs(model3.evaluate(freq_plot))+1e-300),
                  color=_C_BEFORE, lw=2.0, ls="--", label="Before")
ax_bode3.semilogx(f_mhz, 20*np.log10(np.abs(result3.model.evaluate(freq_plot))+1e-300),
                  color=_C_AFTER, lw=2.0, ls=":", label="After")
ax_bode3.set_xlabel("Frequency (MHz)", fontsize=9)
ax_bode3.set_ylabel("|Z| (dB Ω)", fontsize=9)
ax_bode3.set_title(f"Bode — Scenario 3  (RMS +{result3.rms_increase_pct:.1f}%)", fontsize=9)
ax_bode3.legend(fontsize=8)
ax_bode3.grid(True, which="both", color="#cccccc", lw=0.4)
ax_bode3.spines["top"].set_visible(False)
ax_bode3.spines["right"].set_visible(False)
ax_bode3.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(HERE / "06_passivity.png", dpi=150, bbox_inches="tight")
print("\nSaved: 06_passivity.png")
# ─────────────────────────────────────────────────────────────────────────────
# SPICE export — passivity-enforced models for all 3 scenarios
# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: model is already passive — export as-is
# Scenario 2: export the post-enforcement model (result2.model)
# Scenario 3: export the post-enforcement model (result3.model)
#
# Verify with:
#   python plot_ltspice_bode.py --preset passivity1
#   python plot_ltspice_bode.py --preset passivity2
#   python plot_ltspice_bode.py --preset passivity3

print("\n── SPICE export ────────────────────────────────────")

_passive_model1 = model1 if report1.is_passive else result1.model

_exports = [
    ("passivity1", _passive_model1, "PASSIVITY1", "Scenario 1 — clean (noise=0.001)"),
    ("passivity2", result2.model,   "PASSIVITY2", "Scenario 2 — enforced (noise=0.1)"),
    ("passivity3", result3.model,   "PASSIVITY3", "Scenario 3 — enforced (noise=1.0, n=8)"),
]

for stem, model, subckt, desc in _exports:
    foster_net = foster_synthesis(model)
    foster_cir = HERE / f"{stem}_foster.cir"
    beh_cir    = HERE / f"{stem}_behavioral.cir"
    tb_foster  = HERE / f"tb_{stem}_foster.cir"
    tb_beh     = HERE / f"tb_{stem}_behavioral.cir"

    export_spice_foster(foster_net, foster_cir, subckt_name=f"{subckt}_FOSTER")
    export_spice_behavioral(model,  beh_cir,    subckt_name=f"{subckt}_LAPLACE")
    export_spice_test_foster(foster_net, tb_foster,
                             subckt_name=f"{subckt}_FOSTER", subckt_file=foster_cir)
    export_spice_test_behavioral(model, tb_beh,
                                 subckt_name=f"{subckt}_LAPLACE", subckt_file=beh_cir)
    print(f"  {desc}")
    print(f"    Foster  : {foster_cir.name}  +  {tb_foster.name}")
    print(f"    Laplace : {beh_cir.name}  +  {tb_beh.name}")

plt.show()
print("\nDone.")
