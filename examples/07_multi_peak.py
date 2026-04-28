"""
examples/07_multi_peak.py
────────────────────────────────────────────────────────────────────────────────
Five-resonance impedance — peaks at 30 kHz, 60 kHz, 200 kHz, 500 kHz, 700 kHz.

Network topology: five parallel RLC branches wired in series (Foster-I form)

    Port ●──[Z₁]──[Z₂]──[Z₃]──[Z₄]──[Z₅]──● Port

Each branch Zᵢ is a parallel RLC tuned to resonance at fᵢ:

         sLᵢ                    s · ωᵢ/Qᵢ
Zᵢ(s) = ─────────────  =  Rᵢ · ─────────────────────
         LᵢCᵢs²+Lᵢ/Rᵢ·s+1       s² + s·ωᵢ/Qᵢ + ωᵢ²

     where ωᵢ = 2π·fᵢ,  |Zᵢ(jωᵢ)| = Rᵢ  (peak impedance)

Branch parameters
─────────────────
  Branch 1:  f₁ =  30 kHz,  Q₁ =  8,  R₁ =  500 Ω
  Branch 2:  f₂ =  60 kHz,  Q₂ = 10,  R₂ =  800 Ω
  Branch 3:  f₃ = 200 kHz,  Q₃ = 15,  R₃ = 1200 Ω
  Branch 4:  f₄ = 500 kHz,  Q₄ = 12,  R₄ =  600 Ω
  Branch 5:  f₅ = 700 kHz,  Q₅ =  7,  R₅ =  300 Ω

Goal
────
Verify that VectorFitter with n_poles=10 (5 peaks × 2 poles each = exact order)
can recover all five resonances from the composite impedance.

Expected output
───────────────
  • Relative RMS error  < 0.1 %
  • Fitted f₀ values within ~0.1 % of true resonant frequencies
  • Fitted Q values within ~1 % of true branch Q values
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vfit import (
    VectorFitter,
    foster_synthesis,
    export_spice_foster,
    export_spice_behavioral,
    export_spice_test_foster,
    export_spice_test_behavioral,
)
from vfit.core.pole_zero import pole_resonant_frequency, pole_quality_factor

HERE = Path(__file__).parent

# ── Branch parameters ──────────────────────────────────────────────────────────
PEAKS = [
    (30e3,   8,  500.0),    # (f0_Hz, Q, R_Ohm)
    (60e3,  10,  800.0),
    (200e3, 15, 1200.0),
    (500e3, 12,  600.0),
    (700e3,  7,  300.0),
]

PEAK_LABELS = ["30 kHz", "60 kHz", "200 kHz", "500 kHz", "700 kHz"]


def parallel_rlc_z(s: np.ndarray, f0: float, Q: float, R: float) -> np.ndarray:
    """Impedance of one parallel RLC branch resonant at f0 with peak value R."""
    w0 = 2.0 * np.pi * f0
    # Z(s) = R * (s*w0/Q) / (s² + s*w0/Q + w0²)  →  |Z(j*w0)| = R
    return R * (s * w0 / Q) / (s**2 + s * (w0 / Q) + w0**2)


# ── Frequency sweep: 5 kHz → 2 MHz ────────────────────────────────────────────
freq  = np.logspace(np.log10(5e3), np.log10(2e6), 600)
omega = 2.0 * np.pi * freq
s     = 1j * omega

# ── Analytical impedance ───────────────────────────────────────────────────────
Z_meas = sum(parallel_rlc_z(s, f0, Q, R) for f0, Q, R in PEAKS)

# ── Save CSV for use by example 04 ────────────────────────────────────────────
csv_path = HERE / "data_multi_peak_Z.csv"
csv_data = np.column_stack([freq / 1e3, Z_meas.real, Z_meas.imag])
header   = "Frequency[kHz],Re[Z_Ohm],Im[Z_Ohm]"
np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
print(f"CSV saved: {csv_path.name}  ({len(freq)} rows)")

# ── Print true branch parameters ──────────────────────────────────────────────
print("=" * 72)
print("Example 07 — Five-Resonance Impedance (Multi-Peak)")
print("=" * 72)
print("\nTrue branch parameters:")
print(f"  {'Branch':>8}  {'f0 (kHz)':>10}  {'Q':>5}  {'R (Ohm)':>8}  "
      f"{'L (uH)':>10}  {'C (nF)':>10}")
for k, (f0, Q, R) in enumerate(PEAKS, 1):
    w0 = 2.0 * np.pi * f0
    L  = R / (Q * w0)
    C  = Q / (R * w0)
    print(f"  {k:>8}  {f0/1e3:>10.0f}  {Q:>5}  {R:>8.0f}  "
          f"{L*1e6:>10.4f}  {C*1e9:>10.4f}")

# ── Vector Fitting ─────────────────────────────────────────────────────────────
# Exact order: 5 resonances → 5 conjugate pairs → 10 poles
N_POLES = 10
print(f"\nVector Fitting: n_poles={N_POLES}  (5 peaks × 2 poles — exact order)")

model = VectorFitter(
    n_poles=N_POLES, n_iter_max=150, weight="inverse"
).fit(freq, Z_meas)

Z_fit   = model.evaluate(freq)
rms_rel = float(np.sqrt(np.mean(np.abs((Z_fit - Z_meas) / Z_meas) ** 2)))

print(f"\nVF result:")
print(f"  RMS absolute  = {model.rms_error:.4e} Ohm")
print(f"  RMS relative  = {rms_rel:.4e}  ({rms_rel * 100:.5f} %)")
print(f"  Iterations    = {len(model.rms_error_history)}")

# ── Pole characterization ──────────────────────────────────────────────────────
print("\nFitted poles (upper half-plane, sorted by frequency):")
print(f"  {'#':>3}  {'f0_fit (kHz)':>13}  {'f0_true (kHz)':>14}  "
      f"{'Q_fit':>8}  {'Q_true':>8}  {'df/f (%)':>10}")

upper = [(pole_resonant_frequency(p), pole_quality_factor(p), p)
         for p in model.poles if p.imag > 0]
upper.sort(key=lambda t: t[0])

for idx, (f0_fit, Q_fit, _) in enumerate(upper):
    if idx < len(PEAKS):
        f0_true, Q_true, _ = PEAKS[idx]
        delta_f = abs(f0_fit - f0_true) / f0_true * 100
    else:
        f0_true, Q_true, delta_f = float("nan"), float("nan"), float("nan")
    print(f"  {idx+1:>3}  {f0_fit/1e3:>13.3f}  {f0_true/1e3:>14.3f}  "
          f"{Q_fit:>8.3f}  {Q_true:>8.3f}  {delta_f:>10.4f}")

# ── Foster RLC synthesis ───────────────────────────────────────────────────────
print("\nFoster RLC synthesis ...")
network   = foster_synthesis(model)
Z_synth   = network.impedance(freq)
rms_synth = float(np.sqrt(np.mean(np.abs((Z_synth - Z_meas) / Z_meas) ** 2)))
print(f"  Round-trip relative RMS = {rms_synth:.4e}  ({rms_synth * 100:.5f} %)")

# ── SPICE export ───────────────────────────────────────────────────────────────
foster_cir = HERE / "multi_peak_foster.cir"
beh_cir    = HERE / "multi_peak_behavioral.cir"
tb_foster  = HERE / "tb_multi_peak_foster.cir"
tb_beh     = HERE / "tb_multi_peak_behavioral.cir"

export_spice_foster(network, foster_cir,  subckt_name="MULTI_PEAK_FOSTER")
export_spice_behavioral(model, beh_cir,   subckt_name="MULTI_PEAK_LAPLACE")
export_spice_test_foster(network, tb_foster,
                         subckt_name="MULTI_PEAK_FOSTER",  subckt_file=foster_cir,
                         freq_start_hz=freq.min(), freq_stop_hz=freq.max())
export_spice_test_behavioral(model, tb_beh,
                             subckt_name="MULTI_PEAK_LAPLACE", subckt_file=beh_cir,
                             freq_start_hz=freq.min(), freq_stop_hz=freq.max())
print(f"\nSPICE exports: {foster_cir.name}, {beh_cir.name}, "
      f"{tb_foster.name}, {tb_beh.name}")

# ── Sanity check: pass / fail ──────────────────────────────────────────────────
PASS_THRESHOLD_REL = 1e-3   # 0.1 % relative RMS
f0_errors = []
for idx, (f0_fit, Q_fit, _) in enumerate(upper):
    if idx < len(PEAKS):
        f0_true = PEAKS[idx][0]
        f0_errors.append(abs(f0_fit - f0_true) / f0_true)

max_f0_err = max(f0_errors) if f0_errors else float("nan")

print("\n" + "-" * 72)
print("PASS / FAIL summary")
print("-" * 72)
_rms_ok = rms_rel < PASS_THRESHOLD_REL
_f0_ok  = max_f0_err < 0.005   # < 0.5 % frequency error
print(f"  Relative RMS < 0.1 % :  {'PASS' if _rms_ok  else 'FAIL'}  "
      f"({rms_rel*100:.5f} %)")
print(f"  Max df/f     < 0.5 % :  {'PASS' if _f0_ok   else 'FAIL'}  "
      f"({max_f0_err*100:.4f} %)")
print("-" * 72)

# ── Plot ───────────────────────────────────────────────────────────────────────
f_khz        = freq / 1e3
peak_f_khz   = [f0 / 1e3 for f0, _, _ in PEAKS]
Z_meas_db    = 20.0 * np.log10(np.abs(Z_meas))
Z_fit_db     = 20.0 * np.log10(np.abs(Z_fit)   + 1e-300)
Z_synth_db   = 20.0 * np.log10(np.abs(Z_synth) + 1e-300)
err_fit      = np.abs((Z_fit   - Z_meas) / Z_meas) * 100.0
err_synth    = np.abs((Z_synth - Z_meas) / Z_meas) * 100.0

fig = plt.figure(figsize=(13, 10))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

ax_mag   = fig.add_subplot(gs[0, :])
ax_phase = fig.add_subplot(gs[1, :], sharex=ax_mag)
ax_err   = fig.add_subplot(gs[2, 0])
ax_conv  = fig.add_subplot(gs[2, 1])

fig.suptitle(
    "Multi-Peak Impedance — 5 Resonances at 30 / 60 / 200 / 500 / 700 kHz\n"
    f"VectorFitter  n_poles={N_POLES}  |  rel-RMS={rms_rel:.2e}  "
    f"|  iters={len(model.rms_error_history)}",
    fontsize=11, fontweight="bold",
)

# — Magnitude ——————————————————————————————————————————————
ax_mag.semilogx(f_khz, Z_meas_db,  color="#2c7bb6", lw=2.2, label="Analytical Z(s)")
ax_mag.semilogx(f_khz, Z_fit_db,   color="#d7191c", lw=1.8, ls="--",
                label=f"VF fit  (rel-RMS={rms_rel:.2e})")
ax_mag.semilogx(f_khz, Z_synth_db, color="#1a9641", lw=1.5, ls=":",
                label=f"Foster synth  (rel-RMS={rms_synth:.2e})")

y_bot, y_top = ax_mag.get_ylim()
for fk, lbl in zip(peak_f_khz, PEAK_LABELS):
    ax_mag.axvline(fk, color="gray", lw=0.8, ls="--", alpha=0.55)
    ax_mag.text(fk * 1.04, y_bot + 0.08 * (y_top - y_bot),
                lbl, fontsize=6.5, color="dimgray", rotation=90, va="bottom")

ax_mag.set_ylabel("|Z| (dB Ω)")
ax_mag.legend(fontsize=8, loc="upper right")
ax_mag.grid(True, which="both", alpha=0.35)
ax_mag.set_title("Bode — Magnitude", fontsize=9, loc="left")
ax_mag.spines[["top", "right"]].set_visible(False)

# — Phase ──────────────────────────────────────────────────
ax_phase.semilogx(f_khz, np.degrees(np.angle(Z_meas)),  color="#2c7bb6", lw=2.2)
ax_phase.semilogx(f_khz, np.degrees(np.angle(Z_fit)),   color="#d7191c", lw=1.8, ls="--")
ax_phase.semilogx(f_khz, np.degrees(np.angle(Z_synth)), color="#1a9641", lw=1.5, ls=":")
for fk in peak_f_khz:
    ax_phase.axvline(fk, color="gray", lw=0.8, ls="--", alpha=0.55)
ax_phase.set_ylabel("Phase (deg)")
ax_phase.set_xlabel("Frequency (kHz)")
ax_phase.grid(True, which="both", alpha=0.35)
ax_phase.set_title("Bode — Phase", fontsize=9, loc="left")
ax_phase.spines[["top", "right"]].set_visible(False)

# — Relative error ─────────────────────────────────────────
ax_err.semilogx(f_khz, err_fit,   color="#d7191c", lw=1.5, label="VF fit")
ax_err.semilogx(f_khz, err_synth, color="#1a9641", lw=1.5, ls=":", label="Foster synth")
ax_err.axhline(PASS_THRESHOLD_REL * 100, color="gray", lw=1.0, ls="--",
               label="0.1 % threshold")
ax_err.set_ylabel("Relative error (%)")
ax_err.set_xlabel("Frequency (kHz)")
ax_err.set_title("Point-wise relative error", fontsize=9, loc="left")
ax_err.legend(fontsize=7.5)
ax_err.grid(True, which="both", alpha=0.35)
ax_err.spines[["top", "right"]].set_visible(False)

# — Convergence ────────────────────────────────────────────
iters = np.arange(1, len(model.rms_error_history) + 1)
ax_conv.semilogy(iters, model.rms_error_history,
                 color="#7b2d8b", lw=1.8, marker="o", ms=3)
ax_conv.set_xlabel("Iteration")
ax_conv.set_ylabel("RMS error (Ω)")
ax_conv.set_title("VF convergence history", fontsize=9, loc="left")
ax_conv.grid(True, which="both", alpha=0.35)
ax_conv.spines[["top", "right"]].set_visible(False)

out_png = HERE / "07_multi_peak.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved plot: {out_png.name}")
print("\nDone.")
