"""
examples/04_from_measurement.py
────────────────────────────────
Full pipeline: load measured data → Vector Fitting → pole/zero analysis
→ Foster RLC synthesis → SPICE export.

This is the template to follow with your own files.
Just change the file path and column names at the top of each section.

Two datasets are demonstrated:

  Dataset A — Lossy inductor impedance  (CSV, Re/Im, MHz)
  ─────────────────────────────────────────────────────────
  File   : data_inductor_Z.csv
  Format : Frequency[MHz], Re[Z_Ohm], Im[Z_Ohm]
  Physics: Z(s) = (R_s + sL) / (1 + sC_p(R_s + sL))
           L=47 nH, R_s=0.8 Ω, C_p=0.3 pF
  Goal   : identify L, R_s, C_p from impedance measurements.

  Dataset B — 4th-order Butterworth filter S21  (CSV, dB+deg, GHz)
  ─────────────────────────────────────────────────────────────────
  File   : data_filter_S21.csv
  Format : Frequency[GHz], |S21|[dB], Phase[deg]
  Physics: 4-pole Butterworth lowpass, f_c=500 MHz
  Goal   : identify poles and predict equivalent RLC ladder.

────────────────────────────────────────────────────────────────────────────────
How to use this template with YOUR data
────────────────────────────────────────────────────────────────────────────────

1. Replace the file path:
       data = load_ri_csv("YOUR_FILE.csv", freq_unit="MHz")

2. Choose the right loader:
   - Re/Im columns        →  load_ri_csv()
   - Magnitude+Phase cols →  load_csv()
   - Touchstone .s1p/s2p  →  load_touchstone()

3. Choose n_poles (start low, increase if RMS is too high):
   - Smooth impedance  : n_poles = 4–8
   - Sharp resonances  : n_poles = 6–12  (2 per resonance + slack)
   - Broadband filter  : n_poles = order of the filter

4. Choose weight strategy:
   - weight='inverse'  (default) : good when noise is proportional to |H|
   - weight='uniform'            : better when SNR varies strongly across band
                                   (e.g. resonators, bandpass systems)

5. Check the summary printout and Bode plot before trusting the synthesis.
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup (not needed once you pip install -e .) ─────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vfit import (
    VectorFitter,
    foster_synthesis,
    export_spice_foster,
    export_spice_behavioral,
    export_spice_test_foster,
    export_spice_test_behavioral,
    load_ri_csv,
    load_csv,
    MeasurementData,
)
from vfit.core.pole_zero import (
    pole_resonant_frequency,
    pole_quality_factor,
    sort_by_frequency,
)
from vfit.visualization import bode_plot, pole_zero_map, convergence_plot


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent

def _header(title: str) -> None:
    bar = "=" * 68
    print(f"\n{bar}\n  {title}\n{bar}")

def _section(title: str) -> None:
    print(f"\n  -- {title} {'-'*(60-len(title))}")


def print_pole_table(model) -> None:
    """Print a formatted pole characterisation table."""
    poles_sorted = sort_by_frequency(model.poles)
    print(f"  {'#':>3}  {'Re(p) [rad/s]':>18}  {'Im(p) [rad/s]':>18}  "
          f"{'f0 [MHz]':>10}  {'Q':>8}  {'type':>8}")
    print(f"  {'-'*3}  {'-'*18}  {'-'*18}  {'-'*10}  {'-'*8}  {'-'*8}")
    for i, p in enumerate(poles_sorted):
        if p.imag < 0:
            continue   # skip conjugate — already shown
        f0_hz = pole_resonant_frequency(p)
        Q      = pole_quality_factor(p)
        f0_mhz = f0_hz / 1e6
        kind   = "real" if abs(p.imag) < 1e-3 * abs(p) else "complex"
        print(f"  {i:>3}  {p.real:>18.4e}  {p.imag:>18.4e}  "
              f"{f0_mhz:>10.3f}  {Q:>8.2f}  {kind:>8}")


def print_rlc_table(network) -> None:
    """Print synthesised RLC element values."""
    print(f"  {'#':>3}  {'type':>7}  {'R [Ohm]':>12}  {'L [H]':>12}  {'C [F]':>12}")
    print(f"  {'-'*3}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*12}")
    for i, b in enumerate(network.branches):
        R = f"{b.R:.4e}" if b.R is not None else "-"
        L = f"{b.L:.4e}" if b.L is not None else "-"
        C = f"{b.C:.4e}" if b.C is not None else "-"
        print(f"  {i:>3}  {b.branch_type:>7}  {R:>12}  {L:>12}  {C:>12}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset A — Lossy inductor impedance
# ─────────────────────────────────────────────────────────────────────────────

_header("Dataset A - Lossy inductor impedance  (Re/Im CSV, MHz)")

# ── Step 1: Load ──────────────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────────────────
# │  Replace this path with your own file.
# │  Adjust freq_col / re_col / im_col to match your column names or indices.
# └─────────────────────────────────────────────────────────────────────────────
data_A = load_ri_csv(
    HERE / "data_inductor_Z.csv",
    freq_col  = "Frequency[MHz]",
    re_col    = "Re[Z_Ohm]",
    im_col    = "Im[Z_Ohm]",
    freq_unit = "MHz",          # explicit — avoids auto-detection ambiguity
    label     = "Inductor Z(jw)",
)
print(data_A.summary())

# ── Step 2: Fit ───────────────────────────────────────────────────────────────
_section("Vector Fitting")

# Start with n_poles=4 for a 2nd-order physical model with parasitics.
# The inductor has one resonance → 2 poles for the resonance + 2 slack.
# Use weight='uniform': the impedance spans ~40 dB, so inverse weight
# would over-weight the low-impedance (high-frequency, capacitive) region.
model_A = VectorFitter(
    n_poles    = 4,
    n_iter_max = 50,
    weight     = "uniform",
).fit(data_A.freq_hz, data_A.H)

H_A_fit  = model_A.evaluate(data_A.freq_hz)
rms_rel  = float(np.sqrt(np.mean(
    np.abs((H_A_fit - data_A.H) / data_A.H) ** 2
)))
print(f"  n_poles     = {model_A.n_poles}")
print(f"  RMS (abs)   = {model_A.rms_error:.4e}")
print(f"  RMS (rel)   = {rms_rel:.4e}")
print(f"  Iterations  = {len(model_A.rms_error_history)}")

# ── Step 3: Pole / zero analysis ──────────────────────────────────────────────
_section("Pole characterisation")
print_pole_table(model_A)

print()
print(f"  d (direct term, ~R_s)  = {model_A.d:.4e} Ohm")
print(f"  e (s-coeff,    ~L_s)   = {model_A.e:.4e} H")
print()
print("  Interpretation:")
print("  The e-term gives the high-frequency inductance directly.")
print("  Complex poles correspond to the self-resonance of the inductor.")
print("  Real poles absorb the lossy/capacitive tails at the band edges.")

# ── Step 4: Foster RLC synthesis ──────────────────────────────────────────────
_section("Foster RLC synthesis")
network_A = foster_synthesis(model_A)
print_rlc_table(network_A)

# Validate round-trip
Z_synth  = network_A.impedance(data_A.freq_hz)
Z_model  = model_A.evaluate(data_A.freq_hz)
rms_synth = float(np.sqrt(np.mean(
    np.abs((Z_synth - Z_model) / (np.abs(Z_model) + 1e-30)) ** 2
)))
print(f"\n  Synthesis round-trip RMS = {rms_synth:.4e}  (model vs network)")

# ── Step 5: SPICE export ──────────────────────────────────────────────────────
_section("SPICE export")
foster_path    = HERE / "inductor_foster.cir"
beh_path       = HERE / "inductor_behavioral.cir"
tb_foster_path = HERE / "tb_inductor_foster.cir"
tb_beh_path    = HERE / "tb_inductor_behavioral.cir"
export_spice_foster(         network_A, foster_path,    subckt_name="INDUCTOR_MODEL")
export_spice_behavioral(     model_A,   beh_path,       subckt_name="INDUCTOR_LAPLACE")
export_spice_test_foster(    network_A, tb_foster_path, subckt_name="INDUCTOR_MODEL",
                             subckt_file=foster_path,
                             freq_start_hz=data_A.freq_hz.min(),
                             freq_stop_hz=data_A.freq_hz.max())
export_spice_test_behavioral(model_A,   tb_beh_path,    subckt_name="INDUCTOR_LAPLACE",
                             subckt_file=beh_path,
                             freq_start_hz=data_A.freq_hz.min(),
                             freq_stop_hz=data_A.freq_hz.max())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset B — 4th-order filter S21
# ─────────────────────────────────────────────────────────────────────────────

_header("Dataset B - 4th-order Butterworth S21  (dB+deg CSV, GHz)")

# ── Step 1: Load ──────────────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────────────────
# │  Magnitude+Phase loader.  Set mag_unit='dB' and phase_unit='deg'.
# └─────────────────────────────────────────────────────────────────────────────
data_B = load_csv(
    HERE / "data_filter_S21.csv",
    freq_col   = "Frequency[GHz]",
    mag_col    = "|S21|[dB]",
    phase_col  = "Phase[deg]",
    mag_unit   = "dB",
    phase_unit = "deg",
    freq_unit  = "GHz",
    label      = "Filter S21",
)
print(data_B.summary())

# ── Step 2: Fit ───────────────────────────────────────────────────────────────
_section("Vector Fitting")

# 4-pole Butterworth → n_poles=4 is exact.
# Add 2 slack poles for the noise and stopband roll-off tails.
model_B = VectorFitter(
    n_poles    = 6,
    n_iter_max = 50,
    weight     = "uniform",   # S21 varies ~40 dB; uniform avoids stop-band bias
).fit(data_B.freq_hz, data_B.H)

H_B_fit = model_B.evaluate(data_B.freq_hz)
rms_rel_B = float(np.sqrt(np.mean(
    np.abs((H_B_fit - data_B.H) / (np.abs(data_B.H) + 1e-6)) ** 2
)))
print(f"  n_poles     = {model_B.n_poles}")
print(f"  RMS (abs)   = {model_B.rms_error:.4e}")
print(f"  RMS (rel)   = {rms_rel_B:.4e}  (relative to |H|+eps)")
print(f"  Iterations  = {len(model_B.rms_error_history)}")

# ── Step 3: Pole analysis ─────────────────────────────────────────────────────
_section("Pole characterisation")
print_pole_table(model_B)
print()
print("  True Butterworth poles (4th order, f_c=500 MHz):")
print("  All have |p|=2*pi*500 MHz, angles +/-pi*(2k+3)/8, k=0,1")

# ── Step 4: Foster synthesis ──────────────────────────────────────────────────
_section("Foster RLC synthesis")
network_B = foster_synthesis(model_B)
print_rlc_table(network_B)

rms_synth_B = float(np.sqrt(np.mean(
    np.abs((network_B.impedance(data_B.freq_hz) - model_B.evaluate(data_B.freq_hz))
           / (np.abs(model_B.evaluate(data_B.freq_hz)) + 1e-30)) ** 2
)))
print(f"\n  Synthesis round-trip RMS = {rms_synth_B:.4e}")

# ── Step 5: SPICE export ──────────────────────────────────────────────────────
_section("SPICE export")
filter_foster_path    = HERE / "filter_foster.cir"
filter_beh_path       = HERE / "filter_behavioral.cir"
tb_filter_foster_path = HERE / "tb_filter_foster.cir"
tb_filter_beh_path    = HERE / "tb_filter_behavioral.cir"
export_spice_foster(         network_B, filter_foster_path,    subckt_name="FILTER_FOSTER")
export_spice_behavioral(     model_B,   filter_beh_path,       subckt_name="FILTER_LAPLACE")
export_spice_test_foster(    network_B, tb_filter_foster_path, subckt_name="FILTER_FOSTER",
                             subckt_file=filter_foster_path,
                             freq_start_hz=data_B.freq_hz.min(),
                             freq_stop_hz=data_B.freq_hz.max())
export_spice_test_behavioral(model_B,   tb_filter_beh_path,    subckt_name="FILTER_LAPLACE",
                             subckt_file=filter_beh_path,
                             freq_start_hz=data_B.freq_hz.min(),
                             freq_stop_hz=data_B.freq_hz.max())


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_C_MEAS  = "#2c7bb6"
_C_FIT   = "#d7191c"
_C_SYNTH = "#1a9641"


# ── Figure 1: Dataset A — full pipeline view ──────────────────────────────────

fig1 = plt.figure(figsize=(16, 10))
fig1.suptitle(
    "Dataset A — Lossy inductor impedance\n"
    "Pipeline: measured data → VF fit → Foster RLC → SPICE",
    fontsize=12, fontweight="bold"
)
gs1 = fig1.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

ax_mag_A   = fig1.add_subplot(gs1[0, 0])
ax_phase_A = fig1.add_subplot(gs1[1, 0])
ax_pz_A    = fig1.add_subplot(gs1[:, 1])
ax_synth_A = fig1.add_subplot(gs1[0, 2])
ax_conv_A  = fig1.add_subplot(gs1[1, 2])

f_mhz = data_A.freq_hz / 1e6

# Bode — measured vs fit vs synthesis
ax_mag_A.semilogx(f_mhz, 20*np.log10(np.abs(data_A.H) + 1e-300),
                  color=_C_MEAS, lw=1.5, alpha=0.7, label="Measured")
ax_mag_A.semilogx(f_mhz, 20*np.log10(np.abs(H_A_fit) + 1e-300),
                  color=_C_FIT, lw=2, ls="--", label=f"VF fit (n={model_A.n_poles})")
ax_mag_A.semilogx(f_mhz, 20*np.log10(np.abs(Z_synth) + 1e-300),
                  color=_C_SYNTH, lw=1.5, ls=":", label="Foster network")

ax_phase_A.semilogx(f_mhz, np.degrees(np.unwrap(np.angle(data_A.H))),
                    color=_C_MEAS, lw=1.5, alpha=0.7)
ax_phase_A.semilogx(f_mhz, np.degrees(np.unwrap(np.angle(H_A_fit))),
                    color=_C_FIT, lw=2, ls="--")
ax_phase_A.semilogx(f_mhz, np.degrees(np.unwrap(np.angle(Z_synth))),
                    color=_C_SYNTH, lw=1.5, ls=":")

for ax, ylabel in [(ax_mag_A, "|Z| (dB Ω)"), (ax_phase_A, "Phase (°)")]:
    ax.set_xlabel("Frequency (MHz)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", color="#cccccc", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
ax_mag_A.legend(fontsize=7)
ax_mag_A.set_title("Bode (measured / fit / synthesis)", fontsize=9)
ax_phase_A.set_title("Phase", fontsize=9)

# Pole-zero map
poles_A = model_A.poles
zeros_A = model_A.zeros
ax_pz_A.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
pad = max(abs(poles_A.real).max() * 0.2, 1e5)
ax_pz_A.fill_betweenx(
    [poles_A.imag.min()*1.3, poles_A.imag.max()*1.3+1],
    0, abs(poles_A.real).max()*1.5,
    alpha=0.05, color="red"
)
ax_pz_A.scatter(poles_A.real, poles_A.imag,
                marker="x", s=160, color="#e66101", linewidths=2.5,
                zorder=5, label=f"Poles ({len(poles_A)})")
if len(zeros_A):
    ax_pz_A.scatter(zeros_A.real, zeros_A.imag,
                    marker="o", s=80, facecolors="none",
                    edgecolors="#1a9641", linewidths=2, zorder=5,
                    label=f"Zeros ({len(zeros_A)})")
ax_pz_A.set_xlabel("σ (rad/s)", fontsize=9)
ax_pz_A.set_ylabel("jω (rad/s)", fontsize=9)
ax_pz_A.set_title("S-plane pole-zero map", fontsize=9)
ax_pz_A.legend(fontsize=8)
ax_pz_A.grid(True, color="#cccccc", lw=0.5)
ax_pz_A.spines["top"].set_visible(False)
ax_pz_A.spines["right"].set_visible(False)
ax_pz_A.tick_params(labelsize=8)

# Synthesis error
synth_err = np.abs(Z_synth - data_A.H)
fit_err   = np.abs(H_A_fit  - data_A.H)
ax_synth_A.semilogx(f_mhz, fit_err,   color=_C_FIT,   lw=1.5, label="VF fit error")
ax_synth_A.semilogx(f_mhz, synth_err, color=_C_SYNTH, lw=1.5, ls="--", label="Synthesis error")
ax_synth_A.set_xlabel("Frequency (MHz)", fontsize=9)
ax_synth_A.set_ylabel("|error| (Ω)", fontsize=9)
ax_synth_A.set_title("Absolute error vs measured", fontsize=9)
ax_synth_A.legend(fontsize=8)
ax_synth_A.grid(True, which="both", color="#cccccc", lw=0.5)
ax_synth_A.spines["top"].set_visible(False)
ax_synth_A.spines["right"].set_visible(False)
ax_synth_A.tick_params(labelsize=8)

# Convergence
hist_A = model_A.rms_error_history
ax_conv_A.semilogy(range(1, len(hist_A)+1), hist_A, "o-",
                   color=_C_MEAS, lw=2, ms=5)
ax_conv_A.axhline(hist_A[-1], color=_C_FIT, ls="--", lw=1,
                  label=f"Final={hist_A[-1]:.2e}")
ax_conv_A.set_xlabel("Iteration", fontsize=9)
ax_conv_A.set_ylabel("RMS Error", fontsize=9)
ax_conv_A.set_title("Convergence", fontsize=9)
ax_conv_A.legend(fontsize=8)
ax_conv_A.grid(True, which="both", color="#cccccc", lw=0.5)
ax_conv_A.spines["top"].set_visible(False)
ax_conv_A.spines["right"].set_visible(False)
ax_conv_A.tick_params(labelsize=8)

fig1.savefig(HERE / "04_inductor_pipeline.png", dpi=150)
print("\nSaved: 04_inductor_pipeline.png")


# ── Figure 2: Dataset B — filter pipeline ─────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(
    "Dataset B — 4th-order Butterworth S21 filter\n"
    "Pipeline: measured S21 → VF fit → Foster RLC",
    fontsize=12, fontweight="bold"
)
ax_bode_B, ax_pz_B, ax_conv_B = axes2

f_ghz = data_B.freq_hz / 1e9

# Bode
ax_bode_B.semilogx(f_ghz, 20*np.log10(np.abs(data_B.H) + 1e-300),
                   color=_C_MEAS, lw=1.5, alpha=0.7, label="Measured")
ax_bode_B.semilogx(f_ghz, 20*np.log10(np.abs(H_B_fit) + 1e-300),
                   color=_C_FIT, lw=2, ls="--",
                   label=f"VF fit (n={model_B.n_poles})")
ax_bode_B.axvline(0.5, color="#888", ls=":", lw=1, label="f_c = 500 MHz")
ax_bode_B.set_xlabel("Frequency (GHz)", fontsize=9)
ax_bode_B.set_ylabel("|S21| (dB)", fontsize=9)
ax_bode_B.set_title("S21 magnitude", fontsize=9)
ax_bode_B.legend(fontsize=8)
ax_bode_B.grid(True, which="both", color="#cccccc", lw=0.5)
ax_bode_B.spines["top"].set_visible(False)
ax_bode_B.spines["right"].set_visible(False)
ax_bode_B.tick_params(labelsize=8)

# Pole-zero map
poles_B = model_B.poles
zeros_B = model_B.zeros
omega_c = 2 * np.pi * 500e6
theta = np.linspace(0, 2*np.pi, 360)
ax_pz_B.plot(omega_c*np.cos(theta), omega_c*np.sin(theta),
             color="#888", lw=0.8, ls=":", label=f"|p|=2π·500 MHz")
ax_pz_B.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax_pz_B.scatter(poles_B.real, poles_B.imag,
                marker="x", s=160, color="#e66101", linewidths=2.5,
                zorder=5, label=f"Poles ({len(poles_B)})")
if len(zeros_B):
    ax_pz_B.scatter(zeros_B.real, zeros_B.imag,
                    marker="o", s=80, facecolors="none",
                    edgecolors="#1a9641", linewidths=2, zorder=5,
                    label=f"Zeros ({len(zeros_B)})")
ax_pz_B.set_xlabel("σ (rad/s)", fontsize=9)
ax_pz_B.set_ylabel("jω (rad/s)", fontsize=9)
ax_pz_B.set_title("S-plane (circle = Butterworth locus)", fontsize=9)
ax_pz_B.legend(fontsize=8)
ax_pz_B.set_aspect("equal")
ax_pz_B.grid(True, color="#cccccc", lw=0.5)
ax_pz_B.spines["top"].set_visible(False)
ax_pz_B.spines["right"].set_visible(False)
ax_pz_B.tick_params(labelsize=8)

# Convergence
hist_B = model_B.rms_error_history
ax_conv_B.semilogy(range(1, len(hist_B)+1), hist_B, "o-",
                   color=_C_MEAS, lw=2, ms=5)
ax_conv_B.axhline(hist_B[-1], color=_C_FIT, ls="--", lw=1,
                  label=f"Final={hist_B[-1]:.2e}")
ax_conv_B.set_xlabel("Iteration", fontsize=9)
ax_conv_B.set_ylabel("RMS Error", fontsize=9)
ax_conv_B.set_title("Convergence", fontsize=9)
ax_conv_B.legend(fontsize=8)
ax_conv_B.grid(True, which="both", color="#cccccc", lw=0.5)
ax_conv_B.spines["top"].set_visible(False)
ax_conv_B.spines["right"].set_visible(False)
ax_conv_B.tick_params(labelsize=8)

fig2.tight_layout()
fig2.savefig(HERE / "04_filter_pipeline.png", dpi=150)
print("Saved: 04_filter_pipeline.png")

plt.close("all")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset C — Multi-peak impedance  (5 resonances: 30/60/200/500/700 kHz)
# ─────────────────────────────────────────────────────────────────────────────

_header("Dataset C - Multi-Peak Impedance  (Re/Im CSV, kHz)")

# ── Step 1: Load ──────────────────────────────────────────────────────────────
# Generated by examples/07_multi_peak.py from five parallel RLC branches.
# True peaks: 30 kHz (Q=8), 60 kHz (Q=10), 200 kHz (Q=15),
#             500 kHz (Q=12), 700 kHz (Q=7)
data_C = load_ri_csv(
    HERE / "data_multi_peak_Z.csv",
    freq_col  = "Frequency[kHz]",
    re_col    = "Re[Z_Ohm]",
    im_col    = "Im[Z_Ohm]",
    freq_unit = "kHz",
    label     = "Multi-Peak Z(jw)",
)
print(data_C.summary())

# ── Step 2: Fit ───────────────────────────────────────────────────────────────
_section("Vector Fitting")

# 5 resonances → 5 conjugate pairs → n_poles = 10 (exact model order).
# weight='inverse': peaks span a wide dynamic range; inverse weighting
# prevents low-impedance troughs from dominating the LS cost.
model_C = VectorFitter(
    n_poles    = 10,
    n_iter_max = 150,
    weight     = "inverse",
).fit(data_C.freq_hz, data_C.H)

H_C_fit  = model_C.evaluate(data_C.freq_hz)
rms_rel_C = float(np.sqrt(np.mean(
    np.abs((H_C_fit - data_C.H) / data_C.H) ** 2
)))
print(f"  n_poles     = {model_C.n_poles}")
print(f"  RMS (abs)   = {model_C.rms_error:.4e} Ohm")
print(f"  RMS (rel)   = {rms_rel_C:.4e}  ({rms_rel_C*100:.5f} %)")
print(f"  Iterations  = {len(model_C.rms_error_history)}")

# ── Step 3: Pole analysis ─────────────────────────────────────────────────────
_section("Pole characterisation")
print_pole_table(model_C)

# Compare fitted resonances to known true values
TRUE_PEAKS = [
    (30e3,   8),
    (60e3,  10),
    (200e3, 15),
    (500e3, 12),
    (700e3,  7),
]
from vfit.core.pole_zero import pole_resonant_frequency, pole_quality_factor

upper_C = sorted(
    [(pole_resonant_frequency(p), pole_quality_factor(p))
     for p in model_C.poles if p.imag > 0],
    key=lambda t: t[0],
)

print()
print(f"  {'#':>3}  {'f0_fit (kHz)':>13}  {'f0_true (kHz)':>14}  "
      f"{'Q_fit':>8}  {'Q_true':>7}  {'df/f (%)':>10}")
for k, (f0_true, Q_true) in enumerate(TRUE_PEAKS):
    # match to the nearest fitted pole in frequency
    best = min(upper_C, key=lambda t: abs(t[0] - f0_true))
    f0_fit, Q_fit = best
    delta_f = abs(f0_fit - f0_true) / f0_true * 100
    print(f"  {k+1:>3}  {f0_fit/1e3:>13.3f}  {f0_true/1e3:>14.3f}  "
          f"{Q_fit:>8.3f}  {Q_true:>7}  {delta_f:>10.4f}")

# ── Step 4: Foster RLC synthesis ──────────────────────────────────────────────
_section("Foster RLC synthesis")
network_C = foster_synthesis(model_C)
print_rlc_table(network_C)

Z_synth_C  = network_C.impedance(data_C.freq_hz)
rms_synth_C = float(np.sqrt(np.mean(
    np.abs((Z_synth_C - model_C.evaluate(data_C.freq_hz))
           / (np.abs(model_C.evaluate(data_C.freq_hz)) + 1e-30)) ** 2
)))
print(f"\n  Synthesis round-trip RMS = {rms_synth_C:.4e}  (model vs network)")

# ── Step 5: SPICE export ──────────────────────────────────────────────────────
_section("SPICE export")
mp_foster_path    = HERE / "multi_peak_from_csv_foster.cir"
mp_beh_path       = HERE / "multi_peak_from_csv_behavioral.cir"
tb_mp_foster_path = HERE / "tb_multi_peak_from_csv_foster.cir"
tb_mp_beh_path    = HERE / "tb_multi_peak_from_csv_behavioral.cir"

export_spice_foster(         network_C, mp_foster_path,    subckt_name="MP_FOSTER")
export_spice_behavioral(     model_C,   mp_beh_path,       subckt_name="MP_LAPLACE")
export_spice_test_foster(    network_C, tb_mp_foster_path, subckt_name="MP_FOSTER",
                             subckt_file=mp_foster_path,
                             freq_start_hz=data_C.freq_hz.min(),
                             freq_stop_hz=data_C.freq_hz.max())
export_spice_test_behavioral(model_C,   tb_mp_beh_path,    subckt_name="MP_LAPLACE",
                             subckt_file=mp_beh_path,
                             freq_start_hz=data_C.freq_hz.min(),
                             freq_stop_hz=data_C.freq_hz.max())

# ── Figure 3: Dataset C — multi-peak pipeline ──────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
fig3.suptitle(
    "Dataset C — Multi-Peak Impedance loaded from CSV\n"
    "5 resonances at 30 / 60 / 200 / 500 / 700 kHz  "
    f"|  n_poles={model_C.n_poles}  |  rel-RMS={rms_rel_C:.2e}",
    fontsize=11, fontweight="bold",
)
ax_mag_C, ax_phase_C, ax_err_C, ax_conv_C = axes3.flat

f_khz_C    = data_C.freq_hz / 1e3
peak_f_khz = [30, 60, 200, 500, 700]

# Magnitude
ax_mag_C.semilogx(f_khz_C, 20*np.log10(np.abs(data_C.H) + 1e-300),
                  color=_C_MEAS, lw=2.0, label="CSV data")
ax_mag_C.semilogx(f_khz_C, 20*np.log10(np.abs(H_C_fit) + 1e-300),
                  color=_C_FIT, lw=1.8, ls="--",
                  label=f"VF fit (n={model_C.n_poles})")
ax_mag_C.semilogx(f_khz_C, 20*np.log10(np.abs(Z_synth_C) + 1e-300),
                  color=_C_SYNTH, lw=1.5, ls=":",
                  label="Foster synthesis")
for fk in peak_f_khz:
    ax_mag_C.axvline(fk, color="gray", lw=0.7, ls="--", alpha=0.5)
ax_mag_C.set_xlabel("Frequency (kHz)", fontsize=9)
ax_mag_C.set_ylabel("|Z| (dB Ohm)", fontsize=9)
ax_mag_C.set_title("Bode — Magnitude", fontsize=9)
ax_mag_C.legend(fontsize=7.5)
ax_mag_C.grid(True, which="both", color="#cccccc", lw=0.5)
ax_mag_C.spines[["top", "right"]].set_visible(False)

# Phase
ax_phase_C.semilogx(f_khz_C, np.degrees(np.angle(data_C.H)),
                    color=_C_MEAS, lw=2.0)
ax_phase_C.semilogx(f_khz_C, np.degrees(np.angle(H_C_fit)),
                    color=_C_FIT, lw=1.8, ls="--")
ax_phase_C.semilogx(f_khz_C, np.degrees(np.angle(Z_synth_C)),
                    color=_C_SYNTH, lw=1.5, ls=":")
for fk in peak_f_khz:
    ax_phase_C.axvline(fk, color="gray", lw=0.7, ls="--", alpha=0.5)
ax_phase_C.set_xlabel("Frequency (kHz)", fontsize=9)
ax_phase_C.set_ylabel("Phase (deg)", fontsize=9)
ax_phase_C.set_title("Bode — Phase", fontsize=9)
ax_phase_C.grid(True, which="both", color="#cccccc", lw=0.5)
ax_phase_C.spines[["top", "right"]].set_visible(False)

# Relative error
err_fit_C   = np.abs((H_C_fit   - data_C.H) / data_C.H) * 100
err_synth_C = np.abs((Z_synth_C - data_C.H) / data_C.H) * 100
ax_err_C.semilogx(f_khz_C, err_fit_C,   color=_C_FIT,   lw=1.5, label="VF fit")
ax_err_C.semilogx(f_khz_C, err_synth_C, color=_C_SYNTH, lw=1.5, ls=":", label="Synthesis")
ax_err_C.axhline(0.1, color="gray", lw=1.0, ls="--", label="0.1 % threshold")
ax_err_C.set_xlabel("Frequency (kHz)", fontsize=9)
ax_err_C.set_ylabel("Relative error (%)", fontsize=9)
ax_err_C.set_title("Point-wise relative error", fontsize=9)
ax_err_C.legend(fontsize=7.5)
ax_err_C.grid(True, which="both", color="#cccccc", lw=0.5)
ax_err_C.spines[["top", "right"]].set_visible(False)

# Convergence
hist_C = model_C.rms_error_history
ax_conv_C.semilogy(range(1, len(hist_C)+1), hist_C, "o-",
                   color=_C_MEAS, lw=2, ms=4)
ax_conv_C.axhline(hist_C[-1], color=_C_FIT, ls="--", lw=1,
                  label=f"Final={hist_C[-1]:.2e}")
ax_conv_C.set_xlabel("Iteration", fontsize=9)
ax_conv_C.set_ylabel("RMS Error (Ohm)", fontsize=9)
ax_conv_C.set_title("Convergence", fontsize=9)
ax_conv_C.legend(fontsize=8)
ax_conv_C.grid(True, which="both", color="#cccccc", lw=0.5)
ax_conv_C.spines[["top", "right"]].set_visible(False)

fig3.tight_layout()
fig3.savefig(HERE / "04_multi_peak_pipeline.png", dpi=150)
plt.close(fig3)
print("\nSaved: 04_multi_peak_pipeline.png")

print("\nDone.")
print()
print("SPICE files written:")
print(f"  {HERE / 'inductor_foster.cir'}")
print(f"  {HERE / 'inductor_behavioral.cir'}")
print(f"  {HERE / 'filter_foster.cir'}")
print(f"  {HERE / 'filter_behavioral.cir'}")
print(f"  {mp_foster_path}")
print(f"  {mp_beh_path}")
