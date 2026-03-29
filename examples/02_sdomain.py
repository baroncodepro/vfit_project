"""
examples/02_sdomain.py
──────────────────────
S-domain system identification with Vector Fitting.

Three transfer functions of increasing complexity are fitted in sequence,
demonstrating how VF handles:

  Case 1 — Simple zero / two real LHP poles
  ─────────────────────────────────────────
    H₁(s) =        s
             ─────────────────
             (s + 1)(s + 3)

    Poles : p = -1, -3  (real, LHP)
    Zero  : z =  0
    DC gain: H₁(0) = 0   (zero at origin)

  Case 2 — Bandpass resonator (complex conjugate pair)
  ─────────────────────────────────────────────────────
    H₂(s) =        2s
             ─────────────────────
             s² + 0.4s + 4

    Poles : p = -0.2 ± j·√3.96  ≈  -0.2 ± j1.99  (underdamped)
    Zero  : z =  0
    Natural frequency: ω₀ = 2 rad/s  →  f₀ ≈ 0.318 Hz
    Q factor         : Q  = ω₀ / (2·0.2) = 5

  Case 3 — Higher-order with RHP poles (enforce_stability demo)
  ─────────────────────────────────────────────────────────────
    H₃(s) =        s
             ─────────────────
             s² − 3s + 2
           =        s
             ─────────────────
             (s − 1)(s − 2)

    Poles : p = +1, +2  (RHP — unstable system)
    VF will detect, warn, and reflect them to LHP.
    The stabilised fit is a valid macromodel for simulation.

Reference: Gustavsen & Semlyen, IEEE Trans. Power Delivery, 1999.
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

from vfit import (
    VectorFitter,
    foster_synthesis,
    export_spice_foster,
    export_spice_behavioral,
    export_spice_test_foster,
    export_spice_test_behavioral,
)
from vfit.core.pole_zero import pole_resonant_frequency, pole_quality_factor
from vfit.visualization import bode_plot, pole_zero_map, convergence_plot

HERE = Path(os.path.dirname(__file__))


# ── Helper: print a labelled section header ───────────────────────────────────

def _header(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — Two real LHP poles
# ─────────────────────────────────────────────────────────────────────────────

_header("Case 1 — H(s) = s / ((s+1)(s+3))")

freq1   = np.logspace(-2, 2, 500)          # 0.01 Hz → 100 Hz
s1      = 1j * 2 * np.pi * freq1
H1_true = s1 / ((s1 + 1) * (s1 + 3))

# Analytical poles and residues (partial fractions):
#   H(s) = -1/(2(s+1))  +  3/(2(s+3))  ... wait, recalculate:
#   c at p=-1: lim_{s→-1} (s+1)·H(s) = -1 / (-1+3)   = -1/2
#   c at p=-3: lim_{s→-3} (s+3)·H(s) = -3 / (-3+1)   = +3/2
true_poles1    = np.array([-1.0, -3.0])
true_residues1 = np.array([-0.5, 1.5])

model1 = VectorFitter(n_poles=2, n_iter_max=30).fit(freq1, H1_true)

H1_fit  = model1.evaluate(freq1)
rms_rel = np.sqrt(np.mean(np.abs((H1_fit - H1_true) / H1_true) ** 2))

print(f"  True  poles     : {true_poles1}")
print(f"  True  residues  : {true_residues1}")
print()
print(f"  Fitted poles    : {np.sort(model1.poles.real)}")
print(f"  Fitted residues : {model1.residues[np.argsort(model1.poles.real)].real}")
print(f"  Fitted zeros    : {model1.zeros.real}")
print()
print(f"  Relative RMS    : {rms_rel:.3e}")
print(f"  Converged in    : {len(model1.rms_error_history)} iterations")


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — Complex conjugate pair (bandpass resonator)
# ─────────────────────────────────────────────────────────────────────────────

_header("Case 2 — H(s) = 2s / (s² + 0.4s + 4)   [bandpass, Q=5]")

omega_0 = 2.0                              # rad/s
alpha   = 0.2                              # damping (= ω₀ / 2Q)
omega_d = np.sqrt(omega_0**2 - alpha**2)   # damped natural frequency

p_upper = -alpha + 1j * omega_d
p_lower = -alpha - 1j * omega_d

freq2   = np.logspace(-2, 1, 600)          # 0.01 Hz → 10 Hz  (centred on resonance)
s2      = 1j * 2 * np.pi * freq2
H2_true = 2 * s2 / (s2**2 + 2 * alpha * s2 + omega_0**2)

model2 = VectorFitter(n_poles=2, n_iter_max=30).fit(freq2, H2_true)

H2_fit  = model2.evaluate(freq2)
rms_rel2 = np.sqrt(np.mean(np.abs((H2_fit - H2_true) / H2_true) ** 2))

# Characterise fitted poles
for i, p in enumerate(model2.poles):
    if p.imag > 0:
        f0_fit = pole_resonant_frequency(p)
        Q_fit  = pole_quality_factor(p)
        break

f0_true = omega_0 / (2 * np.pi)
Q_true  = omega_0 / (2 * alpha)

print(f"  True  ω₀ = {omega_0:.4f} rad/s   →  f₀ = {f0_true:.4f} Hz,  Q = {Q_true:.2f}")
print(f"  Fitted f₀ = {f0_fit:.4f} Hz,  Q = {Q_fit:.2f}")
print(f"  True  poles : {p_upper:.4f},  {p_lower:.4f}")
print(f"  Fitted poles: {model2.poles}")
print()
print(f"  Relative RMS : {rms_rel2:.3e}")
print(f"  Converged in : {len(model2.rms_error_history)} iterations")


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — RHP poles  (stability enforcement demo)
# ─────────────────────────────────────────────────────────────────────────────

_header("Case 3 — H(s) = s / (s²−3s+2) = s / ((s−1)(s−2))   [RHP poles]")

freq3   = np.logspace(-1, 2, 500)          # 0.1 Hz → 100 Hz
s3      = 1j * 2 * np.pi * freq3
H3_true = s3 / (s3**2 - 3 * s3 + 2)

# ── 3a: with stability enforcement (default) ─────────────────────────────────
print("  3a) enforce_stability=True  (default — safe for simulation)")
with warnings.catch_warnings(record=True) as w3a:
    warnings.simplefilter("always")
    model3a = VectorFitter(n_poles=2, n_iter_max=30).fit(freq3, H3_true)

for warn in w3a:
    print(f"       [{warn.category.__name__}] {warn.message}")

print(f"       Fitted poles : {np.round(model3a.poles, 6)}")
print(f"       All LHP?     : {all(p.real <= 0 for p in model3a.poles)}")

# ── 3b: without stability enforcement ────────────────────────────────────────
print()
print("  3b) enforce_stability=False  (recovers true RHP poles exactly)")
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    model3b = VectorFitter(
        n_poles=2, n_iter_max=30, enforce_stability=False
    ).fit(freq3, H3_true)

idx     = np.argsort(model3b.poles.real)
poles_s = model3b.poles[idx]
res_s   = model3b.residues[idx]
zeros_s = model3b.zeros

rms_3b  = np.sqrt(
    np.mean(np.abs((model3b.evaluate(freq3) - H3_true) / H3_true) ** 2)
)

print(f"       True  poles     : +1.000000,  +2.000000")
print(f"       Fitted poles    : {poles_s[0].real:+.6f},  {poles_s[1].real:+.6f}")
print(f"       True  residues  :  -1.000000,  +2.000000")
print(f"       Fitted residues : {res_s[0].real:+.6f},  {res_s[1].real:+.6f}")
print(f"       True  zero      :  0.000000")
print(f"       Fitted zeros    : {np.round(zeros_s.real, 6)}")
print(f"       Relative RMS    : {rms_3b:.3e}  (near machine epsilon)")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

# ── Figure 1: Bode plots for all three cases ──────────────────────────────────

fig1, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
fig1.suptitle("S-domain Vector Fitting — Bode Plots", fontsize=13, fontweight="bold")

cases = [
    (freq1, H1_true, model1, "Case 1:  s / ((s+1)(s+3))"),
    (freq2, H2_true, model2, "Case 2:  2s / (s²+0.4s+4)   [Q=5]"),
    (freq3, H3_true, model3b, "Case 3:  s / (s²−3s+2)   [RHP]"),
]

_C_MEAS = "#2c7bb6"
_C_FIT  = "#d7191c"

for col, (freq, H_true, model, title) in enumerate(cases):
    ax_mag   = axes[0, col]
    ax_phase = axes[1, col]

    f_scale = 1e-3 if freq.max() >= 1e3 else 1.0
    f_label = "Frequency (kHz)" if f_scale == 1e-3 else "Frequency (Hz)"
    f_disp  = freq * f_scale

    H_fit   = model.evaluate(freq)
    mag_m   = 20 * np.log10(np.abs(H_true)  + 1e-300)
    mag_f   = 20 * np.log10(np.abs(H_fit)   + 1e-300)
    ph_m    = np.degrees(np.unwrap(np.angle(H_true)))
    ph_f    = np.degrees(np.unwrap(np.angle(H_fit)))

    ax_mag.semilogx(f_disp, mag_m, color=_C_MEAS, lw=1.5, label="True")
    ax_mag.semilogx(f_disp, mag_f, color=_C_FIT,  lw=2.0, ls="--",
                    label=f"VF fit  (RMS={model.rms_error:.1e})")
    ax_phase.semilogx(f_disp, ph_m, color=_C_MEAS, lw=1.5)
    ax_phase.semilogx(f_disp, ph_f, color=_C_FIT,  lw=2.0, ls="--")

    ax_mag.set_title(title, fontsize=9)
    ax_mag.legend(fontsize=8)
    ax_phase.set_xlabel(f_label, fontsize=9)
    for ax in (ax_mag, ax_phase):
        ax.grid(True, which="both", color="#cccccc", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

axes[0, 0].set_ylabel("Magnitude (dB)", fontsize=9)
axes[1, 0].set_ylabel("Phase (°)",      fontsize=9)

fig1.tight_layout()
fig1.savefig("02_sdomain_bode.png", dpi=150)
print("\nSaved: 02_sdomain_bode.png")


# ── Figure 2: Pole-zero maps for all three cases ──────────────────────────────

fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("S-domain Vector Fitting — Pole-Zero Maps", fontsize=13, fontweight="bold")

pz_cases = [
    (model1,  "Case 1  (real LHP poles)"),
    (model2,  "Case 2  (complex pair, Q=5)"),
    (model3b, "Case 3  (RHP poles, enforce_stability=False)"),
]

_C_POLE = "#e66101"
_C_ZERO = "#1a9641"

for ax, (model, title) in zip(axs2, pz_cases):
    poles = model.poles
    zeros = model.zeros

    ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.4, label="jω axis")
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1e6,
               alpha=0.04, color="red")

    ax.scatter(poles.real, poles.imag, marker="x", s=140, color=_C_POLE,
               linewidths=2.5, zorder=5, label=f"Poles ({len(poles)})")
    ax.scatter(zeros.real, zeros.imag, marker="o", s=90, facecolors="none",
               edgecolors=_C_ZERO, linewidths=2.0, zorder=5,
               label=f"Zeros ({len(zeros)})")

    # Shade RHP
    xlim = ax.get_xlim()
    x_right = max(poles.real.max() * 1.5, 0.5)
    ax.fill_betweenx(
        [poles.imag.min() * 2, poles.imag.max() * 2 + 1],
        0, x_right, alpha=0.06, color="red",
    )

    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, color="#cccccc", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("σ  (rad/s)", fontsize=9)
    ax.set_ylabel("jω  (rad/s)", fontsize=9)
    ax.tick_params(labelsize=8)

fig2.tight_layout()
fig2.savefig("02_sdomain_polezero.png", dpi=150)
print("Saved: 02_sdomain_polezero.png")


# ── Figure 3: Convergence for all three cases ─────────────────────────────────

fig3, axs3 = plt.subplots(1, 3, figsize=(13, 4))
fig3.suptitle("S-domain Vector Fitting — Convergence History",
              fontsize=13, fontweight="bold")

conv_cases = [
    (model1,  "Case 1"),
    (model2,  "Case 2"),
    (model3b, "Case 3  (enforce_stability=False)"),
]

for ax, (model, title) in zip(axs3, conv_cases):
    hist  = model.rms_error_history
    iters = np.arange(1, len(hist) + 1)
    ax.semilogy(iters, hist, "o-", color=_C_MEAS, lw=2, ms=6)
    ax.axhline(hist[-1], color=_C_FIT, ls="--", lw=1,
               label=f"Final = {hist[-1]:.2e}")
    ax.set_xticks(iters)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("Absolute RMS Error", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", color="#cccccc", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

fig3.tight_layout()
fig3.savefig("02_sdomain_convergence.png", dpi=150)
print("Saved: 02_sdomain_convergence.png")


plt.show()
print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# SPICE export — enables LTspice verification via plot_ltspice_bode.py
# ─────────────────────────────────────────────────────────────────────────────
# Note: these are normalised transfer functions (frequency axis in Hz, not MHz).
# The LAPLACE E-source handles any frequency range, so LTspice sweeps fine.
# Testbench sweep ranges are set to cover the fitted frequency axes.
#
# Verify with:
#   python plot_ltspice_bode.py --preset sdomain1
#   python plot_ltspice_bode.py --preset sdomain2
#   python plot_ltspice_bode.py --preset sdomain3

print("\n── SPICE export ────────────────────────────────────")

import re as _re

#  (stable_poles, model, subckt_prefix, sweep Hz range, description)
_cases = [
    (True,  model1,  "SDOMAIN1",  "0.01 100",   "Case 1:  H=s/((s+1)(s+3))"),
    (True,  model2,  "SDOMAIN2",  "0.01 10",    "Case 2:  H=2s/(s^2+0.4s+4)"),
    (False, model3b, "SDOMAIN3",  "0.1  100",   "Case 3:  H=s/(s^2-3s+2) RHP (behavioral only)"),
]

for stable, model, subckt, sweep_range, desc in _cases:
    beh_cir = HERE / f"{subckt.lower()}_behavioral.cir"
    tb_beh  = HERE / f"tb_{subckt.lower()}_behavioral.cir"

    export_spice_behavioral(model, beh_cir, subckt_name=f"{subckt}_LAPLACE")
    export_spice_test_behavioral(model, tb_beh,
                                 subckt_name=f"{subckt}_LAPLACE",
                                 subckt_file=beh_cir)
    txt = tb_beh.read_text(encoding="utf-8")
    txt = _re.sub(r"\.ac dec \d+ \S+ \S+", f".ac dec 100 {sweep_range}", txt)
    tb_beh.write_text(txt, encoding="utf-8")

    if stable:
        foster_net = foster_synthesis(model)
        foster_cir = HERE / f"{subckt.lower()}_foster.cir"
        tb_foster  = HERE / f"tb_{subckt.lower()}_foster.cir"
        export_spice_foster(foster_net, foster_cir, subckt_name=f"{subckt}_FOSTER")
        export_spice_test_foster(foster_net, tb_foster,
                                 subckt_name=f"{subckt}_FOSTER",
                                 subckt_file=foster_cir)
        txt = tb_foster.read_text(encoding="utf-8")
        txt = _re.sub(r"\.ac dec \d+ \S+ \S+", f".ac dec 100 {sweep_range}", txt)
        tb_foster.write_text(txt, encoding="utf-8")
        print(f"  {desc}")
        print(f"    Foster  : {foster_cir.name}  +  {tb_foster.name}")
    else:
        print(f"  {desc}")
        print(f"    (Foster skipped: RHP poles cannot be synthesised as passive RLC)")
    print(f"    Laplace : {beh_cir.name}  +  {tb_beh.name}")
