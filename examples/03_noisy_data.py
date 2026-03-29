"""
examples/03_noisy_data.py
─────────────────────────
Vector Fitting on noisy frequency-domain data.

Real measurements always contain noise.  This example shows what changes
when you add additive complex Gaussian noise:

    H_noisy(jω) = H_true(jω) + ε(ω),
    ε(ω) ~ noise_amp · (N(0,1) + j·N(0,1))

Three lessons covered
──────────────────────
  Lesson 1 — Low-SNR data / real poles (Case 1)
    H₁(s) = s / ((s+1)(s+3))
    mean|H| ≈ 0.09 → SNR ≈ 25 dB
    Observation: poles drift; adding extra poles absorbs noise instead of
    recovering the true system.  The noise floor sets a hard RMS ceiling.

  Lesson 2 — Weighting strategy matters for resonators (Case 2)
    H₂(s) = 2s / (s² + 0.4s + 4)   [Q=5]
    The default inverse weight (1/|H|) over-weights low-|H| frequencies
    where SNR is worst.  Switching to uniform weight fixes this cleanly.

  Lesson 3 — RHP pole sensitivity (Case 3)
    H₃(s) = s / ((s−1)(s−2))   [RHP poles]
    Noise shifts the estimated pole locations away from the true values.
    The further a pole is from the imaginary axis, the harder it is to
    pin down from jω-axis samples.

Key rule of thumb
─────────────────
  The best RMS a fit can ever achieve is the noise floor itself:

      RMS_floor = noise_amp · sqrt(2)   (complex noise, 2 components)

  Any fit with RMS_fit ≈ RMS_floor is as good as the data allows.
  Adding more poles beyond that point fits the noise, not the system.
"""

from __future__ import annotations

import sys
import io
import os
import warnings

# Ensure Unicode output works on Windows terminals with non-UTF-8 code pages
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import re as _re
from pathlib import Path

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


# ── Reproducible noise ────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

NOISE_AMP = 0.005

def add_noise(H: np.ndarray) -> np.ndarray:
    """Add complex Gaussian noise: ε ~ noise_amp · (N(0,1) + j·N(0,1))."""
    return H + NOISE_AMP * (rng.standard_normal(H.shape)
                            + 1j * rng.standard_normal(H.shape))


def noise_floor_rms(n: int) -> float:
    """
    Expected RMS of complex Gaussian noise with amplitude `noise_amp`.
    Each sample has variance noise_amp² in Re and noise_amp² in Im,
    so E[|ε|²] = 2 · noise_amp²  →  RMS_floor = noise_amp · sqrt(2).
    """
    return NOISE_AMP * np.sqrt(2.0)


def _header(title: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# True transfer functions
# ─────────────────────────────────────────────────────────────────────────────

freq1 = np.logspace(-2,  2, 500)
s1    = 1j * 2 * np.pi * freq1
H1    = s1 / ((s1 + 1) * (s1 + 3))

omega_0, alpha_c = 2.0, 0.2
omega_d = np.sqrt(omega_0**2 - alpha_c**2)
freq2 = np.logspace(-2, 1, 600)
s2    = 1j * 2 * np.pi * freq2
H2    = 2 * s2 / (s2**2 + 2 * alpha_c * s2 + omega_0**2)

freq3 = np.logspace(-1, 2, 500)
s3    = 1j * 2 * np.pi * freq3
H3    = s3 / (s3**2 - 3 * s3 + 2)

# Add noise
H1n = add_noise(H1)
H2n = add_noise(H2)
H3n = add_noise(H3)

RMS_FLOOR = noise_floor_rms(len(H1))


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 1 — Low-SNR, real poles
# ─────────────────────────────────────────────────────────────────────────────

_header("Lesson 1 — H(s) = s/((s+1)(s+3))   low SNR, real poles")

snr1_db = 20 * np.log10(np.mean(np.abs(H1)) / NOISE_AMP)
print(f"  noise_amp   = {NOISE_AMP}")
print(f"  mean|H₁|    = {np.mean(np.abs(H1)):.4f}")
print(f"  mean SNR    = {snr1_db:.1f} dB")
print(f"  noise floor = {RMS_FLOOR:.4e}  (best possible RMS)")
print()
print(f"  {'n_poles':>8}  {'fitted poles':>30}  {'RMS vs noisy':>14}  {'RMS vs true':>12}")
print(f"  {'─'*8}  {'─'*30}  {'─'*14}  {'─'*12}")

models1 = {}
for n in [2, 4, 6]:
    m = VectorFitter(n_poles=n, n_iter_max=50).fit(freq1, H1n)
    rms_noisy = m.rms_error
    rms_true  = float(np.sqrt(np.mean(np.abs(m.evaluate(freq1) - H1) ** 2)))
    poles_str = str(np.sort(m.poles.real).round(2))
    print(f"  {n:>8}  {poles_str:>30}  {rms_noisy:>14.3e}  {rms_true:>12.3e}")
    models1[n] = m

print()
print(f"  True poles : -1.0, -3.0")
print(f"  Note: n_poles=2 should match best — extra poles fit noise, not signal.")


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 2 — Weighting strategy for resonator
# ─────────────────────────────────────────────────────────────────────────────

_header("Lesson 2 — H(s) = 2s/(s²+0.4s+4)   weight strategy [Q=5]")

snr2_peak_db = 20 * np.log10(abs(H2).max() / NOISE_AMP)
snr2_low_db  = 20 * np.log10(abs(H2[0])    / NOISE_AMP)
print(f"  noise_amp        = {NOISE_AMP}")
print(f"  peak SNR (f=f₀)  = {snr2_peak_db:.1f} dB   (strong signal)")
print(f"  SNR at f=0.01 Hz = {snr2_low_db:.1f} dB   (noise-dominated)")
print(f"  noise floor      = {RMS_FLOOR:.4e}")
print()
print(f"  {'weight':>10}  {'f₀ (Hz)':>10}  {'Q':>6}  {'RMS vs true':>12}  {'iters':>6}")
print(f"  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*12}  {'─'*6}")

models2 = {}
for weight in ["inverse", "uniform"]:
    m = VectorFitter(n_poles=2, n_iter_max=50, weight=weight).fit(freq2, H2n)
    p_up = [p for p in m.poles if p.imag > 0.5]
    if p_up:
        p    = min(p_up, key=lambda p: abs(p.imag - omega_d))
        f0   = pole_resonant_frequency(p)
        Q    = pole_quality_factor(p)
        rms  = float(np.sqrt(np.mean(np.abs(m.evaluate(freq2) - H2) ** 2)))
        iters = len(m.rms_error_history)
        print(f"  {weight:>10}  {f0:>10.4f}  {Q:>6.2f}  {rms:>12.3e}  {iters:>6}")
    models2[weight] = m

print()
print(f"  True: f₀ = 0.3183 Hz,  Q = 5.00")
print()
print("  Why inverse weight fails here:")
print("  1/|H| gives huge weight to low-f points where |H|≈0.03 and SNR=16 dB.")
print("  Those noisy points pull the poles away from the true resonance.")
print("  Uniform weight treats every frequency equally — correct for flat SNR.")
print()
print("  Rule of thumb: use weight='uniform' when SNR varies by > 20 dB across band.")


# ─────────────────────────────────────────────────────────────────────────────
# Lesson 3 — RHP pole sensitivity under noise
# ─────────────────────────────────────────────────────────────────────────────

_header("Lesson 3 — H(s) = s/((s−1)(s−2))   RHP poles under noise")

snr3_db = 20 * np.log10(np.mean(np.abs(H3)) / NOISE_AMP)
print(f"  noise_amp  = {NOISE_AMP}")
print(f"  mean SNR   = {snr3_db:.1f} dB")
print(f"  True poles : +1.0, +2.0")
print()
print(f"  {'n_poles':>8}  {'fitted poles':>32}  {'RMS vs true':>12}")
print(f"  {'─'*8}  {'─'*32}  {'─'*12}")

models3 = {}
for n in [2, 4]:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = VectorFitter(
            n_poles=n, n_iter_max=50, enforce_stability=False
        ).fit(freq3, H3n)
    rms = float(np.sqrt(np.mean(np.abs(m.evaluate(freq3) - H3) ** 2)))
    poles_str = str(np.sort(m.poles.real).round(3))
    print(f"  {n:>8}  {poles_str:>32}  {rms:>12.3e}")
    models3[n] = m

print()
print("  Observation: the pole near +2 (close to jω axis) is well recovered.")
print("  The pole near +1 (further from jω axis, weaker imprint on H(jω)) drifts more.")
print("  Physical insight: RHP poles far from the jω axis are hard to identify")
print("  from jω-axis samples because their contribution decays as |Re(p)| grows.")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_C_TRUE   = "#444444"
_C_NOISY  = "#aaaaaa"
_C_INV    = "#d7191c"   # inverse weight / n=2
_C_UNI    = "#1a9641"   # uniform weight
_C_OVER   = "#e66101"   # over-fitted (n=4,6)


# ── Figure 1: Lesson 1 — effect of pole count on noisy fit ───────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle(
    f"Lesson 1 — Low-SNR data (noise_amp={NOISE_AMP})\n"
    "H(s) = s / ((s+1)(s+3))   [true poles: −1, −3]",
    fontsize=11, fontweight="bold"
)

ax_mag, ax_err = axes1

# Magnitude
ax_mag.semilogx(freq1, 20 * np.log10(np.abs(H1n) + 1e-300),
                color=_C_NOISY, lw=1.0, alpha=0.7, label="Noisy data")
ax_mag.semilogx(freq1, 20 * np.log10(np.abs(H1)  + 1e-300),
                color=_C_TRUE,  lw=2.0, ls=":",   label="True H(s)")

colors_n = {2: _C_INV, 4: _C_UNI, 6: _C_OVER}
for n, m in models1.items():
    H_fit = m.evaluate(freq1)
    ax_mag.semilogx(freq1, 20 * np.log10(np.abs(H_fit) + 1e-300),
                    color=colors_n[n], lw=1.8, ls="--",
                    label=f"n_poles={n}")

ax_mag.set_xlabel("Frequency (Hz)", fontsize=9)
ax_mag.set_ylabel("Magnitude (dB)", fontsize=9)
ax_mag.set_title("Bode magnitude", fontsize=9)
ax_mag.legend(fontsize=8)
ax_mag.grid(True, which="both", color="#cccccc", lw=0.5)
ax_mag.spines["top"].set_visible(False)
ax_mag.spines["right"].set_visible(False)

# Absolute error vs true H
for n, m in models1.items():
    err = np.abs(m.evaluate(freq1) - H1)
    ax_err.semilogx(freq1, err, color=colors_n[n], lw=1.6, label=f"n_poles={n}")

ax_err.axhline(RMS_FLOOR, color="k", ls=":", lw=1.5,
               label=f"Noise floor ({RMS_FLOOR:.3f})")
ax_err.set_xlabel("Frequency (Hz)", fontsize=9)
ax_err.set_ylabel("|H_fit − H_true|", fontsize=9)
ax_err.set_title("Pointwise error vs true system", fontsize=9)
ax_err.legend(fontsize=8)
ax_err.grid(True, which="both", color="#cccccc", lw=0.5)
ax_err.spines["top"].set_visible(False)
ax_err.spines["right"].set_visible(False)

fig1.tight_layout()
fig1.savefig("03_noisy_lesson1.png", dpi=150)
print("\nSaved: 03_noisy_lesson1.png")


# ── Figure 2: Lesson 2 — inverse vs uniform weight ───────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle(
    f"Lesson 2 — Weight strategy (noise_amp={NOISE_AMP})\n"
    "H(s) = 2s / (s²+0.4s+4)   [Q=5, f₀=0.318 Hz]",
    fontsize=11, fontweight="bold"
)

ax_bode2, ax_snr = axes2

# Bode
ax_bode2.semilogx(freq2, 20 * np.log10(np.abs(H2n) + 1e-300),
                  color=_C_NOISY, lw=1.0, alpha=0.7, label="Noisy data")
ax_bode2.semilogx(freq2, 20 * np.log10(np.abs(H2)  + 1e-300),
                  color=_C_TRUE,  lw=2.0, ls=":",   label="True H(s)")

for weight, color, label in [
    ("inverse", _C_INV, "inverse weight  1/|H|"),
    ("uniform", _C_UNI, "uniform weight"),
]:
    H_fit = models2[weight].evaluate(freq2)
    ax_bode2.semilogx(freq2, 20 * np.log10(np.abs(H_fit) + 1e-300),
                      color=color, lw=2.0, ls="--", label=label)

ax_bode2.axvline(omega_0 / (2 * np.pi), color="#888", ls=":", lw=1,
                 label=f"f₀ = {omega_0/(2*np.pi):.3f} Hz")
ax_bode2.set_xlabel("Frequency (Hz)", fontsize=9)
ax_bode2.set_ylabel("Magnitude (dB)", fontsize=9)
ax_bode2.set_title("Bode magnitude — weighting comparison", fontsize=9)
ax_bode2.legend(fontsize=8)
ax_bode2.grid(True, which="both", color="#cccccc", lw=0.5)
ax_bode2.spines["top"].set_visible(False)
ax_bode2.spines["right"].set_visible(False)

# Local SNR vs frequency
local_snr = 20 * np.log10(np.abs(H2) / NOISE_AMP)
ax_snr.semilogx(freq2, local_snr, color=_C_TRUE, lw=2.0, label="Local SNR (dB)")
ax_snr.axhline(0,  color="k",    lw=0.8, ls=":", label="0 dB  (SNR=1)")
ax_snr.axhline(20, color="#888", lw=0.8, ls="--", label="20 dB threshold")
ax_snr.fill_between(freq2, local_snr, 0,
                    where=local_snr < 20,
                    alpha=0.15, color="red",
                    label="Low-SNR region")
ax_snr.set_xlabel("Frequency (Hz)", fontsize=9)
ax_snr.set_ylabel("Local SNR (dB)", fontsize=9)
ax_snr.set_title("Why inverse weight struggles: SNR varies 45 dB across band", fontsize=9)
ax_snr.legend(fontsize=8)
ax_snr.grid(True, which="both", color="#cccccc", lw=0.5)
ax_snr.spines["top"].set_visible(False)
ax_snr.spines["right"].set_visible(False)

fig2.tight_layout()
fig2.savefig("03_noisy_lesson2.png", dpi=150)
print("Saved: 03_noisy_lesson2.png")


# ── Figure 3: Lesson 3 — RHP pole drift under noise ──────────────────────────

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    f"Lesson 3 — RHP pole identification under noise (noise_amp={NOISE_AMP})\n"
    "H(s) = s / ((s−1)(s−2))   [true poles: +1, +2]",
    fontsize=11, fontweight="bold"
)

ax_bode3, ax_pz = axes3

# Bode
ax_bode3.semilogx(freq3, 20 * np.log10(np.abs(H3n) + 1e-300),
                  color=_C_NOISY, lw=1.0, alpha=0.7, label="Noisy data")
ax_bode3.semilogx(freq3, 20 * np.log10(np.abs(H3)  + 1e-300),
                  color=_C_TRUE,  lw=2.0, ls=":",   label="True H(s)")

colors3 = {2: _C_INV, 4: _C_OVER}
for n, m in models3.items():
    H_fit = m.evaluate(freq3)
    ax_bode3.semilogx(freq3, 20 * np.log10(np.abs(H_fit) + 1e-300),
                      color=colors3[n], lw=1.8, ls="--",
                      label=f"n_poles={n}")

ax_bode3.set_xlabel("Frequency (Hz)", fontsize=9)
ax_bode3.set_ylabel("Magnitude (dB)", fontsize=9)
ax_bode3.set_title("Bode magnitude", fontsize=9)
ax_bode3.legend(fontsize=8)
ax_bode3.grid(True, which="both", color="#cccccc", lw=0.5)
ax_bode3.spines["top"].set_visible(False)
ax_bode3.spines["right"].set_visible(False)

# S-plane: true vs fitted poles
ax_pz.axvline(0, color="k", lw=1.0, ls="--", alpha=0.4)
ax_pz.fill_betweenx([-1, 1], 0, 3, alpha=0.06, color="red")

# True poles
ax_pz.scatter([1, 2], [0, 0], marker="x", s=250, color=_C_TRUE,
              linewidths=3, zorder=6, label="True poles")

# Fitted poles (n_poles=2)
fitted2 = np.sort(models3[2].poles.real)
ax_pz.scatter(fitted2, np.zeros_like(fitted2),
              marker="x", s=140, color=_C_INV, linewidths=2,
              zorder=5, label=f"Fitted n=2  {np.round(fitted2,3)}")

# Fitted poles (n_poles=4)
fitted4 = np.sort(models3[4].poles.real)
ax_pz.scatter(fitted4, np.zeros_like(fitted4),
              marker="x", s=140, color=_C_OVER, linewidths=2,
              zorder=5, label=f"Fitted n=4  {np.round(fitted4,3)}")

# Annotations
ax_pz.annotate("Pole near +2\n(well recovered)", xy=(1.95, 0), xytext=(1.6, 0.4),
               fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))
ax_pz.annotate("Pole near +1\n(drifts more)", xy=(1.0, 0), xytext=(0.2, -0.5),
               fontsize=8, arrowprops=dict(arrowstyle="->", color="#555"))

ax_pz.set_xlim(-0.5, 2.5)
ax_pz.set_ylim(-0.8, 0.8)
ax_pz.set_xlabel("σ  (rad/s)", fontsize=9)
ax_pz.set_ylabel("jω  (rad/s)", fontsize=9)
ax_pz.set_title("S-plane: true vs fitted poles (noise shifts +1 more than +2)", fontsize=9)
ax_pz.legend(fontsize=8, loc="upper left")
ax_pz.grid(True, color="#cccccc", lw=0.5)
ax_pz.spines["top"].set_visible(False)
ax_pz.spines["right"].set_visible(False)

fig3.tight_layout()
fig3.savefig("03_noisy_lesson3.png", dpi=150)
print("Saved: 03_noisy_lesson3.png")

plt.show()
print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# SPICE export — best model per lesson
# ─────────────────────────────────────────────────────────────────────────────
# Lesson 1: n_poles=2 is the correct order (2 real LHP poles)
# Lesson 2: uniform weight gives the best fit for the Q=5 resonator
# Lesson 3: RHP poles -> behavioral-only (Foster synthesis requires stable poles)
#
# Verify with:
#   python plot_ltspice_bode.py --preset noisy1
#   python plot_ltspice_bode.py --preset noisy2
#   python plot_ltspice_bode.py --preset noisy3

print("\n── SPICE export ────────────────────────────────────")

_exports = [
    # (stem, model, subckt, sweep Hz range, has_stable_poles)
    ("noisy1", models1[2],         "NOISY1", "0.01 100", True),
    ("noisy2", models2["uniform"], "NOISY2", "0.01 10",  True),
    ("noisy3", models3[2],         "NOISY3", "0.1  100", False),
]

for stem, model, subckt, sweep_range, stable in _exports:
    beh_cir = HERE / f"{stem}_behavioral.cir"
    tb_beh  = HERE / f"tb_{stem}_behavioral.cir"
    export_spice_behavioral(model, beh_cir, subckt_name=f"{subckt}_LAPLACE")
    export_spice_test_behavioral(model, tb_beh,
                                 subckt_name=f"{subckt}_LAPLACE",
                                 subckt_file=beh_cir)
    txt = tb_beh.read_text(encoding="utf-8")
    txt = _re.sub(r"\.ac dec \d+ \S+ \S+", f".ac dec 100 {sweep_range}", txt)
    tb_beh.write_text(txt, encoding="utf-8")

    if stable:
        foster_net = foster_synthesis(model)
        foster_cir = HERE / f"{stem}_foster.cir"
        tb_foster  = HERE / f"tb_{stem}_foster.cir"
        export_spice_foster(foster_net, foster_cir, subckt_name=f"{subckt}_FOSTER")
        export_spice_test_foster(foster_net, tb_foster,
                                 subckt_name=f"{subckt}_FOSTER",
                                 subckt_file=foster_cir)
        txt = tb_foster.read_text(encoding="utf-8")
        txt = _re.sub(r"\.ac dec \d+ \S+ \S+", f".ac dec 100 {sweep_range}", txt)
        tb_foster.write_text(txt, encoding="utf-8")
        print(f"  {stem}: Foster + Behavioral  (sweep {sweep_range} Hz)")
    else:
        print(f"  {stem}: Behavioral only  (RHP poles, sweep {sweep_range} Hz)")
