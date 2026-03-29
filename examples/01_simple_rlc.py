"""
examples/01_simple_rlc.py
─────────────────────────
Fit a simple 2nd-order series RLC impedance with Vector Fitting.

  Z(s) = R + s*L + 1/(s*C)

Known poles:  p₁,₂ = -R/(2L) ± j*sqrt(1/(LC) - (R/2L)²)

Pipeline
────────
  1. Generate analytical Z(jω) over 1 MHz → 10 GHz
  2. Fit rational model with VectorFitter (n_poles=4)
  3. Foster RLC synthesis → element values
  4. Export SPICE netlist for LTspice verification
  5. Plot: measured vs VF fit vs Foster synthesis

Verify in LTspice
─────────────────
  Open  examples/tb_simple_rlc_foster.cir    (or tb_simple_rlc_behavioral.cir)
  Run   .ac sweep → plot V(1)/I(Vin) or V(out)
  Then: python plot_ltspice_bode.py --preset simple_rlc
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vfit import (
    VectorFitter,
    foster_synthesis,
    export_spice_foster,
    export_spice_behavioral,
    export_spice_test_foster,
    export_spice_test_behavioral,
)

HERE = Path(__file__).parent

# ── Circuit parameters ─────────────────────────────────────────────────────────
R = 10.0        # Ohm
L = 1e-6        # Henry   (1 µH)
C = 1e-12       # Farad   (1 pF)

# ── Frequency sweep ────────────────────────────────────────────────────────────
freq  = np.logspace(6, 10, 300)   # 1 MHz → 10 GHz
omega = 2 * np.pi * freq
s     = 1j * omega

# ── Analytical impedance ───────────────────────────────────────────────────────
Z_meas = R + s * L + 1.0 / (s * C)

# ── Analytical poles ───────────────────────────────────────────────────────────
omega_0 = 1 / np.sqrt(L * C)
alpha   = R / (2 * L)
omega_d = np.sqrt(omega_0**2 - alpha**2)

p1 = -alpha + 1j * omega_d
p2 = -alpha - 1j * omega_d
f0 = omega_0 / (2 * np.pi)

print(f"Resonant frequency : f0 = {f0/1e9:.3f} GHz")
print(f"Damping coefficient: a  = {alpha:.3e} rad/s")
print(f"Pole 1             : p1 = {p1:.4e}")
print(f"Pole 2             : p2 = {p2:.4e}")

# ── Vector Fitting ─────────────────────────────────────────────────────────────
# n_poles=4: 2 for the series resonance pair + 2 for the near-DC capacitive tail
model = VectorFitter(n_poles=4, n_iter_max=100, weight="uniform").fit(freq, Z_meas)

Z_fit   = model.evaluate(freq)
rms_rel = float(np.sqrt(np.mean(np.abs((Z_fit - Z_meas) / Z_meas) ** 2)))
print(f"\nVF fit:  n_poles={model.n_poles}  RMS={model.rms_error:.3e}  "
      f"rel_RMS={rms_rel:.3e}  iters={len(model.rms_error_history)}")
print(f"Fitted poles: {model.poles}")

# ── Foster RLC synthesis ───────────────────────────────────────────────────────
network  = foster_synthesis(model)
Z_synth  = network.impedance(freq)
rms_synth = float(np.sqrt(np.mean(np.abs((Z_synth - Z_meas) / Z_meas) ** 2)))
print(f"Synthesis round-trip rel_RMS = {rms_synth:.3e}")

# ── SPICE export ───────────────────────────────────────────────────────────────
foster_cir = HERE / "simple_rlc_foster.cir"
beh_cir    = HERE / "simple_rlc_behavioral.cir"
tb_foster  = HERE / "tb_simple_rlc_foster.cir"
tb_beh     = HERE / "tb_simple_rlc_behavioral.cir"

export_spice_foster(network, foster_cir,  subckt_name="SIMPLE_RLC_FOSTER")
export_spice_behavioral(model, beh_cir,   subckt_name="SIMPLE_RLC_LAPLACE")
export_spice_test_foster(network, tb_foster,
                         subckt_name="SIMPLE_RLC_FOSTER",  subckt_file=foster_cir)
export_spice_test_behavioral(model, tb_beh,
                             subckt_name="SIMPLE_RLC_LAPLACE", subckt_file=beh_cir)
print(f"\nSPICE files: {foster_cir.name}, {beh_cir.name}, "
      f"{tb_foster.name}, {tb_beh.name}")

# ── Plot ───────────────────────────────────────────────────────────────────────
f_ghz = freq / 1e9

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
fig.suptitle(
    f"Series RLC Impedance — Z(s)=R+sL+1/(sC)   "
    f"R={R}Ω  L={L*1e6:.0f}µH  C={C*1e12:.0f}pF  f0={f0/1e9:.3f} GHz",
    fontsize=10, fontweight="bold",
)

ax1.semilogx(f_ghz, 20*np.log10(np.abs(Z_meas)),
             color="#2c7bb6", lw=2.0, label="Analytical Z(s)")
ax1.semilogx(f_ghz, 20*np.log10(np.abs(Z_fit)+1e-300),
             color="#d7191c", lw=1.8, ls="--",
             label=f"VF fit (n={model.n_poles}, RMS={model.rms_error:.1e})")
ax1.semilogx(f_ghz, 20*np.log10(np.abs(Z_synth)+1e-300),
             color="#1a9641", lw=1.5, ls=":",
             label="Foster synthesis")
ax1.set_ylabel("|Z| (dB Ohm)")
ax1.legend(fontsize=8)
ax1.grid(True, which="both", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.semilogx(f_ghz, np.degrees(np.unwrap(np.angle(Z_meas))),
             color="#2c7bb6", lw=2.0)
ax2.semilogx(f_ghz, np.degrees(np.unwrap(np.angle(Z_fit))),
             color="#d7191c", lw=1.8, ls="--")
ax2.semilogx(f_ghz, np.degrees(np.unwrap(np.angle(Z_synth))),
             color="#1a9641", lw=1.5, ls=":")
ax2.set_ylabel("Phase (deg)")
ax2.set_xlabel("Frequency (GHz)")
ax2.grid(True, which="both", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
out_png = HERE / "01_simple_rlc.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved: {out_png.name}")
