"""
examples/01_simple_rlc.py
─────────────────────────
Fit a simple 2nd-order RLC series impedance.

  Z(s) = R + s*L + 1/(s*C)

Known poles:  p₁,₂ = -R/(2L) ± j*sqrt(1/(LC) - (R/2L)²)
"""

import numpy as np
import matplotlib.pyplot as plt
# from vfit import VectorFitter   # uncomment once Phase 1 is complete

# ── Circuit parameters ────────────────────────────────────────────────────────
R = 10.0        # Ohm
L = 1e-6        # Henry   (1 µH)
C = 1e-12       # Farad   (1 pF)

# ── Frequency sweep ───────────────────────────────────────────────────────────
freq = np.logspace(6, 10, 300)   # 1 MHz → 10 GHz
omega = 2 * np.pi * freq
s = 1j * omega

# ── Analytical impedance ──────────────────────────────────────────────────────
Z_meas = R + s * L + 1.0 / (s * C)

# ── Analytical poles ──────────────────────────────────────────────────────────
omega_0 = 1 / np.sqrt(L * C)
alpha   = R / (2 * L)
omega_d = np.sqrt(omega_0**2 - alpha**2)

p1 = -alpha + 1j * omega_d
p2 = -alpha - 1j * omega_d
f0 = omega_0 / (2 * np.pi)

print(f"Resonant frequency : f₀ = {f0/1e9:.3f} GHz")
print(f"Damping coefficient: α  = {alpha:.3e} rad/s")
print(f"Pole 1             : p₁ = {p1:.4e}")
print(f"Pole 2             : p₂ = {p2:.4e}")

# ── Plot (data only — VF fit to be added in Phase 1) ─────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.semilogx(freq / 1e9, 20 * np.log10(np.abs(Z_meas)))
ax1.set_ylabel("|Z| (dB Ω)")
ax1.set_title("RLC Series Impedance — Measured Data")
ax1.grid(True, which="both", alpha=0.4)

ax2.semilogx(freq / 1e9, np.degrees(np.angle(Z_meas)))
ax2.set_ylabel("Phase (°)")
ax2.set_xlabel("Frequency (GHz)")
ax2.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.savefig("rlc_impedance.png", dpi=150)
plt.show()
print("Saved: rlc_impedance.png")
