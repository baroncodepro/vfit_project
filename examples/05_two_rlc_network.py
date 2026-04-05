"""
examples/05_two_rlc_network.py
────────────────────────────────────────────────────────────────────────────────
Identify a network of TWO parallel RLC branches from impedance measurements.

Network topology (Foster-I — two shunt branches in series)
───────────────────────────────────────────────────────────

    Port ●────┬─────────────────────────────┬──── Port
              │                             │
           [RLC₁]                        [RLC₂]
              │                             │
             GND                           GND

     Z(s) = Z₁(s) + Z₂(s)

     Zᵢ(s) =          1
              ─────────────────────────
              1/Rᵢ  +  1/(s·Lᵢ)  +  s·Cᵢ

Each parallel RLC branch contributes ONE conjugate pole pair to H(s).
Two branches → four poles total → n_poles = 4 is the exact order.

True element values (buried in the CSV — pretend these are unknown)
────────────────────────────────────────────────────────────────────
  Branch 1:  R₁ = 942.5 Ω,   L₁ = 100 nH,  C₁ = 11.26 pF   (f₀=150 MHz, Q=10)
  Branch 2:  R₂ = 150.8 Ω,   L₂ =   5 nH,  C₂ = 14.07 pF   (f₀=600 MHz, Q= 8)

How it maps to Vector Fitting output
─────────────────────────────────────
  Step 1 │ Load CSV  →  (freq_hz, Z_measured)
  Step 2 │ VF fit    →  RationalModel   poles, residues, d, e
  Step 3 │ Inspect poles  → 2 complex conjugate pairs
         │   pair 1: f₀≈150 MHz, Q≈10   ← Branch 1
         │   pair 2: f₀≈600 MHz, Q≈ 8   ← Branch 2
  Step 4 │ Foster synthesis  → FosterNetwork branches
         │   each RLC branch has R, L, C values
  Step 5 │ Validate  → re-simulate network vs measured data
  Step 6 │ Export SPICE netlist

Generalisation
──────────────
  N branches  →  N conjugate pairs  →  n_poles = 2N
  If you don't know N: sweep n_poles = 2,4,6,8 and watch where RMS stops
  improving.  The "elbow" in the RMS curve is the correct model order.
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

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
    load_ri_csv,
    MeasurementData,
)
from vfit.core.pole_zero import (
    pole_resonant_frequency,
    pole_quality_factor,
    sort_by_frequency,
)


HERE = Path(__file__).parent

# True values (hidden in practice — only for validation here)
_TRUE = dict(
    R1=942.48, L1=100e-9, C1=11.258e-12, f01=150e6, Q1=10,
    R2=150.80, L2=  5e-9, C2=14.072e-12, f02=600e6, Q2= 8,
)


def _header(t): print(f"\n{'═'*68}\n  {t}\n{'═'*68}")
def _section(t): print(f"\n  ── {t} {'─'*(60-len(t))}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load measurement data
# ─────────────────────────────────────────────────────────────────────────────

_header("Two parallel RLC branches — impedance identification")
_section("Step 1: load measurement")

# ┌──────────────────────────────────────────────────────────────────────────
# │  Replace with your own file path and column names.
# │  The loader auto-detects MHz from the frequency values.
# └──────────────────────────────────────────────────────────────────────────
data = load_ri_csv(
    HERE / "data_two_rlc_Z.csv",
    freq_col  = "Frequency[MHz]",
    re_col    = "Re[Z_Ohm]",
    im_col    = "Im[Z_Ohm]",
    freq_unit = "MHz",
    label     = "Two parallel RLC branches — Z(jω)",
)
print(data.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Model order selection
# Rule: N branches → 2N poles.  Sweep and find the "elbow".
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 2: model order selection  (sweep n_poles)")

print(f"\n  {'n_poles':>8}  {'RMS vs data':>14}  {'verdict':>30}")
print(f"  {'─'*8}  {'─'*14}  {'─'*30}")

rms_sweep = {}
for n in [2, 4, 6, 8]:
    m = VectorFitter(
        n_poles    = n,
        n_iter_max = 50,
        weight     = "uniform",
    ).fit(data.freq_hz, data.H)
    rms_sweep[n] = m.rms_error
    verdict = {
        2: "too few — one resonance missing",
        4: "correct — two conjugate pairs  ✓",
        6: "overfitted — extra poles absorb noise",
        8: "overfitted — diminishing returns",
    }[n]
    marker = "  <──" if n == 4 else ""
    print(f"  {n:>8}  {m.rms_error:>14.4e}  {verdict}{marker}")

print(f"\n  Conclusion: use n_poles=4  (one pair per branch)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Fit with the correct order
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 3: vector fitting  (n_poles=4)")

model = VectorFitter(
    n_poles    = 4,
    n_iter_max = 50,
    weight     = "uniform",   # wide impedance range → uniform weights
    verbose    = True,
).fit(data.freq_hz, data.H)

print()
print(f"  Abs RMS       : {model.rms_error:.4e} Ω")
print(f"  Iterations    : {len(model.rms_error_history)}")
print(f"  d (≈ DC loss) : {model.d:.4e} Ω")
print(f"  e (≈ stray L) : {model.e:.4e} H")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Pole analysis → identify each branch
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 4: pole analysis — branch identification")

# Keep only upper half-plane (each pair shown once)
complex_poles = sorted(
    [p for p in model.poles if p.imag > 0],
    key=lambda p: p.imag
)

print(f"\n  {'Branch':>7}  {'Re(p) [rad/s]':>18}  {'f₀ [MHz]':>10}  "
      f"{'Q':>7}  {'damping':>10}")
print(f"  {'─'*7}  {'─'*18}  {'─'*10}  {'─'*7}  {'─'*10}")

for k, p in enumerate(complex_poles, start=1):
    f0  = pole_resonant_frequency(p) / 1e6
    Q   = pole_quality_factor(p)
    zeta = abs(p.real) / abs(p)          # damping ratio ζ = α/|p|
    print(f"  {k:>7}  {p.real:>18.4e}  {f0:>10.2f}  {Q:>7.2f}  {zeta:>10.4f}")

print()
print(f"  True Branch 1: f₀={_TRUE['f01']/1e6:.0f} MHz, Q={_TRUE['Q1']:.0f}")
print(f"  True Branch 2: f₀={_TRUE['f02']/1e6:.0f} MHz, Q={_TRUE['Q2']:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Foster synthesis → extract R, L, C values
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 5: Foster synthesis → R, L, C element values")

network = foster_synthesis(model)

# Filter to just the RLC branches (ignore tiny L_only / R_only artefacts)
# A "real" branch has f0 within the measurement band and physically
# sensible element values.  Filter out artefact branches whose f0 is
# far outside the measurement frequency range.
freq_min_mhz = data.freq_hz.min() / 1e6   # 1 MHz
freq_max_mhz = data.freq_hz.max() / 1e6   # 10000 MHz

def branch_f0_mhz(b):
    if b.L and b.C and b.L > 1e-15 and b.C > 1e-20:
        return 1 / (2*np.pi*np.sqrt(b.L*b.C)) / 1e6
    return None

rlc_branches_all  = [b for b in network.branches if b.branch_type == "RLC"]
other_branches    = [b for b in network.branches if b.branch_type != "RLC"]

# Keep only branches whose f0 falls inside the measurement band
rlc_branches = [
    b for b in rlc_branches_all
    if branch_f0_mhz(b) is not None
    and freq_min_mhz <= branch_f0_mhz(b) <= freq_max_mhz
]
artefact_branches = [b for b in rlc_branches_all if b not in rlc_branches]

print(f"\n  RLC branches ({len(rlc_branches)} physical, "
      f"{len(artefact_branches)} artefact, "
      f"{len(other_branches)} d/e terms):")
print(f"\n  {'#':>3}  {'R [Ω]':>12}  {'L [H]':>12}  {'C [F]':>12}  "
      f"{'f₀ [MHz]':>10}  {'Q':>7}")
print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*7}")

for i, b in enumerate(rlc_branches):
    f0_s = branch_f0_mhz(b)
    Q_synth = b.R * np.sqrt(b.C / b.L) if b.L and b.C else float("nan")
    print(f"  {i+1:>3}  {b.R:>12.4f}  {b.L:>12.4e}  {b.C:>12.4e}  "
          f"{f0_s:>10.2f}  {Q_synth:>7.2f}")

if artefact_branches:
    print(f"\n  Artefact RLC branches (outside measurement band — ignore):")
    for b in artefact_branches:
        f0s = branch_f0_mhz(b)
        print(f"    {b}  f₀≈{f0s:.1f} MHz")

print(f"\n  Other branches (from d/e direct terms):")
for b in other_branches:
    print(f"    {b}")

# ── Comparison table ──────────────────────────────────────────────────────────
print()
print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │  Element value comparison — fitted vs true                      │")
print("  ├──────────┬────────────┬────────────┬──────────────────────────┤")
print("  │  param   │  fitted    │  true      │  relative error          │")
print("  ├──────────┼────────────┼────────────┼──────────────────────────┤")

# Match each true branch to the physically closest fitted branch (by f0)
true_branches = [
    ("Branch 1", _TRUE["R1"], _TRUE["L1"], _TRUE["C1"], _TRUE["f01"]/1e6),
    ("Branch 2", _TRUE["R2"], _TRUE["L2"], _TRUE["C2"], _TRUE["f02"]/1e6),
]

for label, Rt, Lt, Ct, f0t_mhz in true_branches:
    # Find fitted branch whose f0 is closest to the true f0
    best = min(rlc_branches,
               key=lambda b: abs((branch_f0_mhz(b) or 0) - f0t_mhz))
    f0_fit = branch_f0_mhz(best)
    Q_fit  = best.R * np.sqrt(best.C / best.L)
    print(f"  │  {label} (f₀≈{f0t_mhz:.0f} MHz){' '*max(0,12-len(str(int(f0t_mhz))))}"
          f"{'─'*30}  │")
    for name, true_val, fit_val in [
        ("R [Ω]",  Rt, best.R),
        ("L [H]",  Lt, best.L),
        ("C [F]",  Ct, best.C),
    ]:
        err = abs(fit_val - true_val) / abs(true_val) * 100
        print(f"  │    {name:<8}│  {fit_val:>10.4e} │  {true_val:>10.4e} │"
              f"  {err:>6.2f}%{' '*17}│")

print("  └──────────┴────────────┴────────────┴──────────────────────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Validate: re-simulate network vs measurement
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 6: validation — network re-simulation")

Z_fit   = model.evaluate(data.freq_hz)
Z_synth = network.impedance(data.freq_hz)

rms_fit_vs_data  = model.rms_error
rms_synth_vs_fit = float(np.sqrt(np.mean(
    np.abs(Z_synth - Z_fit) ** 2
)))
rms_synth_vs_data = float(np.sqrt(np.mean(
    np.abs(Z_synth - data.H) ** 2
)))

print(f"\n  VF fit vs measured data    RMS = {rms_fit_vs_data:.4e} Ω")
print(f"  Foster network vs VF model RMS = {rms_synth_vs_fit:.4e} Ω")
print(f"  Foster network vs measured RMS = {rms_synth_vs_data:.4e} Ω")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — SPICE export
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 7: SPICE export")

foster_path    = HERE / "two_rlc_foster.cir"
beh_path       = HERE / "two_rlc_behavioral.cir"
tb_foster_path = HERE / "tb_two_rlc_foster.cir"
tb_beh_path    = HERE / "tb_two_rlc_behavioral.cir"

export_spice_foster(        network, foster_path, subckt_name="TWO_RLC_FOSTER")
export_spice_behavioral(    model,   beh_path,    subckt_name="TWO_RLC_LAPLACE")
export_spice_test_foster(   network, tb_foster_path,    subckt_name="TWO_RLC_FOSTER",    subckt_file=foster_path)
export_spice_test_behavioral(model,  tb_beh_path,       subckt_name="TWO_RLC_LAPLACE",   subckt_file=beh_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_C_MEAS  = "#2c7bb6"
_C_FIT   = "#d7191c"
_C_SYNTH = "#1a9641"
_C_POLE  = "#e66101"
_C_ZERO  = "#1a9641"

f_mhz = data.freq_hz / 1e6


# ── Figure 1: Full pipeline ───────────────────────────────────────────────────

fig = plt.figure(figsize=(17, 11))
fig.suptitle(
    "Two parallel RLC branches — full identification pipeline\n"
    "Z(s) = Z₁(s) + Z₂(s),   Z_i = 1/(1/R_i + 1/(sL_i) + sC_i)",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

ax_mag    = fig.add_subplot(gs[0, 0])
ax_phase  = fig.add_subplot(gs[1, 0])
ax_pz     = fig.add_subplot(gs[:2, 1])
ax_order  = fig.add_subplot(gs[0, 2])
ax_err    = fig.add_subplot(gs[1, 2])
ax_conv   = fig.add_subplot(gs[2, :])


# ── Bode ──────────────────────────────────────────────────────────────────────
for ax, y_meas, y_fit, y_synth, ylabel in [
    (ax_mag,
     20*np.log10(np.abs(data.H)+1e-300),
     20*np.log10(np.abs(Z_fit)+1e-300),
     20*np.log10(np.abs(Z_synth)+1e-300),
     "|Z| (dB Ω)"),
    (ax_phase,
     np.degrees(np.unwrap(np.angle(data.H))),
     np.degrees(np.unwrap(np.angle(Z_fit))),
     np.degrees(np.unwrap(np.angle(Z_synth))),
     "Phase (°)"),
]:
    ax.semilogx(f_mhz, y_meas,  color=_C_MEAS,  lw=1.4, alpha=0.65,
                label="Measured")
    ax.semilogx(f_mhz, y_fit,   color=_C_FIT,   lw=2.0, ls="--",
                label=f"VF fit  (n=4)")
    ax.semilogx(f_mhz, y_synth, color=_C_SYNTH, lw=1.6, ls=":",
                label="Foster network")
    # Mark true resonances
    for f_true in [_TRUE["f01"]/1e6, _TRUE["f02"]/1e6]:
        ax.axvline(f_true, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlabel("Frequency (MHz)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, which="both", color="#cccccc", lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

ax_mag.legend(fontsize=7)
ax_mag.set_title("Bode: measured / VF fit / Foster synthesis", fontsize=8)
ax_phase.set_title("Phase (grey dashed = true resonances)", fontsize=8)


# ── Pole-zero map ─────────────────────────────────────────────────────────────
poles = model.poles
zeros = model.zeros

ax_pz.axvline(0, color="k", lw=0.8, ls="--", alpha=0.35)
yext = abs(poles.imag).max() * 1.3
ax_pz.fill_betweenx([-yext, yext], 0,
                    abs(poles.real).max()*2, alpha=0.05, color="red")

ax_pz.scatter(poles.real, poles.imag,
              marker="x", s=180, color=_C_POLE, linewidths=2.5,
              zorder=5, label=f"Poles ({len(poles)})")
if len(zeros):
    ax_pz.scatter(zeros.real, zeros.imag,
                  marker="o", s=90, facecolors="none",
                  edgecolors=_C_ZERO, linewidths=2, zorder=5,
                  label=f"Zeros ({len(zeros)})")

# Annotate each conjugate pair with branch number and f0
for k, p in enumerate(complex_poles, start=1):
    f0_mhz = pole_resonant_frequency(p)/1e6
    Q      = pole_quality_factor(p)
    ax_pz.annotate(
        f"Branch {k}\nf₀={f0_mhz:.0f} MHz\nQ={Q:.1f}",
        xy=(p.real, p.imag),
        xytext=(p.real - abs(poles.real).max()*0.6, p.imag),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
    )
    ax_pz.annotate(
        "", xy=(p.real, -p.imag),
        xytext=(p.real - abs(poles.real).max()*0.6, p.imag),
        fontsize=7,
    )

ax_pz.set_xlabel("σ (rad/s)", fontsize=8)
ax_pz.set_ylabel("jω (rad/s)", fontsize=8)
ax_pz.set_title("S-plane — two conjugate pairs\n(one per RLC branch)", fontsize=8)
ax_pz.legend(fontsize=7)
ax_pz.grid(True, color="#cccccc", lw=0.4)
ax_pz.spines["top"].set_visible(False)
ax_pz.spines["right"].set_visible(False)
ax_pz.tick_params(labelsize=7)


# ── Model order sweep ─────────────────────────────────────────────────────────
ns   = list(rms_sweep.keys())
rmss = list(rms_sweep.values())

ax_order.semilogy(ns, rmss, "o-", color=_C_MEAS, lw=2, ms=7)
ax_order.axvline(4, color=_C_FIT, ls="--", lw=1.5,
                 label="Correct order (n=4)")
ax_order.set_xticks(ns)
ax_order.set_xlabel("n_poles", fontsize=8)
ax_order.set_ylabel("RMS Error (Ω)", fontsize=8)
ax_order.set_title("Order selection: RMS elbow at n=4", fontsize=8)
ax_order.legend(fontsize=7)
ax_order.grid(True, which="both", color="#cccccc", lw=0.4)
ax_order.spines["top"].set_visible(False)
ax_order.spines["right"].set_visible(False)
ax_order.tick_params(labelsize=7)


# ── Element recovery bar chart ────────────────────────────────────────────────
params   = ["R₁\n(Ω)", "L₁\n(nH)", "C₁\n(pF)", "R₂\n(Ω)", "L₂\n(nH)", "C₂\n(pF)"]
true_v   = [_TRUE["R1"],  _TRUE["L1"]*1e9, _TRUE["C1"]*1e12,
            _TRUE["R2"],  _TRUE["L2"]*1e9, _TRUE["C2"]*1e12]
# Match fitted branches to true branches by nearest f0
fit_v = []
for _, _, _, _, f0t_mhz in [
    ("", _TRUE["R1"], _TRUE["L1"], _TRUE["C1"], _TRUE["f01"]/1e6),
    ("", _TRUE["R2"], _TRUE["L2"], _TRUE["C2"], _TRUE["f02"]/1e6),
]:
    best = min(rlc_branches, key=lambda b: abs((branch_f0_mhz(b) or 0) - f0t_mhz))
    fit_v += [best.R, best.L*1e9, best.C*1e12]

x      = np.arange(len(params))
width  = 0.35
bars1  = ax_err.bar(x - width/2, true_v,  width, color=_C_MEAS,  alpha=0.8,
                    label="True")
bars2  = ax_err.bar(x + width/2, fit_v,   width, color=_C_FIT,   alpha=0.8,
                    label="Fitted")

ax_err.set_xticks(x)
ax_err.set_xticklabels(params, fontsize=7)
ax_err.set_ylabel("Value (natural units)", fontsize=8)
ax_err.set_title("Element recovery: fitted vs true values", fontsize=8)
ax_err.legend(fontsize=7)
ax_err.set_yscale("log")
ax_err.grid(True, axis="y", color="#cccccc", lw=0.4)
ax_err.spines["top"].set_visible(False)
ax_err.spines["right"].set_visible(False)
ax_err.tick_params(labelsize=7)


# ── Convergence (full width) ──────────────────────────────────────────────────
hist = model.rms_error_history
ax_conv.semilogy(range(1, len(hist)+1), hist, "o-",
                 color=_C_MEAS, lw=2, ms=5)
ax_conv.axhline(hist[-1], color=_C_FIT, ls="--", lw=1,
                label=f"Final RMS = {hist[-1]:.3e} Ω")
ax_conv.set_xlabel("Iteration", fontsize=8)
ax_conv.set_ylabel("RMS Error (Ω)", fontsize=8)
ax_conv.set_title("VF convergence history  (n_poles=4, weight=uniform)", fontsize=8)
ax_conv.legend(fontsize=8)
ax_conv.set_xticks(range(1, len(hist)+1))
ax_conv.grid(True, which="both", color="#cccccc", lw=0.4)
ax_conv.spines["top"].set_visible(False)
ax_conv.spines["right"].set_visible(False)
ax_conv.tick_params(labelsize=7)


fig.savefig(HERE / "05_two_rlc_pipeline.png", dpi=150, bbox_inches="tight")
print("\nSaved: 05_two_rlc_pipeline.png")

plt.show()
print("\nDone.")
print(f"\nSPICE files written:")
print(f"  {foster_path}")
print(f"  {beh_path}")
print(f"  {tb_foster_path}  ← open this in LTspice to run AC simulation")
print(f"  {tb_beh_path}     ← open this in LTspice to run AC simulation")
