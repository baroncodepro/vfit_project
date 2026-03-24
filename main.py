"""
main.py
=======
Interactive Vector Fitting Analysis Tool

Run this file to start the analysis:

    python main.py

You will be guided through three analysis modes:

  1. Load from a measurement file  (CSV or Touchstone .s1p/.s2p)
  2. Define an S-domain transfer function  (poles/zeros or polynomial)
  3. Define an RLC circuit  (specify component values interactively)

All modes produce:
  - Bode plot (magnitude and phase)
  - Pole-zero map (S-plane)
  - Convergence history
  - Foster RLC synthesis table
  - Optional SPICE netlist export (.cir files)
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running directly from the project root without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vfit import (
    VectorFitter,
    foster_synthesis,
    export_spice_foster,
    export_spice_behavioral,
    load_csv,
    load_ri_csv,
    load_touchstone,
    MeasurementData,
)
from vfit.core.pole_zero import (
    pole_resonant_frequency,
    pole_quality_factor,
    sort_by_frequency,
)
from vfit.visualization import bode_plot, pole_zero_map, convergence_plot


# =============================================================================
# Console helpers
# =============================================================================

def _banner() -> None:
    print()
    print("=" * 68)
    print("   VFIT  -  Vector Fitting Analysis Tool")
    print("   Fit rational transfer functions to frequency-domain data")
    print("=" * 68)
    print()


def _section(title: str) -> None:
    print()
    print(f"  --- {title} " + "-" * max(1, 54 - len(title)))


def _ask(prompt: str, default: str = "") -> str:
    """Print a prompt and return the user's answer (stripped).
    Returns default if the user presses Enter without typing."""
    tag = f"  [{default}]" if default else ""
    try:
        answer = input(f"\n  {prompt}{tag}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Interrupted. Goodbye.")
        sys.exit(0)
    return answer if answer else default


def _ask_float(prompt: str, default: float) -> float:
    """Ask for a floating-point value, re-prompt on bad input."""
    while True:
        raw = _ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print(f"    Please enter a number (example: {default}).")


def _ask_int(prompt: str, default: int) -> int:
    """Ask for an integer, re-prompt on bad input."""
    while True:
        raw = _ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"    Please enter a whole number (example: {default}).")


def _ask_yes(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question. Returns True for yes, False for no."""
    tag = "[Y/n]" if default else "[y/N]"
    raw = _ask(f"{prompt} {tag}", "y" if default else "n").lower()
    return raw in ("y", "yes", "1", "true")


def _choose(prompt: str, options: list[str]) -> str:
    """Display a numbered list and return the selected option string."""
    print()
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    print()
    while True:
        raw = _ask(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"    Please type a number between 1 and {len(options)}.")


def _parse_complex_list(text: str) -> np.ndarray:
    """Parse a comma-separated list of complex numbers.

    Accepts forms like: -1, -3+2j, -0.2+1.99j, (-3-2j)
    Engineering 'i' notation is also accepted (replaced with 'j').
    """
    text = text.replace(" ", "").replace("i", "j").replace("(", "").replace(")", "")
    parts: list[complex] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(complex(token))
        except ValueError:
            print(f"    Warning: could not parse '{token}' as a complex number — skipped.")
    return np.array(parts, dtype=complex)


def _freq_display(freq_hz: np.ndarray) -> tuple[np.ndarray, str]:
    """Return (scaled_values, unit_label) for the best human-readable scale."""
    fmax = freq_hz.max()
    if fmax >= 1e9:
        return freq_hz / 1e9, "GHz"
    if fmax >= 1e6:
        return freq_hz / 1e6, "MHz"
    if fmax >= 1e3:
        return freq_hz / 1e3, "kHz"
    return freq_hz, "Hz"


# =============================================================================
# Frequency range prompt (shared by S-domain and RLC modes)
# =============================================================================

def _ask_freq_range() -> np.ndarray:
    """Interactively ask for a log-spaced frequency sweep. Returns freq in Hz."""
    print()
    print("  Define the frequency sweep for evaluation (log-spaced points).")
    f_lo  = float(_ask("Start frequency in Hz  (e.g. 1e3 for 1 kHz,  1e6 for 1 MHz)", "1e3"))
    f_hi  = float(_ask("Stop  frequency in Hz  (e.g. 1e9 for 1 GHz, 1e12 for 1 THz)", "1e9"))
    n_pts = _ask_int("Number of frequency points", 500)
    return np.logspace(np.log10(f_lo), np.log10(f_hi), n_pts)


# =============================================================================
# Shared Vector Fitting pipeline: fit -> report -> plot -> optional export
# =============================================================================

def _run_fit(freq_hz: np.ndarray, H: np.ndarray, label: str) -> object:
    """Ask VF settings, run the fit, print a summary, and return the model."""
    _section("Vector Fitting settings")

    n_poles = _ask_int(
        "Number of poles  (start low, e.g. 4 for smooth data, 6-12 for resonances)",
        6,
    )
    n_iter = _ask_int("Maximum iterations", 30)

    print()
    print("  Weighting strategy affects how the fit is balanced across the frequency range:")
    print("    - inverse: recommended for impedance (Z) and wide dynamic range")
    print("    - uniform: recommended for S-parameters or narrowband data")
    w_choice = _choose("Weighting strategy", [
        "inverse  (1/|H|, good for impedance / wide dynamic range)",
        "uniform  (equal weight, good for S-parameters)",
    ])
    weight = "inverse" if w_choice.startswith("inverse") else "uniform"

    _section("Running Vector Fitting")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = VectorFitter(
            n_poles=n_poles,
            n_iter_max=n_iter,
            weight=weight,
        ).fit(freq_hz, H)

    for w in caught:
        print(f"  Warning: {w.message}")

    H_fit   = model.evaluate(freq_hz)
    rms_rel = float(np.sqrt(np.mean(
        np.abs((H_fit - H) / (np.abs(H) + 1e-30)) ** 2
    )))
    print(f"\n  Poles fitted   : {model.n_poles}")
    print(f"  Iterations     : {len(model.rms_error_history)}")
    print(f"  RMS (absolute) : {model.rms_error:.4e}")
    print(f"  RMS (relative) : {rms_rel:.4e}")
    return model


def _print_pole_table(model) -> None:
    """Print a formatted table of pole locations with frequency and Q factor."""
    poles_sorted = sort_by_frequency(model.poles)
    print(f"\n  {'#':>3}  {'Re(p) [rad/s]':>18}  {'Im(p) [rad/s]':>18}"
          f"  {'f0 [MHz]':>10}  {'Q':>8}  {'type':>8}")
    print(f"  {'─'*3}  {'─'*18}  {'─'*18}  {'─'*10}  {'─'*8}  {'─'*8}")
    shown: set[tuple] = set()
    for i, p in enumerate(poles_sorted):
        key = (round(p.real, 4), round(abs(p.imag), 4))
        if key in shown:
            continue
        shown.add(key)
        f0_hz = pole_resonant_frequency(p)
        Q     = pole_quality_factor(p)
        kind  = "real" if abs(p.imag) < 1e-3 * (abs(p) + 1e-30) else "complex"
        print(f"  {i:>3}  {p.real:>18.4e}  {p.imag:>18.4e}"
              f"  {f0_hz/1e6:>10.3f}  {Q:>8.2f}  {kind:>8}")


def _print_rlc_table(network) -> None:
    """Print a formatted table of synthesised RLC element values."""
    print(f"\n  {'#':>3}  {'type':>7}  {'R [Ohm]':>12}  {'L [H]':>12}  {'C [F]':>12}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*12}")
    for i, b in enumerate(network.branches):
        R = f"{b.R:.4e}" if b.R is not None else "—"
        L = f"{b.L:.4e}" if b.L is not None else "—"
        C = f"{b.C:.4e}" if b.C is not None else "—"
        print(f"  {i:>3}  {b.branch_type:>7}  {R:>12}  {L:>12}  {C:>12}")


def _show_results(freq_hz: np.ndarray, H: np.ndarray, model, label: str) -> None:
    """Print pole table, synthesis, offer SPICE export, then show plots."""

    _section("Pole characterisation")
    _print_pole_table(model)
    print(f"\n  Direct term  d  = {model.d:.4e}  (real constant, acts like resistance)")
    print(f"  Slope  term  e  = {model.e:.4e}  (proportional to s, acts like inductance)")

    _section("Foster RLC synthesis")
    network = None
    try:
        network = foster_synthesis(model)
        _print_rlc_table(network)
        Z_synth = network.impedance(freq_hz)
        Z_model = model.evaluate(freq_hz)
        rms_syn = float(np.sqrt(np.mean(
            np.abs((Z_synth - Z_model) / (np.abs(Z_model) + 1e-30)) ** 2
        )))
        print(f"\n  Round-trip RMS (Foster network vs fitted model) = {rms_syn:.4e}")
        print("  (A small value means the element values accurately reproduce the fit.)")
    except Exception as exc:
        print(f"  Foster synthesis could not be completed: {exc}")

    if network and _ask_yes("\n  Export SPICE netlists (.cir files)?", default=False):
        stem = label.replace(" ", "_").replace("/", "_")[:40]
        foster_path = Path(f"{stem}_foster.cir")
        beh_path    = Path(f"{stem}_behavioral.cir")
        export_spice_foster(network, foster_path,
                            subckt_name=stem.upper() + "_FOSTER")
        export_spice_behavioral(model, beh_path,
                                subckt_name=stem.upper() + "_BEH")
        print(f"  Written: {foster_path}")
        print(f"  Written: {beh_path}")

    _section("Generating plots")
    _plot_pipeline(freq_hz, H, model, network, label)


def _plot_pipeline(
    freq_hz: np.ndarray,
    H: np.ndarray,
    model,
    network,
    label: str,
) -> None:
    """Create a 5-panel summary figure: Bode, pole-zero, error, convergence."""
    H_fit          = model.evaluate(freq_hz)
    f_disp, f_unit = _freq_display(freq_hz)

    C_MEAS  = "#2c7bb6"
    C_FIT   = "#d7191c"
    C_SYNTH = "#1a9641"

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(label, fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    ax_mag  = fig.add_subplot(gs[0, 0])
    ax_ph   = fig.add_subplot(gs[1, 0])
    ax_pz   = fig.add_subplot(gs[:, 1])
    ax_err  = fig.add_subplot(gs[0, 2])
    ax_conv = fig.add_subplot(gs[1, 2])

    # ── Bode ──────────────────────────────────────────────────────────────────
    mag_m = 20 * np.log10(np.abs(H)     + 1e-300)
    mag_f = 20 * np.log10(np.abs(H_fit) + 1e-300)
    ph_m  = np.degrees(np.unwrap(np.angle(H)))
    ph_f  = np.degrees(np.unwrap(np.angle(H_fit)))

    ax_mag.semilogx(f_disp, mag_m, color=C_MEAS, lw=1.5, label="Input data")
    ax_mag.semilogx(f_disp, mag_f, color=C_FIT,  lw=2,   ls="--",
                    label=f"VF fit  (n={model.n_poles})")
    ax_ph.semilogx(f_disp, ph_m, color=C_MEAS, lw=1.5)
    ax_ph.semilogx(f_disp, ph_f, color=C_FIT,  lw=2,   ls="--")

    if network is not None:
        Z_s = network.impedance(freq_hz)
        ax_mag.semilogx(f_disp, 20 * np.log10(np.abs(Z_s) + 1e-300),
                        color=C_SYNTH, lw=1.5, ls=":", label="Foster network")
        ax_ph.semilogx(f_disp, np.degrees(np.unwrap(np.angle(Z_s))),
                       color=C_SYNTH, lw=1.5, ls=":")

    ax_mag.set_ylabel("Magnitude (dB)", fontsize=9)
    ax_mag.set_title("Bode plot", fontsize=9)
    ax_mag.legend(fontsize=8)
    ax_ph.set_ylabel("Phase (deg)", fontsize=9)
    ax_ph.set_xlabel(f"Frequency ({f_unit})", fontsize=9)

    for ax in (ax_mag, ax_ph):
        ax.grid(True, which="both", color="#cccccc", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    # ── Pole-zero map ─────────────────────────────────────────────────────────
    poles = model.poles
    zeros = model.zeros

    ax_pz.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax_pz.scatter(poles.real, poles.imag, marker="x", s=150, color="#e66101",
                  linewidths=2.5, zorder=5, label=f"Poles ({len(poles)})")
    if len(zeros) > 0:
        ax_pz.scatter(zeros.real, zeros.imag, marker="o", s=90,
                      facecolors="none", edgecolors="#1a9641",
                      linewidths=2, zorder=5, label=f"Zeros ({len(zeros)})")

    if len(poles) > 0:
        x_max  = max(abs(poles.real).max() * 1.5, 1.0)
        y_span = [poles.imag.min() * 1.4 - 1, poles.imag.max() * 1.4 + 1]
        ax_pz.fill_betweenx(y_span, 0, x_max, alpha=0.05, color="red",
                            label="Unstable region (RHP)")

    ax_pz.set_xlabel("Real part  (rad/s)", fontsize=9)
    ax_pz.set_ylabel("Imaginary part  (rad/s)", fontsize=9)
    ax_pz.set_title("S-plane  pole-zero map", fontsize=9)
    ax_pz.legend(fontsize=8)
    ax_pz.grid(True, color="#cccccc", lw=0.5)
    ax_pz.spines["top"].set_visible(False)
    ax_pz.spines["right"].set_visible(False)
    ax_pz.tick_params(labelsize=8)

    # ── Absolute error ────────────────────────────────────────────────────────
    err_vf = np.abs(H_fit - H)
    ax_err.semilogx(f_disp, err_vf, color=C_FIT,  lw=1.5, label="VF fit error")
    if network is not None:
        ax_err.semilogx(f_disp, np.abs(Z_s - H), color=C_SYNTH, lw=1.5, ls="--",
                        label="Foster network error")
    ax_err.set_ylabel("|error|", fontsize=9)
    ax_err.set_xlabel(f"Frequency ({f_unit})", fontsize=9)
    ax_err.set_title("Absolute error vs input data", fontsize=9)
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which="both", color="#cccccc", lw=0.5)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)
    ax_err.tick_params(labelsize=8)

    # ── Convergence ───────────────────────────────────────────────────────────
    hist = model.rms_error_history
    ax_conv.semilogy(range(1, len(hist) + 1), hist, "o-", color=C_MEAS, lw=2, ms=5)
    ax_conv.axhline(hist[-1], color=C_FIT, ls="--", lw=1,
                    label=f"Final = {hist[-1]:.2e}")
    ax_conv.set_xlabel("Iteration", fontsize=9)
    ax_conv.set_ylabel("RMS error", fontsize=9)
    ax_conv.set_title("Convergence history", fontsize=9)
    ax_conv.legend(fontsize=8)
    ax_conv.grid(True, which="both", color="#cccccc", lw=0.5)
    ax_conv.spines["top"].set_visible(False)
    ax_conv.spines["right"].set_visible(False)
    ax_conv.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_name = label.replace(" ", "_").replace("/", "_")[:50] + ".png"
    fig.savefig(png_name, dpi=150)
    print(f"  Plot saved: {png_name}")
    plt.show()


# =============================================================================
# Mode 1 — Load from a measurement file
# =============================================================================

def _mode_csv() -> None:
    """Guide the user through loading a CSV or Touchstone measurement file."""
    _section("Load from a measurement file")

    print()
    print("  Supported file formats:")
    print("    1. CSV file with Real + Imaginary columns")
    print("    2. CSV file with Magnitude + Phase columns")
    print("    3. Touchstone file  (.s1p or .s2p from a VNA)")
    print()

    fmt = _choose("Select file format", [
        "CSV  —  two data columns: Re(H) and Im(H)",
        "CSV  —  two data columns: Magnitude and Phase",
        "Touchstone  (.s1p or .s2p)",
    ])

    print()
    print("  Tip: you can copy and paste the full file path here.")
    file_path = _ask("Path to your data file")
    if not Path(file_path).exists():
        print(f"\n  Error: file not found at '{file_path}'")
        print("  Check the path and try this mode again.")
        return

    print()
    print("  Frequency unit — leave blank to let the tool auto-detect from the data.")
    freq_unit_raw = _ask(
        "Frequency unit  (Hz / kHz / MHz / GHz,  or leave blank = auto-detect)", ""
    )
    freq_unit = freq_unit_raw.strip() or None

    # ── Format A: Re / Im ─────────────────────────────────────────────────────
    if fmt.startswith("CSV  —  two data columns: Re"):
        print()
        print("  Enter the column name exactly as it appears in the file header,")
        print("  or enter the column number (0 = first column, 1 = second, ...).")
        freq_col = _ask("Frequency column  (name or index)", "0")
        re_col   = _ask("Real part column  (name or index)", "1")
        im_col   = _ask("Imaginary part column  (name or index)", "2")

        freq_col = int(freq_col) if freq_col.isdigit() else freq_col
        re_col   = int(re_col)   if re_col.isdigit()   else re_col
        im_col   = int(im_col)   if im_col.isdigit()   else im_col

        data = load_ri_csv(
            file_path,
            freq_col=freq_col,
            re_col=re_col,
            im_col=im_col,
            freq_unit=freq_unit,
        )

    # ── Format B: Magnitude / Phase ───────────────────────────────────────────
    elif fmt.startswith("CSV  —  two data columns: Mag"):
        freq_col  = _ask("Frequency column  (name or index)", "0")
        mag_col   = _ask("Magnitude column  (name or index)", "1")
        phase_col = _ask("Phase column      (name or index)", "2")

        mag_unit = _choose("Magnitude unit", ["linear", "dB"])
        phase_u  = _choose("Phase unit",     ["deg", "rad"])

        freq_col  = int(freq_col)  if freq_col.isdigit()  else freq_col
        mag_col   = int(mag_col)   if mag_col.isdigit()   else mag_col
        phase_col = int(phase_col) if phase_col.isdigit() else phase_col

        data = load_csv(
            file_path,
            freq_col=freq_col,
            mag_col=mag_col,
            phase_col=phase_col,
            mag_unit=mag_unit,
            phase_unit=phase_u,
            freq_unit=freq_unit,
        )

    # ── Format C: Touchstone ──────────────────────────────────────────────────
    else:
        print()
        print("  For a 2-port (.s2p) file, specify which S-parameter to extract.")
        print("  Examples:  1,1  for S11     1,2  for S21     2,1  for S12")
        port_str = _ask("S-parameter port pair  (row, column)", "1,1")
        try:
            parts = [p.strip() for p in port_str.split(",")]
            port  = (int(parts[0]), int(parts[1]))
        except (IndexError, ValueError):
            print("  Could not parse port pair — defaulting to S11 (1,1).")
            port = (1, 1)
        data = load_touchstone(file_path, port=port)

    print()
    print(data.summary())

    model = _run_fit(data.freq_hz, data.H, data.label)
    _show_results(data.freq_hz, data.H, model, data.label)


# =============================================================================
# Mode 2 — S-domain transfer function
# =============================================================================

def _mode_sdomain() -> None:
    """Guide the user through defining H(s) analytically."""
    _section("S-domain transfer function")

    print()
    print("  Define your transfer function H(s) in one of two ways:")
    print()

    sub = _choose("Input method", [
        "Poles and zeros  (enter the pole and zero locations directly)",
        "Polynomial coefficients  (enter numerator and denominator as polynomials in s)",
    ])

    if sub.startswith("Poles"):
        freq_hz, H = _sdomain_pz()
    else:
        freq_hz, H = _sdomain_poly()

    label = "S-domain transfer function"
    model = _run_fit(freq_hz, H, label)
    _show_results(freq_hz, H, model, label)


def _sdomain_pz() -> tuple[np.ndarray, np.ndarray]:
    """Collect poles, zeros, gain from the user and evaluate H(s) on a frequency grid."""
    print()
    print("  Enter pole and zero locations as comma-separated complex numbers.")
    print()
    print("  Format examples:")
    print("    Two real poles              :  -1, -3")
    print("    Complex conjugate pair      :  -0.2+2j, -0.2-2j")
    print("    Mix of real and complex     :  -5, -1+3j, -1-3j")
    print("    Engineering notation (i=j)  :  -1+2i, -1-2i")
    print()

    poles_str = _ask("Poles  (comma-separated complex numbers)")
    zeros_str = _ask(
        "Zeros  (comma-separated, or leave blank if there are no zeros)", ""
    )
    gain = _ask_float("Gain  (scalar multiplier applied to the whole function)", 1.0)

    poles = _parse_complex_list(poles_str)
    zeros = (
        _parse_complex_list(zeros_str)
        if zeros_str.strip()
        else np.array([], dtype=complex)
    )

    if len(poles) == 0:
        print("  No valid poles were parsed. Returning to the menu.")
        return np.array([1.0]), np.array([1.0 + 0j])

    freq_hz = _ask_freq_range()
    s = 1j * 2 * np.pi * freq_hz

    # H(s) = gain * prod(s - z_i) / prod(s - p_i)
    num = np.ones_like(s)
    for z in zeros:
        num *= (s - z)
    den = np.ones_like(s)
    for p in poles:
        den *= (s - p)
    H = gain * num / den

    print(f"\n  H(s) has {len(poles)} pole(s) and {len(zeros)} zero(s),  gain = {gain}")
    print(f"  |H| range on the sweep: {np.abs(H).min():.3e}  to  {np.abs(H).max():.3e}")
    return freq_hz, H


def _sdomain_poly() -> tuple[np.ndarray, np.ndarray]:
    """Collect polynomial coefficients from the user and evaluate H(s)."""
    print()
    print("  Enter polynomial coefficients from the HIGHEST power of s down to s^0.")
    print()
    print("  Examples:")
    print("    s                =>  1, 0          (meaning: 1*s^1 + 0*s^0)")
    print("    s^2 + 4s + 3     =>  1, 4, 3")
    print("    2s               =>  2, 0")
    print("    s^2 + 0.4s + 4   =>  1, 0.4, 4")
    print()

    num_str = _ask(
        "Numerator coefficients    (highest power first, e.g.  1, 0  for 's')"
    )
    den_str = _ask(
        "Denominator coefficients  (highest power first, e.g.  1, 4, 3  for 's^2+4s+3')"
    )

    try:
        num_coeffs = [float(x.strip()) for x in num_str.split(",") if x.strip()]
        den_coeffs = [float(x.strip()) for x in den_str.split(",") if x.strip()]
    except ValueError as exc:
        print(f"  Error reading coefficients: {exc}")
        return np.array([1.0]), np.array([1.0 + 0j])

    if not num_coeffs or not den_coeffs:
        print("  One or both coefficient lists were empty. Returning to the menu.")
        return np.array([1.0]), np.array([1.0 + 0j])

    freq_hz = _ask_freq_range()
    s = 1j * 2 * np.pi * freq_hz

    H = np.polyval(num_coeffs, s) / np.polyval(den_coeffs, s)

    print(f"\n  Numerator degree  : {len(num_coeffs) - 1}")
    print(f"  Denominator degree: {len(den_coeffs) - 1}")
    print(f"  |H| range on the sweep: {np.abs(H).min():.3e}  to  {np.abs(H).max():.3e}")
    return freq_hz, H


# =============================================================================
# Mode 3 — RLC circuit builder
# =============================================================================

def _mode_rlc() -> None:
    """Guide the user through building a multi-branch RLC circuit."""
    _section("RLC circuit builder")

    print()
    print("  Add one or more circuit branches.  The total impedance Z(s) is the")
    print("  SUM of all branch impedances  (all branches connected in series).")
    print()
    print("  Each branch is a series combination of R, L, and/or C elements.")
    print()
    print("  Examples of common circuits:")
    print("    Simple series RLC :  one branch  R+L+C")
    print("    Lossy inductor    :  one branch  R+L  (winding resistance + inductance)")
    print("    Resonator         :  one branch  R+L+C  (the resonant tank)")
    print("    Two-stage filter  :  two branches  R+L  and  R+C")
    print()

    branches: list[dict] = []

    while True:
        print(f"\n  Branches added so far: {len(branches)}")
        for i, b in enumerate(branches):
            parts = []
            if "R" in b:
                parts.append(f"R = {b['R']:.4g} Ohm")
            if "L" in b:
                parts.append(f"L = {b['L']:.4g} H")
            if "C" in b:
                parts.append(f"C = {b['C']:.4g} F")
            print(f"    [{i}]  {b['label']}:  {',  '.join(parts)}")

        if not _ask_yes("Add another branch?", default=True):
            break

        branch = _ask_rlc_branch()
        if branch:
            branches.append(branch)

    if not branches:
        print("\n  No branches were defined. Returning to the main menu.")
        return

    freq_hz = _ask_freq_range()
    s = 1j * 2 * np.pi * freq_hz

    Z_total = np.zeros_like(s, dtype=complex)
    for b in branches:
        Z_total += _branch_impedance(b, s)

    print(f"\n  Circuit: {len(branches)} branch(es) in series.")
    print(f"  |Z| range: {np.abs(Z_total).min():.3e}  to  {np.abs(Z_total).max():.3e}  Ohm")

    label = "RLC circuit  Z(s)"
    model = _run_fit(freq_hz, Z_total, label)
    _show_results(freq_hz, Z_total, model, label)


def _ask_rlc_branch() -> dict | None:
    """Interactively collect one RLC branch definition. Returns None if empty."""
    print()
    btype = _choose("Branch topology", [
        "R only           (pure resistor)",
        "L only           (pure inductor)",
        "C only           (pure capacitor)",
        "R + L in series  (winding resistance with inductance)",
        "R + C in series  (resistor with capacitor)",
        "R + L + C in series  (full series RLC branch)",
    ])

    branch: dict = {"label": btype.split("(")[0].strip()}
    has_R = "R" in btype and not btype.startswith("L") and not btype.startswith("C")
    has_L = "L" in btype
    has_C = "C" in btype

    print()
    if has_R:
        branch["R"] = _ask_float(
            "  R value in Ohms  (e.g. 50 for 50 Ohm,  0.1 for 100 mOhm)", 50.0
        )
    if has_L:
        branch["L"] = _ask_float(
            "  L value in Henries  (e.g. 1e-9 for 1 nH,  1e-6 for 1 uH)", 1e-9
        )
    if has_C:
        branch["C"] = _ask_float(
            "  C value in Farads  (e.g. 1e-12 for 1 pF,  1e-9 for 1 nF)", 1e-12
        )

    if not any(k in branch for k in ("R", "L", "C")):
        print("  No element values were entered. Branch not added.")
        return None

    return branch


def _branch_impedance(branch: dict, s: np.ndarray) -> np.ndarray:
    """Compute Z(s) = R + sL + 1/(sC) for a series branch."""
    Z = np.zeros_like(s, dtype=complex)
    if "R" in branch:
        Z += branch["R"]
    if "L" in branch:
        Z += s * branch["L"]
    if "C" in branch:
        Z += 1.0 / (s * branch["C"])
    return Z


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    _banner()

    print("  This tool fits a rational transfer function H(s) to frequency-domain data")
    print("  using the Vector Fitting (VF) algorithm  (Gustavsen & Semlyen, 1999).")
    print()
    print("  At the end of each analysis you will receive:")
    print("    - Bode plot  (magnitude and phase vs frequency)")
    print("    - S-plane pole-zero map")
    print("    - VF convergence history")
    print("    - Foster RLC synthesis  (element values R, L, C)")
    print("    - Optional SPICE netlist export  (.cir files)")

    while True:
        print()
        print()
        print("  ================================================================")
        print("   Choose your analysis input")
        print("  ================================================================")

        mode = _choose("Input method", [
            "1. Load from a measurement file  (CSV or Touchstone .s1p / .s2p)",
            "2. Define an S-domain transfer function  (poles/zeros or polynomial)",
            "3. Define an RLC circuit  (specify R, L, C component values)",
            "4. Exit",
        ])

        if mode.startswith("1"):
            _mode_csv()
        elif mode.startswith("2"):
            _mode_sdomain()
        elif mode.startswith("3"):
            _mode_rlc()
        else:
            print()
            print("  Goodbye.")
            break

        print()
        if not _ask_yes("Run another analysis?", default=True):
            print()
            print("  Goodbye.")
            break


if __name__ == "__main__":
    main()
