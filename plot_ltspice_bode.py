"""
plot_ltspice_bode.py
====================
Read LTspice .raw files (tb_foster.raw, tb_behavioral.raw) and produce
a Bode plot (magnitude + phase) comparing Foster RLC vs Behavioral Laplace.

Usage:
    python plot_ltspice_bode.py

Output:
    ltspice_bode_comparison.png
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# LTspice binary .raw parser
# ---------------------------------------------------------------------------

@dataclass
class RawFile:
    title: str = ""
    date: str = ""
    flags: str = ""
    n_vars: int = 0
    n_points: int = 0
    variables: list[tuple[int, str, str]] = field(default_factory=list)
    freq: np.ndarray = field(default_factory=lambda: np.array([]))
    data: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (n_vars-1, n_points)


def _decode_header(raw_bytes: bytes) -> tuple[dict, int]:
    """Decode UTF-16LE header and return (fields_dict, binary_start_offset)."""
    # Decode as UTF-16LE, replacing bad surrogates
    chars = []
    i = 0
    while i < len(raw_bytes) - 1:
        word = raw_bytes[i:i + 2]
        try:
            c = word.decode("utf-16-le")
        except UnicodeDecodeError:
            c = "?"
        chars.append(c)
        if "".join(chars[-8:]) == "Binary:\n":
            binary_start = i + 2
            return "".join(chars), binary_start
        i += 2
    raise ValueError("Could not find 'Binary:' marker in raw file header.")


def load_raw(path: Path, max_points: int | None = None) -> RawFile:
    """Load an LTspice binary .raw file.

    Parameters
    ----------
    path:
        Path to the .raw file.
    max_points:
        If given, only read this many points (downsampled evenly for large files).
    """
    with open(path, "rb") as fh:
        header_blob = fh.read(8192)   # header is always < 8 KB
        header_txt, bin_start = _decode_header(header_blob)
        # Seek back and read from binary start
        fh.seek(bin_start)
        bin_data = fh.read()

    rf = RawFile()

    # --- parse header fields ---
    for line in header_txt.splitlines():
        line = line.strip()
        if line.lower().startswith("title:"):
            rf.title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            rf.date = line.split(":", 1)[1].strip()
        elif line.lower().startswith("flags:"):
            rf.flags = line.split(":", 1)[1].strip()
        elif line.lower().startswith("no. variables:"):
            rf.n_vars = int(line.split(":")[-1].strip())
        elif line.lower().startswith("no. points:"):
            rf.n_points = int(line.split(":")[-1].strip())

    # --- parse variable list (tab-separated: index  name  type) ---
    var_section = False
    for line in header_txt.splitlines():
        if line.strip().lower() == "variables:":
            var_section = True
            continue
        if var_section and line.strip().lower() == "binary:":
            break
        if var_section and line.strip():
            parts = line.split("\t")
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                idx = int(parts[0]) if parts[0].isdigit() else len(rf.variables)
                name = parts[1] if len(parts) > 1 else f"var{idx}"
                vtype = parts[2] if len(parts) > 2 else ""
                rf.variables.append((idx, name, vtype))

    # --- binary data: each point = n_vars complex doubles (16 bytes each) ---
    bytes_per_point = rf.n_vars * 16  # 2 float64 per complex
    expected = rf.n_points * bytes_per_point
    if len(bin_data) < expected:
        raise ValueError(
            f"Binary section too short: got {len(bin_data)} bytes, "
            f"expected {expected} for {rf.n_points} points x {rf.n_vars} vars."
        )

    # Optionally downsample for huge files
    indices = np.arange(rf.n_points)
    if max_points is not None and rf.n_points > max_points:
        indices = np.linspace(0, rf.n_points - 1, max_points, dtype=int)

    # Read all data as float64, then reshape
    # Layout: [pt0_var0_re, pt0_var0_im, pt0_var1_re, pt0_var1_im, ...]
    all_floats = np.frombuffer(bin_data[:expected], dtype="<f8").reshape(
        rf.n_points, rf.n_vars, 2
    )
    # Downsample
    all_floats = all_floats[indices]

    # Build complex array: shape (n_selected, n_vars)
    cmplx = all_floats[..., 0] + 1j * all_floats[..., 1]

    rf.freq = cmplx[:, 0].real          # frequency is always real
    rf.data = cmplx[:, 1:].T            # shape (n_vars-1, n_points)
    rf.n_points = len(indices)

    return rf


def _var_index(rf: RawFile, name: str) -> int:
    """Return the data index (0-based, excluding frequency) for a variable name."""
    for idx, vname, _ in rf.variables:
        if idx == 0:
            continue  # skip frequency variable
        if vname.lower() == name.lower():
            return idx - 1  # offset by 1 because freq is stripped from data
    names = [v[1] for v in rf.variables if v[0] != 0]
    raise KeyError(f"Variable '{name}' not found. Available: {names}")


# ---------------------------------------------------------------------------
# Bode plot
# ---------------------------------------------------------------------------

def load_csv_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency (Hz) and Z (complex) from the two_rlc measurement CSV.

    Expects columns: Frequency[MHz], Re[Z_Ohm], Im[Z_Ohm]
    (case-insensitive, order-independent).
    """
    import csv
    freq_hz: list[float] = []
    Z: list[complex] = []

    def _find_col(headers: list[str], keyword: str, exclude: str = "") -> str:
        """Return the first header containing keyword but not exclude."""
        for h in headers:
            hl = h.lower()
            if keyword in hl and (not exclude or exclude not in hl):
                return h
        raise KeyError(f"No column matching '{keyword}' (excluding '{exclude}') in {headers}")

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        col_f  = _find_col(headers, "freq")
        col_re = _find_col(headers, "re[")          # "Re[Z_Ohm]" — bracket makes it unique
        col_im = _find_col(headers, "im[")          # "Im[Z_Ohm]"
        for row in reader:
            f_mhz = float(row[col_f])
            re    = float(row[col_re])
            im    = float(row[col_im])
            freq_hz.append(f_mhz * 1e6)
            Z.append(complex(re, im))
    return np.array(freq_hz), np.array(Z)


def plot_bode_comparison(
    foster_path: Path,
    behavioral_path: Path,
    out_png: Path,
    csv_path: Path | None = None,
    max_points: int = 2000,
) -> None:
    """Load both raw files and produce a 2-row Bode comparison plot."""

    print(f"Loading {foster_path.name}  ...", end=" ", flush=True)
    rf_f = load_raw(foster_path, max_points=max_points)
    print(f"OK  ({rf_f.n_points} points, {len(rf_f.variables)} vars)")

    print(f"Loading {behavioral_path.name}  ...", end=" ", flush=True)
    rf_b = load_raw(behavioral_path, max_points=max_points)
    print(f"OK  ({rf_b.n_points} points, {len(rf_b.variables)} vars)")

    # ---- Foster: Z = V(1) / I(Vin) ----------------------------------------
    v1     = rf_f.data[_var_index(rf_f, "V(1)")]
    ivin_f = rf_f.data[_var_index(rf_f, "I(Vin)")]
    # LTspice voltage-source convention: I(Vin) flows from + to - INSIDE the
    # source, which is opposite to the load current direction.
    # For passive load driven by Vin 1 0 AC 1: Z = V(1) / (-I(Vin))
    Z_foster = v1 / (-ivin_f)

    # ---- Behavioral: V(out) = H(s) * V(in) = Z(s) * 1V -------------------
    Z_beh = rf_b.data[_var_index(rf_b, "V(out)")]

    freq_f = rf_f.freq
    freq_b = rf_b.freq

    # ---- Bode: magnitude in dB, phase in degrees --------------------------
    mag_f_db = 20 * np.log10(np.abs(Z_foster) + 1e-300)
    ph_f_deg = np.degrees(np.unwrap(np.angle(Z_foster)))

    mag_b_db = 20 * np.log10(np.abs(Z_beh) + 1e-300)
    ph_b_deg = np.degrees(np.unwrap(np.angle(Z_beh)))

    # ---- Optional measurement data ----------------------------------------
    freq_csv = Z_csv = None
    if csv_path is not None and csv_path.exists():
        print(f"Loading {csv_path.name}  ...", end=" ", flush=True)
        freq_csv, Z_csv = load_csv_data(csv_path)
        print(f"OK  ({len(freq_csv)} points)")

    # ---- Plot ---------------------------------------------------------------
    C_MEAS   = "#1a9641"
    C_FOSTER = "#d7191c"
    C_BEH    = "#2c7bb6"

    # 3 rows when measurement CSV is available (adds error subplot), else 2
    has_csv = Z_csv is not None
    nrows = 3 if has_csv else 2
    height_ratios = [3, 1.5, 2] if has_csv else [3, 2]
    fig, axes = plt.subplots(
        nrows, 1, figsize=(12, 8 if has_csv else 7), sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax_mag = axes[0]
    ax_err = axes[1] if has_csv else None
    ax_ph  = axes[-1]

    fig.suptitle(
        "LTspice Verification — Foster RLC vs Behavioral Laplace\n"
        f"(Foster: {foster_path.stem}.cir  |  Behavioral: {behavioral_path.stem}.cir)",
        fontsize=11, fontweight="bold",
    )

    # Magnitude
    if Z_csv is not None:
        mag_csv_db = 20 * np.log10(np.abs(Z_csv) + 1e-300)
        ax_mag.semilogx(freq_csv, mag_csv_db, color=C_MEAS, lw=2.5, zorder=3,
                        label="Measurement data")
    ax_mag.semilogx(freq_f, mag_f_db, color=C_FOSTER, lw=1.8, zorder=2, ls="--",
                    label="Foster RLC network (LTspice)")
    ax_mag.semilogx(freq_b, mag_b_db, color=C_BEH,    lw=1.5, zorder=2, ls=":",
                    label="Behavioral Laplace H(s) (LTspice)")
    ax_mag.set_ylabel("|Z(jω)|  (dB)", fontsize=10)
    ax_mag.legend(fontsize=9)
    ax_mag.grid(True, which="both", color="#cccccc", lw=0.5)
    ax_mag.spines["top"].set_visible(False)
    ax_mag.spines["right"].set_visible(False)
    ax_mag.tick_params(labelsize=9)

    # Error vs measurement (dB deviation) — only when CSV available
    if has_csv and ax_err is not None:
        # Interpolate LTspice results onto CSV frequency grid
        Z_f_interp = np.interp(freq_csv, freq_f, Z_foster.real) + \
                     1j * np.interp(freq_csv, freq_f, Z_foster.imag)
        Z_b_interp = np.interp(freq_csv, freq_b, Z_beh.real) + \
                     1j * np.interp(freq_csv, freq_b, Z_beh.imag)
        err_f_db = 20 * np.log10(np.abs(Z_f_interp) + 1e-300) - mag_csv_db
        err_b_db = 20 * np.log10(np.abs(Z_b_interp) + 1e-300) - mag_csv_db
        ax_err.semilogx(freq_csv, err_f_db, color=C_FOSTER, lw=1.5, ls="--",
                        label="Foster error vs measurement")
        ax_err.semilogx(freq_csv, err_b_db, color=C_BEH,    lw=1.5, ls=":",
                        label="Behavioral error vs measurement")
        ax_err.axhline(0, color="#888", lw=0.8, ls="-")
        ax_err.set_ylabel("Error  (dB)", fontsize=10)
        ax_err.legend(fontsize=8, loc="upper left")
        ax_err.grid(True, which="both", color="#cccccc", lw=0.5)
        ax_err.spines["top"].set_visible(False)
        ax_err.spines["right"].set_visible(False)
        ax_err.tick_params(labelsize=9)

    # Phase
    if Z_csv is not None:
        ph_csv_deg = np.degrees(np.unwrap(np.angle(Z_csv)))
        ax_ph.semilogx(freq_csv, ph_csv_deg, color=C_MEAS, lw=2.5, zorder=3)
    ax_ph.semilogx(freq_f, ph_f_deg, color=C_FOSTER, lw=1.8, zorder=2, ls="--")
    ax_ph.semilogx(freq_b, ph_b_deg, color=C_BEH,    lw=1.5, zorder=2, ls=":")
    ax_ph.set_ylabel("Phase  (degrees)", fontsize=10)
    ax_ph.set_xlabel("Frequency  (Hz)", fontsize=10)
    ax_ph.grid(True, which="both", color="#cccccc", lw=0.5)
    ax_ph.spines["top"].set_visible(False)
    ax_ph.spines["right"].set_visible(False)
    ax_ph.tick_params(labelsize=9)

    # Nice frequency tick labels
    for ax in (ax_mag, ax_ph):
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: (
                    f"{x/1e9:.3g} GHz" if x >= 1e9
                    else f"{x/1e6:.3g} MHz" if x >= 1e6
                    else f"{x/1e3:.3g} kHz" if x >= 1e3
                    else f"{x:.3g} Hz"
                )
            )
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_png}")


# ---------------------------------------------------------------------------
# Entry point — accepts CLI arguments for any example
# ---------------------------------------------------------------------------
#
# Usage:
#   python plot_ltspice_bode.py                          # default: two_rlc
#   python plot_ltspice_bode.py --foster examples/tb_inductor_foster.raw \
#                               --behavioral examples/tb_inductor_behavioral.raw \
#                               --csv examples/data_inductor_Z.csv \
#                               --out inductor_bode.png
#
# All paths may be absolute or relative to the current working directory.

if __name__ == "__main__":
    import argparse

    base = Path(__file__).parent / "examples"

    parser = argparse.ArgumentParser(
        description="Plot LTspice AC Bode: Foster RLC vs Behavioral Laplace"
    )
    parser.add_argument(
        "--foster", type=Path,
        default=base / "tb_foster.raw",
        help="Path to Foster testbench .raw file (default: examples/tb_foster.raw)",
    )
    parser.add_argument(
        "--behavioral", type=Path,
        default=base / "tb_behavioral.raw",
        help="Path to Behavioral testbench .raw file",
    )
    parser.add_argument(
        "--csv", type=Path,
        default=base / "data_two_rlc_Z.csv",
        help="Optional measurement CSV (Re/Im, frequency in MHz) for overlay",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "ltspice_bode_comparison.png",
        help="Output PNG file path",
    )
    parser.add_argument(
        "--max-points", type=int, default=2000,
        help="Max data points to plot per trace (downsamples large .raw files)",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv.exists() else None
    if args.csv and not args.csv.exists():
        print(f"Note: CSV not found at {args.csv} — skipping measurement overlay.")

    plot_bode_comparison(
        foster_path=args.foster,
        behavioral_path=args.behavioral,
        out_png=args.out,
        csv_path=csv_path,
        max_points=args.max_points,
    )
