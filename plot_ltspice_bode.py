"""
plot_ltspice_bode.py
====================
Read LTspice .raw files and produce a Bode plot comparing
Foster RLC vs Behavioral Laplace against optional measurement data.

Built-in presets (use --preset <name> to skip typing long paths):

    two_rlc   — examples/05_two_rlc_network.py       (default)
    inductor  — examples/04_from_measurement.py  Dataset A
    filter    — examples/04_from_measurement.py  Dataset B
    sdomain1  — examples/02_sdomain.py           Case 1
    sdomain2  — examples/02_sdomain.py           Case 2
    sdomain3  — examples/02_sdomain.py           Case 3

Usage:
    # Named preset — sets all paths automatically:
    python plot_ltspice_bode.py --preset inductor
    python plot_ltspice_bode.py --preset filter

    # Fully custom — specify every path explicitly:
    python plot_ltspice_bode.py \\
        --foster    examples/tb_inductor_foster.raw \\
        --behavioral examples/tb_inductor_behavioral.raw \\
        --csv       examples/data_inductor_Z.csv \\
        --out       inductor_bode.png

    # Behavioral-only (no Foster raw file available):
    python plot_ltspice_bode.py \\
        --behavioral examples/tb_filter_behavioral.raw \\
        --csv        examples/data_filter_S21.csv \\
        --out        filter_beh.png

Notes
-----
- --foster is optional; omit it to show behavioral-only.
- --csv   is optional; omit it to skip the measurement overlay.
- --behavioral is optional; omit it to show foster-only.
- CSV auto-detection: Re/Im columns *or* magnitude-dB + phase-deg columns.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Presets — named shortcuts for each example study
# ---------------------------------------------------------------------------

class _Preset(NamedTuple):
    foster: str | None        # relative path to Foster .raw (or None)
    behavioral: str | None    # relative path to Behavioral .raw (or None)
    csv: str | None           # relative path to measurement CSV (or None)
    out: str                  # output PNG filename
    title: str                # plot super-title
    sweep: str                # human-readable frequency sweep description


_HERE = Path(__file__).parent
_EX   = _HERE / "examples"

PRESETS: dict[str, _Preset] = {
    "two_rlc": _Preset(
        foster     = str(_EX / "tb_foster.raw"),
        behavioral = str(_EX / "tb_behavioral.raw"),
        csv        = str(_EX / "data_two_rlc_Z.csv"),
        out        = "ltspice_bode_two_rlc.png",
        title      = "Two-RLC Network  (05_two_rlc_network.py)",
        sweep      = "1 MHz - 10 GHz",
    ),
    "inductor": _Preset(
        foster     = str(_EX / "tb_inductor_foster.raw"),
        behavioral = str(_EX / "tb_inductor_behavioral.raw"),
        csv        = str(_EX / "data_inductor_Z.csv"),
        out        = "ltspice_bode_inductor.png",
        title      = "Lossy Inductor Impedance  (04_from_measurement.py — Dataset A)",
        sweep      = "10 MHz - 1 GHz",
    ),
    "filter": _Preset(
        foster     = str(_EX / "tb_filter_foster.raw"),
        behavioral = str(_EX / "tb_filter_behavioral.raw"),
        csv        = str(_EX / "data_filter_S21.csv"),
        out        = "ltspice_bode_filter.png",
        title      = "4th-order Butterworth S21  (04_from_measurement.py — Dataset B)",
        sweep      = "50 MHz - 5 GHz",
    ),
    "simple_rlc": _Preset(
        foster     = str(_EX / "tb_simple_rlc_foster.raw"),
        behavioral = str(_EX / "tb_simple_rlc_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_simple_rlc.png",
        title      = "Series RLC Impedance  (01_simple_rlc.py)",
        sweep      = "1 MHz - 10 GHz",
    ),
    "noisy1": _Preset(
        foster     = str(_EX / "tb_noisy1_foster.raw"),
        behavioral = str(_EX / "tb_noisy1_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_noisy1.png",
        title      = "Noisy Data Lesson 1: H=s/((s+1)(s+3)) n=2  (03_noisy_data.py)",
        sweep      = "0.01 Hz - 100 Hz",
    ),
    "noisy2": _Preset(
        foster     = str(_EX / "tb_noisy2_foster.raw"),
        behavioral = str(_EX / "tb_noisy2_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_noisy2.png",
        title      = "Noisy Data Lesson 2: H=2s/(s^2+0.4s+4) uniform weight  (03_noisy_data.py)",
        sweep      = "0.01 Hz - 10 Hz",
    ),
    "noisy3": _Preset(
        foster     = None,
        behavioral = str(_EX / "tb_noisy3_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_noisy3.png",
        title      = "Noisy Data Lesson 3: H=s/(s^2-3s+2) RHP behavioral only  (03_noisy_data.py)",
        sweep      = "0.1 Hz - 100 Hz",
    ),
    "passivity1": _Preset(
        foster     = str(_EX / "tb_passivity1_foster.raw"),
        behavioral = str(_EX / "tb_passivity1_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_passivity1.png",
        title      = "Passivity Scenario 1: noise=0.001 (already passive)  (06_passivity.py)",
        sweep      = "1 MHz - 10 GHz",
    ),
    "passivity2": _Preset(
        foster     = str(_EX / "tb_passivity2_foster.raw"),
        behavioral = str(_EX / "tb_passivity2_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_passivity2.png",
        title      = "Passivity Scenario 2: noise=0.1 after enforcement  (06_passivity.py)",
        sweep      = "1 MHz - 10 GHz",
    ),
    "passivity3": _Preset(
        foster     = str(_EX / "tb_passivity3_foster.raw"),
        behavioral = str(_EX / "tb_passivity3_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_passivity3.png",
        title      = "Passivity Scenario 3: noise=1.0 n=8 after enforcement  (06_passivity.py)",
        sweep      = "1 MHz - 10 GHz",
    ),
    "sdomain1": _Preset(
        foster     = str(_EX / "tb_sdomain1_foster.raw"),
        behavioral = str(_EX / "tb_sdomain1_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_sdomain1.png",
        title      = "S-domain Case 1:  H(s) = s / ((s+1)(s+3))  (02_sdomain.py)",
        sweep      = "0.01 Hz - 100 Hz",
    ),
    "sdomain2": _Preset(
        foster     = str(_EX / "tb_sdomain2_foster.raw"),
        behavioral = str(_EX / "tb_sdomain2_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_sdomain2.png",
        title      = "S-domain Case 2:  H(s) = 2s / (s^2+0.4s+4)  Q=5  (02_sdomain.py)",
        sweep      = "0.01 Hz - 10 Hz",
    ),
    "sdomain3": _Preset(
        foster     = str(_EX / "tb_sdomain3_foster.raw"),
        behavioral = str(_EX / "tb_sdomain3_behavioral.raw"),
        csv        = None,
        out        = "ltspice_bode_sdomain3.png",
        title      = "S-domain Case 3:  H(s) = s / (s^2-3s+2)  RHP  (02_sdomain.py)",
        sweep      = "0.1 Hz - 100 Hz",
    ),
}


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
    data: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_vars-1, n_points)


def _decode_header(raw_bytes: bytes) -> tuple[str, int]:
    """Decode UTF-16LE header; return (header_text, binary_start_offset)."""
    chars: list[str] = []
    i = 0
    while i < len(raw_bytes) - 1:
        word = raw_bytes[i:i + 2]
        try:
            c = word.decode("utf-16-le")
        except UnicodeDecodeError:
            c = "?"
        chars.append(c)
        if "".join(chars[-8:]) == "Binary:\n":
            return "".join(chars), i + 2
        i += 2
    raise ValueError("Could not find 'Binary:' marker in .raw file header.")


def load_raw(path: Path, max_points: int | None = None) -> RawFile:
    """Load an LTspice binary .raw file (AC complex format)."""
    with open(path, "rb") as fh:
        header_blob = fh.read(8192)
        header_txt, bin_start = _decode_header(header_blob)
        fh.seek(bin_start)
        bin_data = fh.read()

    rf = RawFile()

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

    var_section = False
    for line in header_txt.splitlines():
        if line.strip().lower() == "variables:":
            var_section = True
            continue
        if var_section and line.strip().lower() == "binary:":
            break
        if var_section and line.strip():
            parts = [p for p in line.split("\t") if p]
            if len(parts) >= 2:
                idx   = int(parts[0]) if parts[0].isdigit() else len(rf.variables)
                name  = parts[1] if len(parts) > 1 else f"var{idx}"
                vtype = parts[2] if len(parts) > 2 else ""
                rf.variables.append((idx, name, vtype))

    bytes_per_point = rf.n_vars * 16
    expected        = rf.n_points * bytes_per_point
    if len(bin_data) < expected:
        raise ValueError(
            f"Binary section too short: {len(bin_data)} bytes, "
            f"expected {expected} ({rf.n_points} pts × {rf.n_vars} vars × 16 B)."
        )

    indices = np.arange(rf.n_points)
    if max_points is not None and rf.n_points > max_points:
        indices = np.linspace(0, rf.n_points - 1, max_points, dtype=int)

    all_floats = np.frombuffer(bin_data[:expected], dtype="<f8").reshape(
        rf.n_points, rf.n_vars, 2
    )[indices]

    cmplx      = all_floats[..., 0] + 1j * all_floats[..., 1]
    rf.freq    = cmplx[:, 0].real
    rf.data    = cmplx[:, 1:].T
    rf.n_points = len(indices)

    return rf


def _var_index(rf: RawFile, name: str) -> int:
    """Return the data-array index (0-based, frequency excluded) for a variable."""
    for idx, vname, _ in rf.variables:
        if idx == 0:
            continue
        if vname.lower() == name.lower():
            return idx - 1
    names = [v[1] for v in rf.variables if v[0] != 0]
    raise KeyError(f"Variable '{name}' not found. Available: {names}")


# ---------------------------------------------------------------------------
# CSV loader — auto-detects Re/Im or dB+phase formats
# ---------------------------------------------------------------------------

def load_csv_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency (Hz) and H (complex) from a measurement CSV.

    Two formats are auto-detected:

    Format A — Re/Im (e.g. data_inductor_Z.csv, data_two_rlc_Z.csv):
        Frequency[MHz],  Re[Z_Ohm],  Im[Z_Ohm]

    Format B — Magnitude dB + Phase deg (e.g. data_filter_S21.csv):
        Frequency[GHz],  |S21|[dB],  Phase[deg]
    """
    import csv as _csv

    def _find(headers: list[str], *keywords: str) -> str | None:
        """Return first header whose lower-case form contains ALL keywords."""
        for h in headers:
            hl = h.lower()
            if all(k in hl for k in keywords):
                return h
        return None

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader   = _csv.DictReader(fh)
        headers  = list(reader.fieldnames or [])

        # ── detect frequency column + unit ──────────────────────────────────
        col_f = (_find(headers, "freq", "ghz")
                 or _find(headers, "freq", "mhz")
                 or _find(headers, "freq"))
        if col_f is None:
            raise KeyError(f"No frequency column found in {headers}")

        freq_unit_mhz = 1e3 if "ghz" in col_f.lower() else 1.0   # GHz→MHz=×1000

        # ── detect data format ───────────────────────────────────────────────
        col_re = _find(headers, "re[")          # "Re[Z_Ohm]"
        col_im = _find(headers, "im[")          # "Im[Z_Ohm]"
        col_db = (_find(headers, "db")
                  or _find(headers, "|s", "db")
                  or _find(headers, "mag"))      # "|S21|[dB]"
        col_ph = (_find(headers, "phase")
                  or _find(headers, "angle"))    # "Phase[deg]"

        use_ri = (col_re is not None and col_im is not None)
        use_db = (col_db is not None and col_ph is not None)

        if not use_ri and not use_db:
            raise KeyError(
                f"Cannot detect Re/Im or dB+Phase columns in {headers}.\n"
                f"Expected: Re[...]/Im[...] OR |...|[dB]/Phase[deg]."
            )

        freq_hz: list[float] = []
        H: list[complex] = []

        for row in reader:
            f_mhz = float(row[col_f]) * freq_unit_mhz
            freq_hz.append(f_mhz * 1e6)

            if use_ri:
                re = float(row[col_re])
                im = float(row[col_im])
                H.append(complex(re, im))
            else:
                db_val  = float(row[col_db])
                ph_deg  = float(row[col_ph])
                mag     = 10 ** (db_val / 20.0)
                ph_rad  = np.deg2rad(ph_deg)
                H.append(mag * np.exp(1j * ph_rad))

    return np.array(freq_hz), np.array(H)


# ---------------------------------------------------------------------------
# Bode comparison plot
# ---------------------------------------------------------------------------

def plot_bode_comparison(
    out_png: Path,
    foster_path: Path | None = None,
    behavioral_path: Path | None = None,
    csv_path: Path | None = None,
    foster_var: str = "V(1)",
    behavioral_var: str = "V(out)",
    title: str = "LTspice Verification — Foster RLC vs Behavioral Laplace",
    max_points: int = 2000,
    ylabel: str = "|Z(jω)|  (dB)",
) -> None:
    """Load .raw files and produce a Bode comparison plot.

    Parameters
    ----------
    out_png:
        Output PNG file path.
    foster_path:
        Path to Foster testbench .raw (optional — skip if None).
    behavioral_path:
        Path to Behavioral testbench .raw (optional — skip if None).
    csv_path:
        Path to measurement CSV for overlay (optional — skip if None or missing).
    foster_var:
        LTspice variable name for impedance in Foster raw file.
        Default ``"V(1)"`` — used as Z = V(1) / (-I(Vin)).
    behavioral_var:
        LTspice variable name in Behavioral raw file.
        Default ``"V(out)"`` — direct H(s)*1V output.
    title:
        Plot super-title.
    max_points:
        Downsample large .raw files to this many points.
    ylabel:
        Y-axis label for magnitude subplot (e.g. '|Z| (dB Ω)' or '|H| (dB)').
    """
    Z_foster = freq_f = None
    Z_beh    = freq_b = None

    if foster_path is not None and foster_path.exists():
        print(f"Loading {foster_path.name}  ...", end=" ", flush=True)
        rf_f = load_raw(foster_path, max_points=max_points)
        print(f"OK  ({rf_f.n_points} points, {len(rf_f.variables)} vars)")
        v1        = rf_f.data[_var_index(rf_f, foster_var)]
        ivin_f    = rf_f.data[_var_index(rf_f, "I(Vin)")]
        Z_foster  = v1 / (-ivin_f)
        freq_f    = rf_f.freq
    elif foster_path is not None:
        print(f"Warning: Foster .raw not found: {foster_path}  (skipping Foster trace)")

    if behavioral_path is not None and behavioral_path.exists():
        print(f"Loading {behavioral_path.name}  ...", end=" ", flush=True)
        rf_b = load_raw(behavioral_path, max_points=max_points)
        print(f"OK  ({rf_b.n_points} points, {len(rf_b.variables)} vars)")
        Z_beh  = rf_b.data[_var_index(rf_b, behavioral_var)]
        freq_b = rf_b.freq
    elif behavioral_path is not None:
        print(f"Warning: Behavioral .raw not found: {behavioral_path}  (skipping)")

    if Z_foster is None and Z_beh is None:
        raise FileNotFoundError(
            "No .raw files found.  Run LTspice on the testbench first."
        )

    freq_csv = Z_csv = None
    if csv_path is not None and csv_path.exists():
        print(f"Loading {csv_path.name}  ...", end=" ", flush=True)
        freq_csv, Z_csv = load_csv_data(csv_path)
        print(f"OK  ({len(freq_csv)} points)")
    elif csv_path is not None:
        print(f"Note: CSV not found: {csv_path}  (skipping measurement overlay)")

    # ── Build subplots ──────────────────────────────────────────────────────
    C_MEAS   = "#1a9641"
    C_FOSTER = "#d7191c"
    C_BEH    = "#2c7bb6"

    has_csv  = Z_csv is not None
    nrows    = 3 if has_csv else 2
    h_ratios = [3, 1.5, 2] if has_csv else [3, 2]

    fig, axes = plt.subplots(
        nrows, 1, figsize=(12, 8 if has_csv else 7), sharex=True,
        gridspec_kw={"height_ratios": h_ratios},
    )
    ax_mag = axes[0]
    ax_err = axes[1] if has_csv else None
    ax_ph  = axes[-1]

    fig.suptitle(title, fontsize=11, fontweight="bold")

    # ── Magnitude ───────────────────────────────────────────────────────────
    if Z_csv is not None:
        mag_csv_db = 20 * np.log10(np.abs(Z_csv) + 1e-300)
        ax_mag.semilogx(freq_csv, mag_csv_db, color=C_MEAS, lw=2.5, zorder=3,
                        label="Measurement data")

    if Z_foster is not None:
        mag_f_db = 20 * np.log10(np.abs(Z_foster) + 1e-300)
        ax_mag.semilogx(freq_f, mag_f_db, color=C_FOSTER, lw=1.8, zorder=2, ls="--",
                        label="Foster RLC (LTspice)")

    if Z_beh is not None:
        mag_b_db = 20 * np.log10(np.abs(Z_beh) + 1e-300)
        ax_mag.semilogx(freq_b, mag_b_db, color=C_BEH, lw=1.5, zorder=2, ls=":",
                        label="Behavioral Laplace (LTspice)")

    ax_mag.set_ylabel(ylabel, fontsize=10)
    ax_mag.legend(fontsize=9)
    ax_mag.grid(True, which="both", color="#cccccc", lw=0.5)
    ax_mag.spines["top"].set_visible(False)
    ax_mag.spines["right"].set_visible(False)
    ax_mag.tick_params(labelsize=9)

    # ── Error vs measurement ─────────────────────────────────────────────────
    if has_csv and ax_err is not None:
        if Z_foster is not None:
            Zf_i = (np.interp(freq_csv, freq_f, Z_foster.real)
                    + 1j * np.interp(freq_csv, freq_f, Z_foster.imag))
            err_f_db = (20 * np.log10(np.abs(Zf_i) + 1e-300)) - mag_csv_db
            ax_err.semilogx(freq_csv, err_f_db, color=C_FOSTER, lw=1.5, ls="--",
                            label="Foster error")
        if Z_beh is not None:
            Zb_i = (np.interp(freq_csv, freq_b, Z_beh.real)
                    + 1j * np.interp(freq_csv, freq_b, Z_beh.imag))
            err_b_db = (20 * np.log10(np.abs(Zb_i) + 1e-300)) - mag_csv_db
            ax_err.semilogx(freq_csv, err_b_db, color=C_BEH, lw=1.5, ls=":",
                            label="Behavioral error")
        ax_err.axhline(0, color="#888", lw=0.8)
        ax_err.set_ylabel("Error  (dB)", fontsize=10)
        ax_err.legend(fontsize=8, loc="upper left")
        ax_err.grid(True, which="both", color="#cccccc", lw=0.5)
        ax_err.spines["top"].set_visible(False)
        ax_err.spines["right"].set_visible(False)
        ax_err.tick_params(labelsize=9)

    # ── Phase ────────────────────────────────────────────────────────────────
    if Z_csv is not None:
        ph_csv_deg = np.degrees(np.unwrap(np.angle(Z_csv)))
        ax_ph.semilogx(freq_csv, ph_csv_deg, color=C_MEAS, lw=2.5, zorder=3)
    if Z_foster is not None:
        ph_f_deg = np.degrees(np.unwrap(np.angle(Z_foster)))
        ax_ph.semilogx(freq_f, ph_f_deg, color=C_FOSTER, lw=1.8, zorder=2, ls="--")
    if Z_beh is not None:
        ph_b_deg = np.degrees(np.unwrap(np.angle(Z_beh)))
        ax_ph.semilogx(freq_b, ph_b_deg, color=C_BEH, lw=1.5, zorder=2, ls=":")

    ax_ph.set_ylabel("Phase  (degrees)", fontsize=10)
    ax_ph.set_xlabel("Frequency  (Hz)", fontsize=10)
    ax_ph.grid(True, which="both", color="#cccccc", lw=0.5)
    ax_ph.spines["top"].set_visible(False)
    ax_ph.spines["right"].set_visible(False)
    ax_ph.tick_params(labelsize=9)

    # Nice frequency tick labels
    fmt = mticker.FuncFormatter(
        lambda x, _: (
            f"{x/1e9:.3g} GHz" if x >= 1e9
            else f"{x/1e6:.3g} MHz" if x >= 1e6
            else f"{x/1e3:.3g} kHz" if x >= 1e3
            else f"{x:.3g} Hz"
        )
    )
    ax_mag.xaxis.set_major_formatter(fmt)
    ax_ph.xaxis.set_major_formatter(fmt)
    if ax_err is not None:
        ax_err.xaxis.set_major_formatter(fmt)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    base = _HERE / "examples"

    parser = argparse.ArgumentParser(
        description="Plot LTspice AC Bode: Foster RLC vs Behavioral Laplace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Built-in presets (--preset NAME):",
            *[f"  {k:12s}  {v.title[:60]}" for k, v in PRESETS.items()],
            "",
            "Examples:",
            "  python plot_ltspice_bode.py --all                   # plot every available preset",
            "  python plot_ltspice_bode.py --preset inductor",
            "  python plot_ltspice_bode.py --preset filter",
            "  python plot_ltspice_bode.py --foster tb.raw --behavioral tb_beh.raw",
        ]),
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), default=None,
        help="Named study preset — sets all default paths automatically.",
    )
    parser.add_argument(
        "--foster", type=Path, default=None,
        help="Foster testbench .raw file (overrides preset).",
    )
    parser.add_argument(
        "--behavioral", type=Path, default=None,
        help="Behavioral testbench .raw file (overrides preset).",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Measurement CSV for overlay (Re/Im or dB+deg, auto-detected).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG file path.",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Plot super-title (overrides preset).",
    )
    parser.add_argument(
        "--foster-var", type=str, default="V(1)",
        help="LTspice probe variable for Foster impedance (default: V(1)).",
    )
    parser.add_argument(
        "--behavioral-var", type=str, default="V(out)",
        help="LTspice probe variable for Behavioral output (default: V(out)).",
    )
    parser.add_argument(
        "--ylabel", type=str, default=None,
        help="Y-axis label override (default: '|Z(jω)|  (dB)').",
    )
    parser.add_argument(
        "--max-points", type=int, default=2000,
        help="Max points per trace (downsamples large .raw files).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all presets and generate one PNG per study.",
    )
    args = parser.parse_args()


    def _run_preset(name: str, p: _Preset) -> None:
        """Plot one preset; skip silently if no .raw files exist."""
        foster_p = Path(p.foster)     if p.foster     else None
        beh_p    = Path(p.behavioral) if p.behavioral else None
        csv_p    = Path(p.csv)        if p.csv        else None

        # Skip preset entirely when neither raw file exists
        has_foster = foster_p is not None and foster_p.exists()
        has_beh    = beh_p    is not None and beh_p.exists()
        if not has_foster and not has_beh:
            print(f"[{name}] No .raw files found — skipping "
                  f"(run LTspice on the testbench first)")
            return

        out_p  = Path(p.out)
        ylabel = "|H(jω)|  (dB)" if csv_p and "s21" in str(csv_p).lower() \
                 else "|Z(jω)|  (dB Ω)"

        print(f"\n{'='*60}")
        print(f"Preset: {name}")
        plot_bode_comparison(
            out_png         = out_p,
            foster_path     = foster_p,
            behavioral_path = beh_p,
            csv_path        = csv_p,
            title           = p.title,
            max_points      = args.max_points,
            ylabel          = ylabel,
        )


    if args.all:
        # ── Run every preset ────────────────────────────────────────────────
        print(f"Running all {len(PRESETS)} presets ...\n")
        skipped = 0
        for name, p in PRESETS.items():
            foster_p = Path(p.foster)     if p.foster     else None
            beh_p    = Path(p.behavioral) if p.behavioral else None
            if (foster_p is None or not foster_p.exists()) and \
               (beh_p    is None or not beh_p.exists()):
                skipped += 1
                print(f"[{name:12s}] skipped — no .raw files")
            else:
                _run_preset(name, p)
        print(f"\nDone. ({len(PRESETS) - skipped} plotted, {skipped} skipped)")

    else:
        # ── Single preset / custom paths ────────────────────────────────────
        preset = PRESETS.get(args.preset) if args.preset else PRESETS["two_rlc"]

        foster_path     = args.foster     or (Path(preset.foster)     if preset.foster     else None)
        behavioral_path = args.behavioral or (Path(preset.behavioral) if preset.behavioral else None)
        csv_path        = args.csv        or (Path(preset.csv)        if preset.csv        else None)
        out_png         = args.out        or Path(preset.out)
        title           = args.title      or preset.title

        if args.ylabel:
            ylabel = args.ylabel
        elif csv_path and "s21" in str(csv_path).lower():
            ylabel = "|H(jω)|  (dB)"
        else:
            ylabel = "|Z(jω)|  (dB Ω)"

        plot_bode_comparison(
            out_png         = out_png,
            foster_path     = foster_path,
            behavioral_path = behavioral_path,
            csv_path        = csv_path,
            foster_var      = args.foster_var,
            behavioral_var  = args.behavioral_var,
            title           = title,
            max_points      = args.max_points,
            ylabel          = ylabel,
        )
