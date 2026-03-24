"""
vfit/utils/data_loader.py
─────────────────────────
Load measured frequency-domain data from common file formats into
a normalised (freq_hz, H_complex) pair ready for VectorFitter.

Supported formats
─────────────────
  CSV / TSV  — flexible column-based loader (load_csv / load_ri_csv)
  Touchstone — .s1p / .s2p via scikit-rf  (load_touchstone)

All loaders return a MeasurementData dataclass with:
    freq_hz  : np.ndarray  (N,)   frequency in Hz, strictly positive, sorted
    H        : np.ndarray  (N,)   complex response H(j·2π·f)
    label    : str                human-readable description
    source   : str                original file path

Frequency unit auto-detection
──────────────────────────────
If the maximum value in the frequency column is:
  > 1e8   → assumed Hz   (already correct)
  > 1e5   → assumed MHz  (×1e6)
  > 1e2   → assumed MHz  (×1e6)   common in VNA exports
  ≤ 1e2   → assumed GHz  (×1e9)
You can always override with the `freq_unit` parameter.

CSV column conventions
──────────────────────
load_csv()   — one complex column as magnitude+phase or real+imag
load_ri_csv() — two explicit columns: Re(H) and Im(H)

Typical VNA export (CSV, tab-separated, MHz):
    Frequency[MHz]   Re[Z]   Im[Z]
    100              48.3    2.1
    ...

Touchstone (.s1p):
    !  Keysight VNA export
    # MHz S MA R 50
    100   0.99  -1.2
    ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeasurementData:
    """
    Normalised measurement ready for VectorFitter.

    Attributes
    ----------
    freq_hz : ndarray (N,)
        Frequency in Hz, sorted ascending, strictly positive.
    H : ndarray complex (N,)
        Complex frequency response H(j·2π·f).
    label : str
        Human-readable description (e.g. 'Z11 — component_A.csv').
    source : str
        Original file path.
    notes : list[str]
        Any warnings or auto-detection messages recorded during loading.
    """
    freq_hz : np.ndarray
    H       : np.ndarray
    label   : str = "measurement"
    source  : str = ""
    notes   : list[str] = field(default_factory=list)

    def __post_init__(self):
        self.freq_hz = np.asarray(self.freq_hz, dtype=float)
        self.H       = np.asarray(self.H,       dtype=complex)
        if self.freq_hz.shape != self.H.shape:
            raise ValueError(
                f"freq_hz and H must have the same length: "
                f"{self.freq_hz.shape} vs {self.H.shape}"
            )

    @property
    def n_points(self) -> int:
        return len(self.freq_hz)

    @property
    def freq_range_str(self) -> str:
        lo, hi = self.freq_hz[0], self.freq_hz[-1]
        def _fmt(f):
            if f >= 1e9:  return f"{f/1e9:.3g} GHz"
            if f >= 1e6:  return f"{f/1e6:.3g} MHz"
            if f >= 1e3:  return f"{f/1e3:.3g} kHz"
            return f"{f:.3g} Hz"
        return f"{_fmt(lo)} – {_fmt(hi)}"

    def summary(self) -> str:
        lines = [
            f"MeasurementData: {self.label}",
            f"  Source     : {self.source}",
            f"  Points     : {self.n_points}",
            f"  Freq range : {self.freq_range_str}",
            f"  |H| range  : {np.abs(self.H).min():.3e} – {np.abs(self.H).max():.3e}",
        ]
        for note in self.notes:
            lines.append(f"  Note       : {note}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency unit helpers
# ─────────────────────────────────────────────────────────────────────────────

_UNIT_SCALE = {
    "hz":  1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}

def _resolve_freq_scale(raw_freq: np.ndarray, freq_unit: Optional[str]) -> tuple[float, str]:
    """
    Return (scale_factor, detected_unit_string).
    If freq_unit is given explicitly, use it.
    Otherwise auto-detect from the magnitude of raw_freq values.
    """
    if freq_unit is not None:
        key = freq_unit.lower().strip()
        if key not in _UNIT_SCALE:
            raise ValueError(
                f"Unknown freq_unit '{freq_unit}'. "
                f"Choose from: {list(_UNIT_SCALE.keys())}"
            )
        return _UNIT_SCALE[key], freq_unit.upper()

    fmax = float(raw_freq.max())
    if fmax > 1e8:
        return 1.0,   "Hz  (auto)"
    if fmax > 1e5:
        return 1e6,   "MHz (auto)"
    if fmax > 1e2:
        return 1e6,   "MHz (auto)"
    return 1e9, "GHz (auto)"


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader — Real + Imaginary columns
# ─────────────────────────────────────────────────────────────────────────────

def load_ri_csv(
    path: str | Path,
    freq_col: int | str = 0,
    re_col:   int | str = 1,
    im_col:   int | str = 2,
    freq_unit: Optional[str] = None,
    delimiter: Optional[str] = None,
    skip_rows: int = 0,
    label: Optional[str] = None,
) -> MeasurementData:
    """
    Load a CSV/TSV with explicit Re(H) and Im(H) columns.

    Parameters
    ----------
    path : str or Path
        Path to the file.
    freq_col, re_col, im_col : int or str
        Column index (0-based) or header name for frequency, real, imaginary.
    freq_unit : str, optional
        'Hz', 'kHz', 'MHz', or 'GHz'.  Auto-detected if None.
    delimiter : str, optional
        Column delimiter.  Auto-detected (comma/tab/space) if None.
    skip_rows : int
        Number of header rows to skip (auto-detected if 0).
    label : str, optional
        Human label for the dataset.  Defaults to the filename.

    Returns
    -------
    MeasurementData

    Examples
    --------
    # Frequency[MHz], Re[Z], Im[Z]
    >>> data = load_ri_csv("impedance.csv", freq_unit="MHz")

    # Tab-separated, header on row 0:  freq   real   imag
    >>> data = load_ri_csv("vna_export.txt", freq_col="freq",
    ...                     re_col="real", im_col="imag")
    """
    path  = Path(path)
    notes = []

    raw_text = path.read_text(errors="replace")
    delim    = delimiter or _sniff_delimiter(raw_text)
    header, skip = _sniff_header(raw_text, delim, skip_rows)

    import csv, io
    reader = csv.reader(io.StringIO(raw_text), delimiter=delim)
    rows   = list(reader)

    # Resolve column indices from names if needed
    fi = _col_idx(freq_col, header)
    ri = _col_idx(re_col,   header)
    ii = _col_idx(im_col,   header)

    freq_raw, re_vals, im_vals = [], [], []
    for row in rows[skip:]:
        if not row or row[0].startswith(("#", "!", "%")):
            continue
        try:
            freq_raw.append(float(row[fi]))
            re_vals.append(float(row[ri]))
            im_vals.append(float(row[ii]))
        except (ValueError, IndexError):
            continue

    freq_raw = np.array(freq_raw)
    H        = np.array(re_vals) + 1j * np.array(im_vals)

    scale, unit_str = _resolve_freq_scale(freq_raw, freq_unit)
    notes.append(f"Frequency unit detected/used: {unit_str}")
    freq_hz = freq_raw * scale

    freq_hz, H = _sort_and_validate(freq_hz, H, notes)

    return MeasurementData(
        freq_hz = freq_hz,
        H       = H,
        label   = label or path.name,
        source  = str(path),
        notes   = notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV loader — Magnitude + Phase columns
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(
    path: str | Path,
    freq_col:  int | str = 0,
    mag_col:   int | str = 1,
    phase_col: int | str = 2,
    mag_unit:  str = "linear",
    phase_unit: str = "deg",
    freq_unit: Optional[str] = None,
    delimiter: Optional[str] = None,
    label: Optional[str] = None,
) -> MeasurementData:
    """
    Load a CSV with magnitude + phase columns.

    Parameters
    ----------
    path : str or Path
    freq_col, mag_col, phase_col : int or str
        Column index or header name.
    mag_unit : 'linear' or 'dB'
        Whether magnitude column is linear or dB.
    phase_unit : 'deg' or 'rad'
        Whether phase column is in degrees or radians.
    freq_unit : str, optional
        'Hz', 'kHz', 'MHz', 'GHz'.  Auto-detected if None.
    delimiter : str, optional
    label : str, optional

    Returns
    -------
    MeasurementData

    Examples
    --------
    # Frequency[GHz], |S11|[dB], Phase[deg]
    >>> data = load_csv("s11.csv", mag_unit="dB", freq_unit="GHz")
    """
    path  = Path(path)
    notes = []

    raw_text = path.read_text(errors="replace")
    delim    = delimiter or _sniff_delimiter(raw_text)
    header, skip = _sniff_header(raw_text, delim, 0)

    import csv, io
    reader = csv.reader(io.StringIO(raw_text), delimiter=delim)
    rows   = list(reader)

    fi = _col_idx(freq_col,  header)
    mi = _col_idx(mag_col,   header)
    pi = _col_idx(phase_col, header)

    freq_raw, mags, phases = [], [], []
    for row in rows[skip:]:
        if not row or row[0].startswith(("#", "!", "%")):
            continue
        try:
            freq_raw.append(float(row[fi]))
            mags.append(float(row[mi]))
            phases.append(float(row[pi]))
        except (ValueError, IndexError):
            continue

    freq_raw = np.array(freq_raw)
    mags     = np.array(mags)
    phases   = np.array(phases)

    # Convert units
    if mag_unit == "dB":
        mags = 10 ** (mags / 20.0)
        notes.append("Magnitude converted from dB to linear.")
    if phase_unit == "deg":
        phases = np.radians(phases)

    H = mags * np.exp(1j * phases)

    scale, unit_str = _resolve_freq_scale(freq_raw, freq_unit)
    notes.append(f"Frequency unit detected/used: {unit_str}")
    freq_hz = freq_raw * scale

    freq_hz, H = _sort_and_validate(freq_hz, H, notes)

    return MeasurementData(
        freq_hz = freq_hz,
        H       = H,
        label   = label or path.name,
        source  = str(path),
        notes   = notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Touchstone loader (.s1p / .s2p)
# ─────────────────────────────────────────────────────────────────────────────

def load_touchstone(
    path: str | Path,
    port: tuple[int, int] = (1, 1),
    label: Optional[str] = None,
) -> MeasurementData:
    """
    Load a Touchstone file (.s1p or .s2p) via scikit-rf.

    Parameters
    ----------
    path : str or Path
        Path to .s1p or .s2p file.
    port : tuple (i, j), optional
        Which S-parameter to extract (1-indexed, default S11).
        Ignored for .s1p (always S11).
    label : str, optional

    Returns
    -------
    MeasurementData

    Raises
    ------
    ImportError  if scikit-rf is not installed.
    ValueError   if the requested port is out of range.

    Examples
    --------
    >>> data = load_touchstone("filter.s2p", port=(1, 2))  # S21
    """
    try:
        import skrf
    except ImportError:
        raise ImportError(
            "scikit-rf is required for Touchstone loading. "
            "Install it with:  pip install scikit-rf"
        )

    path  = Path(path)
    notes = []

    nw    = skrf.Network(str(path))
    freq_hz = nw.f  # already in Hz

    i, j = port[0] - 1, port[1] - 1   # convert to 0-indexed
    if i >= nw.s.shape[1] or j >= nw.s.shape[2]:
        raise ValueError(
            f"Port ({port[0]},{port[1]}) out of range for "
            f"{nw.s.shape[1]}-port network."
        )

    H = nw.s[:, i, j]
    param_name = f"S{port[0]}{port[1]}"
    notes.append(f"Extracted {param_name} from {path.name}")

    freq_hz, H = _sort_and_validate(freq_hz, H, notes)

    return MeasurementData(
        freq_hz = freq_hz,
        H       = H,
        label   = label or f"{param_name} — {path.name}",
        source  = str(path),
        notes   = notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sniff_delimiter(text: str) -> str:
    """Detect the column delimiter from the first non-comment data line."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!", "%")):
            continue
        if "\t" in line:
            return "\t"
        if "," in line:
            return ","
        return " "
    return ","


def _sniff_header(text: str, delim: str, forced_skip: int) -> tuple[list[str], int]:
    """
    Return (header_names, n_rows_to_skip).
    header_names is empty if no header row is found.
    """
    if forced_skip > 0:
        return [], forced_skip

    import csv, io
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    for i, row in enumerate(reader):
        if not row or row[0].startswith(("#", "!", "%")):
            continue
        # If first token is not a number, treat as header
        try:
            float(row[0].strip())
            return [], i          # numeric → no header, skip to this row
        except ValueError:
            return [c.strip() for c in row], i + 1   # string → header row
    return [], 0


def _col_idx(col: int | str, header: list[str]) -> int:
    """Resolve a column specifier (int index or header name) to an int index."""
    if isinstance(col, int):
        return col
    if not header:
        raise ValueError(
            f"Column name '{col}' given but no header row was detected. "
            "Use integer column indices or set skip_rows manually."
        )
    try:
        # Case-insensitive, strip whitespace
        return [h.lower() for h in header].index(col.lower().strip())
    except ValueError:
        raise ValueError(
            f"Column '{col}' not found in header: {header}"
        )


def _sort_and_validate(
    freq_hz: np.ndarray,
    H: np.ndarray,
    notes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by frequency, remove non-positive or non-finite points."""
    # Sort
    order   = np.argsort(freq_hz)
    freq_hz = freq_hz[order]
    H       = H[order]

    # Remove bad points
    mask = (
        np.isfinite(freq_hz) &
        np.isfinite(H.real)  &
        np.isfinite(H.imag)  &
        (freq_hz > 0)
    )
    n_bad = int((~mask).sum())
    if n_bad > 0:
        notes.append(f"Removed {n_bad} non-finite or non-positive-frequency points.")
    freq_hz = freq_hz[mask]
    H       = H[mask]

    if len(freq_hz) == 0:
        raise ValueError("No valid data points remain after filtering.")

    return freq_hz, H
