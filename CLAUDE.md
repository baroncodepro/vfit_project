# CLAUDE.md — AI Assistant Context for Vector Fitting Project

> This file gives Claude (or any AI coding assistant) the full context needed to
> contribute effectively to this codebase without repeated explanation.

---

## Project Purpose

This project implements **Vector Fitting (VF)** — a numerical algorithm that fits
a rational transfer function H(s) to measured frequency-domain data. The primary
applications are:

- **RLC impedance/admittance modeling**: fit Z(jω) or Y(jω) measurements
- **S-domain system identification**: extract poles and zeros from response data
- **SPICE macromodel generation**: export fitted rational functions as SPICE netlists

The output is always a rational function in pole-zero-gain (or residue) form.

---

## Core Mathematical Concepts

### Transfer Function Form

```
         K · Π(s - zᵢ)       N   cᵢ
H(s) = ───────────────── = d + Σ ──────
           Π(s - pᵢ)          i  s - pᵢ
```

- `pᵢ` — poles (may be complex; always appear in conjugate pairs for real systems)
- `zᵢ` — zeros
- `cᵢ` — residues (complex amplitudes for each pole)
- `d`  — direct term (real-valued)
- `K`  — gain

### Vector Fitting Algorithm (Gustavsen-Semlyen)

1. **Initialize**: Place N starting poles `{a₁, a₂, ..., aₙ}` on the imaginary axis,
   log-spaced across the frequency band of interest.
2. **LS Step**: Construct and solve the weighted least-squares problem:
   ```
   minimize  Σ_k |σ(jωₖ)·H(jωₖ) - H̃(jωₖ)|²
   ```
   where `σ(s)` is a rational weight function.
3. **Pole Relocation**: New poles = zeros of `σ(s)`.
4. **Residue Fit**: Solve for residues `cᵢ` with poles fixed.
5. **Iterate**: Repeat steps 2–4 until convergence (RMS error < tolerance).
6. **Stability enforcement**: Flip unstable poles to left half-plane if needed.

### S-Domain Conventions

- Frequency variable: `s = jω = j·2πf`
- Poles in the **left half-plane (LHP)**: stable system
- Poles in the **right half-plane (RHP)**: unstable — flag and correct
- Complex poles always appear as conjugate pairs: `p = α ± jβ`

---

## Architecture Decisions

| Decision | Choice | Reason |
|---|---|---|
| Pole format | `numpy` complex arrays | Native conjugate pairing support |
| LS solver | `scipy.linalg.lstsq` | Stable, handles rank-deficient cases |
| Complex conjugate pairs | Always enforce | Physical (real-valued) systems requirement |
| Passivity | Optional post-processing | Needed for passive RLC networks |
| Frequency input | Hz (not rad/s) internally | User-friendly; convert to ω inside algorithms |
| SPICE export | Foster/Cauer ladder synthesis | Most SPICE-compatible RLC topology |

---

## Key Modules — What Each Does

### `src/vfit/core/vector_fitting.py`
- **Class**: `VectorFitter`
- **Main method**: `fit(freq_hz, H_complex, n_poles, ...)`
- Runs the full VF iteration loop
- Returns a `RationalModel` object

### `src/vfit/core/rational_function.py`
- **Class**: `RationalModel`
- Stores poles, zeros, residues, gain, direct term
- Methods: `evaluate(s)`, `poles_zeros_gain()`, `to_zpk()`, `to_residues()`
- Wraps `scipy.signal.ZerosPolesGain`

### `src/vfit/core/pole_zero.py`
- Utilities for pole/zero manipulation
- `enforce_conjugate_pairs(poles)` — ensure conjugate symmetry
- `stabilize_poles(poles)` — flip RHP poles to LHP
- `sort_by_frequency(poles)` — sort by |Im(p)|

### `src/vfit/rlc/rlc_synthesis.py`
- Convert rational model → RLC ladder network
- `foster_synthesis(model)` → list of R, L, C element values
- `cauer_synthesis(model)` → continued fraction expansion
- Outputs element list suitable for SPICE export

### `src/vfit/export/spice_export.py`
- `to_spice_netlist(rlc_network, filename)` — write `.cir` file
- Uses `B` (behavioral) sources or ladder subcircuits

---

## Coding Conventions

### Python Style
- **Python 3.9+** required
- Follow **PEP 8** strictly
- Use **type hints** everywhere: `def fit(self, freq: np.ndarray, H: np.ndarray) -> RationalModel:`
- Docstrings: **NumPy style**

```python
def fit(self, freq: np.ndarray, H: np.ndarray, n_poles: int = 10) -> RationalModel:
    """
    Fit a rational function to frequency-domain data using Vector Fitting.

    Parameters
    ----------
    freq : np.ndarray, shape (N,)
        Frequency array in Hz.
    H : np.ndarray, shape (N,) complex
        Complex frequency response values H(j*2*pi*freq).
    n_poles : int, optional
        Number of poles in the rational approximation. Default is 10.

    Returns
    -------
    RationalModel
        Fitted rational function model with poles, zeros, and residues.

    Raises
    ------
    ValueError
        If freq and H have different lengths or n_poles < 1.
    """
```

### Numerical Conventions
- Frequency axes are always **Hz** at the public API; converted to `ω = 2πf` internally
- Complex data stored as `np.complex128` (double precision)
- Poles/zeros stored as `np.ndarray` of `complex`
- Conjugate pairs stored once (the pair is implicit); full list reconstructed when needed

### Testing
- Every public method has a unit test in `tests/`
- Use `pytest` with `numpy.testing.assert_allclose` for numerical comparisons
- Test fixtures in `tests/fixtures/` (CSV files with known-answer data)
- Minimum test coverage target: **80%**

---

## Common Tasks for AI Assistant

### Add a new export format
1. Create `src/vfit/export/new_format.py`
2. Implement `export(model: RationalModel, filename: str) -> None`
3. Register in `src/vfit/__init__.py`
4. Add tests in `tests/test_export_new_format.py`

### Add a new visualization
1. Create `src/vfit/visualization/new_plot.py`
2. Function signature: `def plot(freq, H_meas, model: RationalModel, ax=None, **kwargs)`
3. Always support `ax=None` → create new figure if not provided
4. Return `(fig, ax)` tuple

### Debug a bad fit
Ask: "The VF fit is diverging / has high RMS error. Check these:"
- Are initial poles log-spaced across the full frequency range?
- Is the data normalized (scale H to order-of-magnitude 1)?
- Is `n_poles` too low / too high?
- Are there sharp resonances requiring denser pole placement?
- Check `model.rms_error_history` for convergence plot

### Understand a pole-zero result
- **Real pole on negative real axis**: low-pass or decay mode
- **Complex conjugate pair**: resonance at `f₀ = |Im(p)| / (2π)`, Q = `|p| / (2·|Re(p)|)`
- **Zero on imaginary axis**: transmission null (anti-resonance)
- **RHP pole**: instability — should not appear after `stabilize_poles()`

---

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/
```

---

## Do NOT

- Do not use `np.matrix` — use `np.ndarray` only
- Do not hardcode frequency units — always document and convert
- Do not return bare tuples from public APIs — use dataclasses or named objects
- Do not silently ignore RHP poles — log a warning and enforce stability
- Do not mix rad/s and Hz without explicit conversion

---

## Glossary

| Term | Meaning |
|---|---|
| VF | Vector Fitting |
| RMS | Root mean square fitting error |
| LHP / RHP | Left / Right Half Plane of the s-domain |
| ZPK | Zeros-Poles-Gain representation |
| PFE | Partial Fraction Expansion |
| Foster | Series/parallel RLC ladder synthesis form |
| Cauer | Continued-fraction RLC ladder synthesis form |
| SNR | Signal-to-noise ratio of input data |
