# Vector Fitting Analysis for RLC & S-Domain Systems

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

## Overview

A Python-based tool for **Vector Fitting (VF)** of frequency-domain data to rational transfer functions expressed in pole-zero form. Designed for:

- RLC circuit identification from impedance/admittance measurements
- S-domain system identification (poles and zeros)
- Broadband macromodel generation
- Passive component modeling

---

## Features

- **Vector Fitting Algorithm** (Sanathanan-Koerner / Gustavsen-Semlyen method)
- **Pole-Zero Extraction** from fitted rational functions
- **RLC Network Synthesis** from rational models
- **Stability enforcement** (passivity check & correction)
- **Interactive plots**: Bode, Nyquist, pole-zero map
- **Export**: SPICE subcircuit netlist, Touchstone (.s1p/.s2p), MATLAB `.mat`

---

## Project Structure

```
vector_fitting/
├── README.md                  # This file
├── CLAUDE.md                  # AI assistant context and conventions
├── PLAN.md                    # Development roadmap
├── requirements.txt           # Python dependencies
├── setup.py / pyproject.toml  # Package configuration
│
├── src/
│   └── vfit/
│       ├── __init__.py
│       ├── core/
│       │   ├── vector_fitting.py      # Main VF algorithm
│       │   ├── pole_zero.py           # Pole/zero extraction & manipulation
│       │   ├── rational_function.py   # H(s) = K * prod(s-zi)/prod(s-pi)
│       │   └── residues.py            # Partial fraction expansion
│       │
│       ├── rlc/
│       │   ├── rlc_synthesis.py       # Fit → RLC network synthesis
│       │   ├── impedance.py           # Z(jω) modeling
│       │   └── admittance.py          # Y(jω) modeling
│       │
│       ├── solvers/
│       │   ├── least_squares.py       # LS problem setup
│       │   ├── passivity.py           # Passivity enforcement
│       │   └── stability.py           # Pole stabilization
│       │
│       ├── export/
│       │   ├── spice_export.py        # SPICE netlist generator
│       │   ├── touchstone.py          # Touchstone file I/O
│       │   └── matlab_export.py       # MATLAB .mat export
│       │
│       ├── visualization/
│       │   ├── bode_plot.py           # Magnitude/phase Bode plots
│       │   ├── nyquist_plot.py        # Nyquist diagram
│       │   ├── pole_zero_map.py       # S-plane pole/zero plot
│       │   └── convergence_plot.py    # VF iteration convergence
│       │
│       └── utils/
│           ├── frequency.py           # Frequency axis helpers
│           ├── data_loader.py         # Load CSV / Touchstone / raw data
│           └── validators.py          # Input validation
│
├── tests/
│   ├── test_vector_fitting.py
│   ├── test_pole_zero.py
│   ├── test_rlc_synthesis.py
│   └── fixtures/                      # Test data files
│
├── examples/
│   ├── 01_simple_rlc.py               # Basic RLC impedance fit
│   ├── 02_highorder_filter.py         # High-order filter identification
│   ├── 03_multiport.py                # 2-port S-parameter fitting
│   ├── 04_spice_export.py             # Export to SPICE
│   └── notebooks/
│       ├── intro_vector_fitting.ipynb
│       └── rlc_synthesis_demo.ipynb
│
└── docs/
    ├── theory.md                      # Mathematical background
    ├── api_reference.md               # Full API docs
    └── examples.md                    # Annotated example walkthroughs
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/yourname/vector-fitting.git
cd vector-fitting
pip install -e ".[dev]"
```

### Basic Usage

```python
import numpy as np
from vfit import VectorFitter
from vfit.visualization import bode_plot, pole_zero_map

# 1. Define frequency data (Hz) and complex response H(jω)
freq = np.logspace(3, 9, 200)          # 1 kHz – 1 GHz
H_meas = load_your_data(freq)          # complex array

# 2. Run Vector Fitting (N=10 poles)
vf = VectorFitter(n_poles=10, mode='auto')
model = vf.fit(freq, H_meas)

# 3. Inspect poles and zeros
print("Poles:", model.poles)
print("Zeros:", model.zeros)
print("Residues:", model.residues)

# 4. Visualize
bode_plot(freq, H_meas, model)
pole_zero_map(model)

# 5. Export to SPICE
model.export_spice("output/rlc_model.cir")
```

---

## Mathematical Background

The rational transfer function is expressed as:

```
         K · Π(s - zᵢ)
H(s) = ─────────────────
           Π(s - pᵢ)
```

Vector Fitting iteratively solves for poles `pᵢ`, zeros `zᵢ`, and gain `K` by:
1. Starting with initial pole set (logarithmically spaced on the imaginary axis)
2. Solving a weighted least-squares problem for residues
3. Relocating poles using the zeros of the fitted denominator
4. Repeating until convergence (typically 3–10 iterations)

See [`docs/theory.md`](docs/theory.md) for full derivation.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Core numerical operations |
| `scipy` | Least squares, signal processing |
| `matplotlib` | Plotting |
| `pandas` | Data I/O |
| `scikit-rf` | Touchstone file support |
| `pytest` | Testing |

---

## References

1. Gustavsen & Semlyen, "Rational approximation of frequency domain responses by vector fitting," *IEEE Trans. Power Delivery*, 1999.
2. Gustavsen, "Improving the pole relocating properties of vector fitting," *IEEE Trans. Power Delivery*, 2006.
3. Deschrijver et al., "Macromodeling of Multiport Systems Using a Fast Implementation of the Vector Fitting Method," *IEEE MWCL*, 2008.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
