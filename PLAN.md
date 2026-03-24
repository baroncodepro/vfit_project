# PLAN.md — Development Roadmap

## Vision

Build a robust, well-tested Python library for Vector Fitting of frequency-domain
data, targeting RF/power engineers who need:
1. Accurate rational approximations of measured S/Z/Y parameters
2. Pole-zero insight into system dynamics
3. Ready-to-simulate SPICE / Touchstone exports

---

## Phase 0 — Scaffolding (Week 1)

**Goal**: Runnable project skeleton with CI passing.

- [ ] `pyproject.toml` with dependencies, metadata, entry points
- [ ] `src/vfit/__init__.py` with clean public API surface
- [ ] `pytest` + `ruff` configured in `pyproject.toml`
- [ ] GitHub Actions CI: lint + test on push
- [ ] `CLAUDE.md`, `README.md`, `PLAN.md` (this file)
- [ ] Stub classes: `VectorFitter`, `RationalModel`

**Deliverable**: `pip install -e .` works; `pytest` collects (but skips) all tests.

---

## Phase 1 — Core Algorithm (Weeks 2–3)

**Goal**: Working scalar Vector Fitting implementation.

### 1.1 Least-Squares Problem Setup
- [ ] Build the VF system matrix `A` for given poles and frequency data
- [ ] Implement weighted LS solve using `scipy.linalg.lstsq`
- [ ] Extract residues `cᵢ` and direct term `d` from solution vector

### 1.2 Pole Relocation
- [ ] Compute zeros of fitted denominator polynomial σ(s)
- [ ] Enforce conjugate symmetry of new poles
- [ ] Stabilize: flip any RHP poles to LHP (mirror across imaginary axis)

### 1.3 Iteration Loop
- [ ] Initial pole placement: log-spaced imaginary pairs + real poles
- [ ] Convergence check: RMS error between H_fit and H_meas
- [ ] Max iteration guard + warning on non-convergence
- [ ] Store `rms_error_history` per iteration

### 1.4 RationalModel
- [ ] `poles`, `zeros`, `residues`, `gain`, `d_term` attributes
- [ ] `evaluate(freq)` → complex array
- [ ] `to_zpk()` → `scipy.signal.ZerosPolesGain`
- [ ] `__repr__` with pole/zero count and RMS error

**Tests**: 
- Simple 2nd-order RLC (known poles/zeros analytically)
- 4th-order Butterworth filter (known poles)
- Convergence in ≤ 10 iterations for smooth data

---

## Phase 2 — RLC Synthesis (Weeks 4–5)

**Goal**: Convert rational model to physical RLC element values.

### 2.1 Impedance Analysis
- [ ] Detect if response is Z(jω), Y(jω), or dimensionless H(jω)
- [ ] Identify Foster-I (series) vs Foster-II (parallel) topology from poles/residues

### 2.2 Foster Synthesis
- [ ] Real pole → R+L series branch (or R+C parallel)
- [ ] Complex conjugate pair → RLC resonator branch
- [ ] Direct term `d` → series resistance
- [ ] Constant term → shunt capacitance

### 2.3 Cauer Synthesis (optional stretch goal)
- [ ] Continued fraction expansion of Z(s)
- [ ] Ladder network element extraction

### 2.4 Validation
- [ ] Re-simulate synthesized RLC network in frequency domain
- [ ] Compare against original VF model (should match to numerical precision)

**Tests**:
- Synthesize a known R-L-C series circuit; verify element values
- Round-trip: fit → synthesize → re-simulate → compare to original data

---

## Phase 3 — Visualization (Week 6)

**Goal**: Informative, publication-quality plots.

- [ ] `bode_plot(freq, H_meas, model)` — magnitude (dB) + phase (°)
- [ ] `nyquist_plot(freq, H_meas, model)` — complex plane trajectory
- [ ] `pole_zero_map(model)` — s-plane with stability boundary
- [ ] `convergence_plot(model)` — RMS error vs iteration number
- [ ] All plots: measured (scatter) vs fitted (line), legend, grid

---

## Phase 4 — Export & I/O (Week 7)

**Goal**: Interoperability with industry tools.

### 4.1 SPICE Export
- [ ] Foster ladder netlist (`.cir`) with R, L, C elements
- [ ] Subcircuit wrapper with `SUBCKT` / `ENDS`
- [ ] Optional: behavioral `B`-source for direct rational function

### 4.2 Touchstone Import/Export
- [ ] Read `.s1p` / `.s2p` files (via `scikit-rf`)
- [ ] Write fitted model back as Touchstone

### 4.3 CSV / Raw Data Loader
- [ ] `load_csv(path, freq_col, re_col, im_col)` 
- [ ] Auto-detect frequency units (Hz / kHz / MHz / GHz)
- [ ] `load_touchstone(path)` wrapper

---

## Phase 5 — Advanced Features (Weeks 8–9)

**Goal**: Production-grade robustness features.

### 5.1 Passivity Enforcement
- [ ] Check passivity: Re[Z(jω)] ≥ 0 for all ω (for impedance)
- [ ] Passivity correction via Hamiltonian matrix perturbation (Gustavsen 2008)

### 5.2 Multi-Port / Matrix Fitting (stretch)
- [ ] Fit each element of S-parameter matrix independently
- [ ] Common pole set option (shared poles across matrix elements)

### 5.3 Automatic Order Selection
- [ ] Sweep `n_poles` from `n_min` to `n_max`
- [ ] Select order by AIC / BIC or user-specified RMS threshold
- [ ] Report trade-off curve (order vs RMS)

---

## Phase 6 — Documentation & Polish (Week 10)

- [ ] `docs/theory.md` — full mathematical derivation with LaTeX
- [ ] `docs/api_reference.md` — auto-generated from docstrings (Sphinx / pdoc)
- [ ] Jupyter notebooks: `intro_vector_fitting.ipynb`, `rlc_synthesis_demo.ipynb`
- [ ] PyPI-ready package (wheel + sdist)
- [ ] Badge: coverage ≥ 85%, all tests green

---

## Milestone Summary

| Milestone | Target | Key Deliverable |
|---|---|---|
| M0: Scaffold | Week 1 | Installable skeleton, CI green |
| M1: Core VF | Week 3 | Scalar VF converges on test cases |
| M2: RLC Synth | Week 5 | Foster synthesis with round-trip validation |
| M3: Plots | Week 6 | Bode, Nyquist, pole-zero map |
| M4: Export | Week 7 | SPICE netlist + Touchstone I/O |
| M5: Advanced | Week 9 | Passivity, auto order selection |
| M6: Docs | Week 10 | Notebooks + API docs + PyPI |

---

## Risk & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| VF diverges for noisy data | Medium | Add data smoothing pre-processing; use robust LS |
| RLC synthesis yields negative elements | Medium | Flag and warn; offer behavioral SPICE export fallback |
| Passivity enforcement breaks fit accuracy | Low | Make passivity correction optional; track error delta |
| Multi-port scope creep | High | Lock to scalar first; multi-port is explicit stretch goal |
