# Theory: Vector Fitting for S-Domain Rational Approximation

## 1. Problem Statement

Given N samples of a frequency-domain response:

```
{(jω₁, H₁), (jω₂, H₂), ..., (jωN, HN)}
```

Find a rational function:

```
         cₘ        c₁         c₂
H(s) ≈ ──── + ────────── + ────────── + ... + d
         1     s - a₁      s - a₂
```

That minimizes the weighted least-squares error:

```
E = Σₖ wₖ · |H(jωₖ) - Ĥ(jωₖ)|²
```

where `wₖ` is a frequency-dependent weight (usually 1/|Hₖ| for log-scale accuracy).

---

## 2. Pole-Zero Representation

The equivalent pole-zero-gain form is:

```
         K · (s - z₁)(s - z₂)···(s - zₘ)
H(s) = ─────────────────────────────────────
              (s - p₁)(s - p₂)···(s - pₙ)
```

Converting between representations:
- **Poles** `pᵢ` ↔ roots of denominator polynomial
- **Zeros** `zᵢ` ↔ roots of numerator polynomial
- **Residues** via partial fraction: `cᵢ = lim(s→pᵢ) (s-pᵢ)·H(s)`

---

## 3. Vector Fitting Algorithm

### 3.1 Initialization

Place `n` starting poles `{a₁, ..., aₙ}` as conjugate pairs on the imaginary axis:

```
aᵢ = -α + j·ωᵢ,   aᵢ₊₁ = -α - j·ωᵢ      (complex pair)
```

where `α` is a small damping term (e.g., `α = ωᵢ / 100`) and `ωᵢ` are log-spaced
across the frequency band.

### 3.2 Weighted Least-Squares Formulation

Define sigma function (rational weight):

```
         n    dᵢ
σ(s) = 1 + Σ ──────
             s - aᵢ
```

The VF trick rewrites the approximation as:

```
σ(s) · H(s) = H̃(s)

where H̃(s) = Σ cᵢ/(s-aᵢ) + d
```

Evaluating at sample points and linearizing gives an overdetermined LS system:

```
[Φ_H  |  -Φ_σ·H] · [c; d; c̃; d̃]ᵀ  =  H
```

where `Φ` matrices are Cauchy/Loewner matrices evaluated at poles and frequencies.

Solve using `scipy.linalg.lstsq` (SVD-based, numerically stable).

### 3.3 Pole Relocation

New poles = zeros of σ(s). To find them:
1. Form companion matrix of denominator of σ(s)
2. Compute eigenvalues → new pole candidates `{ã₁, ..., ãₙ}`
3. Force conjugate symmetry; stabilize RHP poles

### 3.4 Residue Computation

With poles fixed at `{ãᵢ}`, solve the simpler LS problem for residues `{cᵢ}` and `d`.

### 3.5 Convergence

Iterate steps 3.2–3.4 until:

```
RMS = sqrt(Σ|H(jωₖ) - Ĥ(jωₖ)|² / N)  <  tolerance
```

Typical convergence: 3–10 iterations for well-conditioned data.

---

## 4. Stability Enforcement

After convergence, any poles `pᵢ` with `Re(pᵢ) > 0` (RHP) indicate instability.

**Correction**: Mirror to LHP: `p̃ᵢ = -Re(pᵢ) + j·Im(pᵢ)`

Then re-solve for residues with stabilized poles.

---

## 5. RLC Network Synthesis (Foster Forms)

Given a rational impedance model Z(s), synthesize as an RLC ladder.

### Foster-I (Series arm):

Each conjugate pole pair `p = -α ± jβ` corresponds to a parallel RLC:

```
R = 1/Re(c),   ω₀ = |p|,   Q = |Im(p)|/(2·|Re(p)|)
L = 1/(2·Re(c)),   C = 2·Re(c)/|p|²
```

### Real pole `p = -α` (no imaginary part):

```
R = 1/c,   L = 1/(c·α)
```

### Direct term `d`:

```
R_series = d
```

---

## 6. Key References

1. **Gustavsen & Semlyen (1999)** — Original Vector Fitting paper.  
   *IEEE Trans. Power Delivery*, 14(3), 1052–1061.

2. **Gustavsen (2006)** — Improved pole relocation.  
   *IEEE Trans. Power Delivery*, 21(3), 1587–1592.

3. **Deschrijver et al. (2008)** — Fast VF for multiport.  
   *IEEE MWCL*, 18(6), 383–385.

4. **Triverio et al. (2007)** — Stability and passivity enforcement.  
   *IEEE Trans. Advanced Packaging*, 30(4), 795–804.
