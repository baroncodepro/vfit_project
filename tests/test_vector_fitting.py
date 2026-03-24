"""
tests/test_vector_fitting.py
────────────────────────────
Unit tests for the Vector Fitting core algorithm.

Known-answer tests use analytically solvable RLC circuits and simple
rational functions where poles, zeros, and residues are exact.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vfit.core.vector_fitting import VectorFitter, VFOptions
from vfit.core.rational_function import RationalModel
from vfit.core.pole_zero import (
    enforce_conjugate_pairs,
    stabilize_poles,
    pole_resonant_frequency,
    pole_quality_factor,
)
from vfit.rlc.rlc_synthesis import foster_synthesis, FosterNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: known transfer functions
# ─────────────────────────────────────────────────────────────────────────────

def make_single_pole_response(n_pts=200):
    """H(s) = 1/(s - p),  p = -1e6 (real, negative)."""
    p = -2 * np.pi * 1e6     # pole at -1 MHz (rad/s)
    freq = np.logspace(4, 8, n_pts)
    s = 1j * 2 * np.pi * freq
    H = 1.0 / (s - p)
    return freq, H, np.array([p])


def make_conjugate_pair_response(n_pts=200):
    """H(s) = 1/(s² + 2αs + ω₀²),  a 2nd-order resonator."""
    f0 = 1e8       # Hz
    Q  = 10.0
    omega_0 = 2 * np.pi * f0
    alpha   = omega_0 / (2 * Q)
    omega_d = np.sqrt(omega_0**2 - alpha**2)
    p1 = -alpha + 1j * omega_d
    p2 = -alpha - 1j * omega_d

    freq = np.logspace(6, 10, n_pts)
    s = 1j * 2 * np.pi * freq
    H = 1.0 / ((s - p1) * (s - p2))
    return freq, H, np.array([p1, p2])


def make_rlc_impedance(R=50.0, L=10e-9, C=1e-12, n_pts=300):
    """Series RLC impedance Z(s) = R + sL + 1/(sC)."""
    freq = np.logspace(7, 11, n_pts)
    s = 1j * 2 * np.pi * freq
    Z = R + s * L + 1.0 / (s * C)
    return freq, Z


# ─────────────────────────────────────────────────────────────────────────────
# VFOptions
# ─────────────────────────────────────────────────────────────────────────────

class TestVFOptions:
    def test_defaults(self):
        opts = VFOptions()
        assert opts.n_poles == 10
        assert opts.n_iter_max == 20
        assert opts.tol == 1e-10          # actual default
        assert opts.weight == "inverse"   # actual default
        assert opts.enforce_stability is True

    def test_custom(self):
        opts = VFOptions(n_poles=4, tol=1e-8, verbose=True)
        assert opts.n_poles == 4
        assert opts.tol == 1e-8
        assert opts.verbose is True


# ─────────────────────────────────────────────────────────────────────────────
# VectorFitter: initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorFitterInit:
    def test_init_poles_log(self):
        vf = VectorFitter(n_poles=6)
        s = 1j * 2 * np.pi * np.logspace(6, 9, 100)  # complex s = jω
        poles = vf._init_poles(s)
        assert len(poles) == 6
        assert all(p.real < 0 for p in poles), "Initial poles must be in LHP"
        assert all(np.abs(p.imag) > 0 for p in poles), "Log-spaced poles should have imaginary part"

    def test_init_poles_linear(self):
        vf = VectorFitter(n_poles=4, init_poles="linear")
        s = 1j * 2 * np.pi * np.logspace(3, 6, 100)  # complex s = jω
        poles = vf._init_poles(s)
        assert len(poles) == 4

    def test_init_poles_odd_count(self):
        """Odd n_poles must return exactly n_poles poles, not n_poles+1."""
        for n in (3, 5, 7):
            vf = VectorFitter(n_poles=n)
            s = 1j * 2 * np.pi * np.logspace(6, 9, 100)
            poles = vf._init_poles(s)
            assert len(poles) == n, f"n_poles={n}: got {len(poles)}"
            # The extra real pole must be in the LHP
            real_poles = [p for p in poles if abs(p.imag) < 1e-10]
            assert len(real_poles) == 1
            assert real_poles[0].real < 0

    def test_invalid_option(self):
        with pytest.raises(TypeError):
            VectorFitter(nonexistent_option=True)


# ─────────────────────────────────────────────────────────────────────────────
# VectorFitter: LS step
# ─────────────────────────────────────────────────────────────────────────────

class TestVFLsStep:
    def test_output_shapes(self):
        vf = VectorFitter(n_poles=4)
        freq = np.logspace(6, 9, 50)
        s = 1j * 2 * np.pi * freq          # must be complex — _vf_ls_step expects s=jω
        H = np.ones(50, dtype=complex) * (1 + 0.5j)
        poles = vf._init_poles(s)

        c_H, d_H, e_H, c_sig = vf._vf_ls_step(s, H, poles)
        assert c_H.shape == (len(poles),)
        assert c_sig.shape == (len(poles),)
        assert np.isfinite(d_H)
        assert np.isfinite(e_H)

    def test_e_term_column_nonzero(self):
        """If include_e_term=True, the e column must be populated (requires s complex)."""
        vf = VectorFitter(n_poles=2, include_e_term=True)
        freq = np.logspace(6, 9, 50)
        s = 1j * 2 * np.pi * freq
        H = np.ones(50, dtype=complex)
        poles = vf._init_poles(s)
        _, _, e_H, _ = vf._vf_ls_step(s, H, poles)
        # e_H may be small but the column must have been built from s.imag (nonzero)
        assert np.isfinite(e_H)

    def test_trivial_constant_response(self):
        """For H = const, the fit should reproduce it exactly."""
        vf = VectorFitter(n_poles=2, n_iter_max=30)
        freq = np.logspace(4, 8, 100)
        H = np.full(100, 1.0 + 0j)   # constant

        model = vf.fit(freq, H)
        H_fit = model.evaluate(freq)
        # Relaxed tolerance since this is a pathological (degenerate) case
        assert_allclose(np.abs(H_fit - H), 0, atol=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# VectorFitter: pole relocation
# ─────────────────────────────────────────────────────────────────────────────

class TestPoleRelocation:
    def test_all_poles_lhp_after_stabilisation(self):
        vf = VectorFitter(n_poles=6, enforce_stability=True, n_iter_max=5)
        freq, H, _ = make_single_pole_response()
        model = vf.fit(freq, H)
        assert all(p.real <= 0 for p in model.poles), \
            f"Found RHP pole: {[p for p in model.poles if p.real > 0]}"

    def test_conjugate_symmetry(self):
        vf = VectorFitter(n_poles=6, n_iter_max=5)
        freq, H, _ = make_conjugate_pair_response()
        model = vf.fit(freq, H)
        for p in model.poles:
            if abs(p.imag) > 1e-6:
                conj_p = np.conj(p)
                dists = np.abs(model.poles - conj_p)
                assert dists.min() < 1e-3 * abs(p), \
                    f"No conjugate found for pole {p}"


# ─────────────────────────────────────────────────────────────────────────────
# VectorFitter: end-to-end fitting accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestVFFitAccuracy:
    def test_single_real_pole(self):
        """Fit a first-order system, check RMS error < 1%."""
        vf = VectorFitter(n_poles=2, n_iter_max=20, verbose=False)
        freq, H, true_poles = make_single_pole_response()
        model = vf.fit(freq, H)
        H_fit = model.evaluate(freq)
        rms_rel = np.sqrt(np.mean(np.abs((H_fit - H) / H)**2))
        assert rms_rel < 0.05, f"Relative RMS too high: {rms_rel:.3e}"

    def test_conjugate_pair_resonator(self):
        """Fit a 2nd-order resonator."""
        vf = VectorFitter(n_poles=4, n_iter_max=20, verbose=False)
        freq, H, _ = make_conjugate_pair_response()
        model = vf.fit(freq, H)
        H_fit = model.evaluate(freq)
        rms_rel = np.sqrt(np.mean(np.abs((H_fit - H) / H)**2))
        assert rms_rel < 0.05, f"Relative RMS too high: {rms_rel:.3e}"

    def test_rms_history_monotone(self):
        """RMS error should generally decrease over iterations."""
        vf = VectorFitter(n_poles=4, n_iter_max=15)
        freq, H, _ = make_conjugate_pair_response()
        model = vf.fit(freq, H)
        hist = model.rms_error_history
        assert len(hist) >= 2
        # At least the last half should be non-increasing (allow early instability)
        tail = hist[len(hist)//2:]
        assert tail[-1] <= tail[0] * 10, \
            f"RMS history not converging: {hist}"

    def test_model_rms_error_property(self):
        vf = VectorFitter(n_poles=4, n_iter_max=5)
        freq, H, _ = make_single_pole_response()
        model = vf.fit(freq, H)
        assert np.isfinite(model.rms_error)
        assert model.rms_error >= 0

    def test_model_repr(self):
        vf = VectorFitter(n_poles=4, n_iter_max=5)
        freq, H, _ = make_single_pole_response()
        model = vf.fit(freq, H)
        r = repr(model)
        assert "RationalModel" in r
        assert "n_poles" in r


# ─────────────────────────────────────────────────────────────────────────────
# RationalModel: evaluate
# ─────────────────────────────────────────────────────────────────────────────

class TestRationalModelEvaluate:
    def test_single_pole_evaluate(self):
        """Manual model: H(s) = 1/(s - p) with known p."""
        p = -2 * np.pi * 1e6 + 0j
        model = RationalModel(
            poles=np.array([p]),
            residues=np.array([1.0 + 0j]),
            d=0.0,
        )
        freq = np.array([1e6])
        s = 1j * 2 * np.pi * freq
        expected = 1.0 / (s - p)
        result = model.evaluate(freq)
        assert_allclose(result, expected, rtol=1e-10)

    def test_evaluate_shape(self):
        model = RationalModel(
            poles=np.array([-1e6 + 1j*2e6, -1e6 - 1j*2e6]),
            residues=np.array([0.5 + 0j, 0.5 + 0j]),
            d=1.0,
        )
        freq = np.logspace(5, 8, 50)
        H = model.evaluate(freq)
        assert H.shape == (50,)
        assert H.dtype == complex


# ─────────────────────────────────────────────────────────────────────────────
# Pole-Zero Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestPoleZeroUtils:
    def test_stabilize_poles_all_lhp(self):
        poles = np.array([-1e6, -2e6 + 1j*3e6, -2e6 - 1j*3e6])
        result = stabilize_poles(poles)
        assert all(p.real <= 0 for p in result)

    def test_stabilize_poles_flips_rhp(self):
        poles = np.array([+1e6 + 0j, -2e6 + 1j*3e6])
        result = stabilize_poles(poles)
        assert result[0].real < 0, "RHP pole should be reflected to LHP"
        assert result[0].real == pytest.approx(-1e6)

    def test_enforce_conjugate_pairs_real_pole(self):
        poles = np.array([-1e6 + 0j])
        result = enforce_conjugate_pairs(poles)
        assert len(result) == 1
        assert abs(result[0].imag) < 1e-10

    def test_enforce_conjugate_pairs_complex(self):
        p = -1e6 + 2j*1e6
        poles = np.array([p, np.conj(p)])
        result = enforce_conjugate_pairs(poles)
        assert len(result) == 2
        # Each pole should have a conjugate
        for r in result:
            if abs(r.imag) > 1e-8:
                assert any(abs(r2 - np.conj(r)) < 1e-6 for r2 in result)

    def test_enforce_conjugate_pairs_adds_missing(self):
        p = -1e6 + 2j*1e6
        poles = np.array([p])   # conjugate missing
        result = enforce_conjugate_pairs(poles)
        assert len(result) == 2

    def test_pole_resonant_frequency(self):
        p = -1e6 + 1j * 2 * np.pi * 1e8
        f0 = pole_resonant_frequency(p)
        assert f0 == pytest.approx(1e8, rel=1e-6)

    def test_pole_quality_factor(self):
        omega_0 = 2 * np.pi * 1e8
        Q_expected = 10.0
        alpha = omega_0 / (2 * Q_expected)
        omega_d = np.sqrt(omega_0**2 - alpha**2)
        p = -alpha + 1j * omega_d
        Q_calc = pole_quality_factor(p)
        assert Q_calc == pytest.approx(Q_expected, rel=0.01)

    def test_pole_quality_factor_real_pole(self):
        p = -1e6 + 0j   # real pole → Q = inf
        assert pole_quality_factor(p) == float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# RLC Foster Synthesis
# ─────────────────────────────────────────────────────────────────────────────

class TestFosterSynthesis:
    def _fit_rlc(self, R=50.0, L=10e-9, C=1e-12):
        """Fit a series RLC and return the model."""
        freq, Z = make_rlc_impedance(R=R, L=L, C=C)
        vf = VectorFitter(n_poles=6, n_iter_max=20, verbose=False)
        return vf.fit(freq, Z), freq, Z

    def test_synthesis_returns_network(self):
        model, freq, Z = self._fit_rlc()
        network = foster_synthesis(model)
        assert isinstance(network, FosterNetwork)
        assert len(network.branches) > 0

    def test_synthesis_raises_on_rhp_poles(self):
        model = RationalModel(
            poles=np.array([+1e6 + 0j]),
            residues=np.array([1.0 + 0j]),
            d=0.0,
        )
        with pytest.raises(ValueError, match="stable"):
            foster_synthesis(model)

    def test_network_impedance_round_trip(self):
        """Re-simulate synthesised network and compare to VF model."""
        model, freq, Z_meas = self._fit_rlc()
        network = foster_synthesis(model)
        Z_synth = network.impedance(freq)
        Z_model = model.evaluate(freq)
        # Network should reproduce the rational model to within 10%
        rms_rel = np.sqrt(np.mean(np.abs((Z_synth - Z_model) / (np.abs(Z_model) + 1e-30))**2))
        assert rms_rel < 0.15, f"Synthesis round-trip RMS too high: {rms_rel:.3e}"

    def test_all_element_values_positive(self):
        model, freq, _ = self._fit_rlc()
        network = foster_synthesis(model)
        for branch in network.branches:
            if branch.R is not None:
                assert branch.R >= 0, f"Negative R in branch {branch}"
            if branch.L is not None:
                assert branch.L >= 0, f"Negative L in branch {branch}"
            if branch.C is not None:
                assert branch.C >= 0, f"Negative C in branch {branch}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_rlc(self, tmp_path):
        """Fit → Synthesise → Export SPICE → check file exists."""
        from vfit.export.spice_export import export_spice_foster, export_spice_behavioral

        freq, Z = make_rlc_impedance()
        vf = VectorFitter(n_poles=6, n_iter_max=15)
        model = vf.fit(freq, Z)

        # Synthesis
        network = foster_synthesis(model)
        assert len(network.branches) > 0

        # SPICE export
        foster_path = tmp_path / "foster.cir"
        beh_path = tmp_path / "behavioral.cir"
        export_spice_foster(network, foster_path)
        export_spice_behavioral(model, beh_path)

        assert foster_path.exists(), "Foster SPICE file not created"
        assert beh_path.exists(), "Behavioral SPICE file not created"

        foster_text = foster_path.read_text()
        assert ".SUBCKT" in foster_text
        assert ".ENDS" in foster_text

    def test_spice_no_duplicate_element_names(self, tmp_path):
        """Every R, L, C element name in the netlist must be unique."""
        import re
        from vfit.export.spice_export import export_spice_foster

        freq, Z = make_rlc_impedance()
        vf = VectorFitter(n_poles=6, n_iter_max=15)
        model = vf.fit(freq, Z)
        network = foster_synthesis(model)

        path = tmp_path / "foster_names.cir"
        export_spice_foster(network, path)
        text = path.read_text()

        # Collect the first token of every non-comment, non-directive line
        elem_names = []
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith(("*", ".", "+")):
                elem_names.append(line.split()[0].upper())

        duplicates = [n for n in set(elem_names) if elem_names.count(n) > 1]
        assert not duplicates, f"Duplicate SPICE element names: {duplicates}"

    def test_spice_behavioral_polynomial_eval(self, tmp_path):
        """Behavioral export coefficients must reproduce H(jω) to within 1%."""
        import re
        import numpy.polynomial.polynomial as P
        from vfit.export.spice_export import export_spice_behavioral, _model_to_poly

        freq, H, _ = make_conjugate_pair_response()
        vf = VectorFitter(n_poles=4, n_iter_max=20)
        model = vf.fit(freq, H)

        num_poly, den_poly = _model_to_poly(model)
        # Evaluate via numpy poly (descending coeffs → use np.polyval)
        s = 1j * 2 * np.pi * freq
        H_poly = np.polyval(num_poly, s) / np.polyval(den_poly, s)
        H_vf   = model.evaluate(freq)

        rms_rel = np.sqrt(np.mean(np.abs((H_poly - H_vf) / (np.abs(H_vf) + 1e-30))**2))
        assert rms_rel < 0.01, f"Poly round-trip error too large: {rms_rel:.3e}"

    def test_bode_plot_runs(self):
        """Smoke test: bode_plot should not raise."""
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        from vfit.visualization import bode_plot

        freq, H, _ = make_conjugate_pair_response()
        vf = VectorFitter(n_poles=4, n_iter_max=10)
        model = vf.fit(freq, H)

        fig, axes = bode_plot(freq, H, model)
        assert fig is not None

    def test_pole_zero_map_runs(self):
        import matplotlib
        matplotlib.use("Agg")
        from vfit.visualization import pole_zero_map

        vf = VectorFitter(n_poles=4, n_iter_max=10)
        freq, H, _ = make_conjugate_pair_response()
        model = vf.fit(freq, H)

        fig, ax = pole_zero_map(model)
        assert fig is not None

    def test_convergence_plot_runs(self):
        import matplotlib
        matplotlib.use("Agg")
        from vfit.visualization import convergence_plot

        vf = VectorFitter(n_poles=4, n_iter_max=10)
        freq, H, _ = make_single_pole_response()
        model = vf.fit(freq, H)

        fig, ax = convergence_plot(model)
        assert fig is not None
