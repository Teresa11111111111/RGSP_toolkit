"""Parity against the preserved pre-calculator source, when available in a checkout."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np

from rgsp_toolkit import compensation, graph_phases

ARCHIVE = Path(__file__).resolve().parents[1] / "legacy/paper_toolkit"


@unittest.skipUnless(ARCHIVE.is_dir(), "legacy source not included in installed distribution")
class LegacyParityTests(unittest.TestCase):
    def test_original_phase_and_inverse_loss_formulas(self):
        package = types.ModuleType("_rgsp_legacy_parity")
        package.__path__ = [str(ARCHIVE)]
        sys.modules[package.__name__] = package
        old_phase = importlib.import_module(package.__name__ + ".phase_encoding")
        old_amplitude = importlib.import_module(package.__name__ + ".amplitude_compensation")
        rng = np.random.default_rng(9012026)
        phase_error = amplitude_error = 0.0
        for n in range(2, 9):
            angles = rng.uniform(-np.pi, np.pi, n)
            adjacency = np.zeros((n, n))
            edges = [(i, i + 1) for i in range(n - 1)]
            for i, j in edges:
                adjacency[i, j] = adjacency[j, i] = 1
            _, old = old_phase.phase_values(angles, adjacency)
            new = graph_phases(n, edges, angles)
            phase_error = max(phase_error, float(np.max(abs(np.exp(1j * old) - np.exp(1j * new)))))
            eta = rng.uniform(0.01, 1, 2 ** n)
            amplitude_error = max(amplitude_error, float(np.max(abs(
                old_amplitude.loss_compensated_amplitudes(eta) - compensation(eta)[0]))))
        # Modulo reduction and summation order differ; compare unit phasors.
        self.assertLess(phase_error, 1e-12)
        self.assertLess(amplitude_error, 2e-15)
        print(f"\nLegacy parity n=2..8: max phase-phasor error {phase_error:.3g}; "
              f"amplitude error {amplitude_error:.3g}")
