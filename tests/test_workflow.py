import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from rgsp_toolkit.cli import load_json
from rgsp_toolkit.io import channel_arrays, compile_config

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "rgsp_toolkit", *map(str, args)],
                              capture_output=True, text=True)

    def test_all_examples_and_export_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("lossy_line", "product_state", "three_clients"):
                out = Path(tmp) / name
                result = self.run_cli("compile", ROOT / "examples" / f"{name}.json", "--out", out)
                self.assertEqual(result.returncode, 0, result.stderr)
                summary_text = (out / "summary.json").read_text()
                summary = json.loads(summary_text)
                self.assertAlmostEqual(summary["predicted_conditioned_fidelity"], 1)
                self.assertAlmostEqual(summary["input_power_sum"], 1)
                with (out / "source_modes.csv").open() as f:
                    source = list(csv.DictReader(f))
                self.assertEqual(len(source), summary["modes"])
                self.assertEqual(source[0]["bitstring_q0_first"], "0" * summary["qubits"])
                private = json.loads((out / "private_corrections.json").read_text())
                self.assertIn("encryption_angles_rad", private)
                if name == "product_state":
                    self.assertFalse((out / "server_measurement_instructions.json").exists())
                    self.assertEqual(private["herald_outcome_status"], "assumed_all_zero_for_preview")
                else:
                    server = json.loads((out / "server_measurement_instructions.json").read_text())
                    self.assertEqual(set(server), {"qubit_order", "measurement_angles_rad", "basis", "angle_reference"})
                for path in out.glob("client_*_phases.csv"):
                    with path.open() as f:
                        self.assertNotIn("input_amplitude", next(csv.reader(f)))
                rerun = self.run_cli("compile", ROOT / "examples" / f"{name}.json", "--out", out)
                self.assertEqual(rerun.returncode, 2)
                self.assertEqual((out / "summary.json").read_text(), summary_text)

    def test_local_client_and_random_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "local"
            result = self.run_cli("client", ROOT / "examples/local_client.json", "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out / "client_phases.csv").open() as f:
                rows = list(csv.DictReader(f))
            plan = compile_config(load_json(ROOT / "examples/three_clients.json"))
            assert_allclose([float(row["phase_rad"]) for row in rows], plan.client_updates_rad[1])
            self.assertFalse((out / "source_modes.csv").exists())
            result = self.run_cli("random-mask", "--qubits", 3, "--out", Path(tmp) / "mask")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("hiding_angles_rad", result.stdout)

    def test_calibrated_channels_compose(self):
        eta, chi = channel_arrays(1, [
            {"model": "measured", "intensity_efficiencies": [0.8, 0.6], "phases_rad": [0.1, -0.2]},
            {"model": "measured", "intensity_efficiencies": [0.5, 0.4], "phases_rad": [0.3, 0.1]}])
        assert_allclose(eta, [0.4, 0.24])
        assert_allclose(np.exp(1j * chi), np.exp(1j * np.array([0.4, -0.1])))

    def test_strict_configs(self):
        for cfg in (
            {"qubits": 2, "clietns": []},
            {"qubits": 2, "herald_bits": None},
            {"qubits": 2, "herald_bits": [0, 0], "measurement": {"adapted_angles_rad": [0, 0], "outcome_mask_bits": None}},
            {"qubits": 2, "clients": [{}, {"amplitudes": [1, 1, 1, 1]}]},
            {"qubits": 2, "clients": []},
            {"qubits": 2, "channels": []},
            {"qubits": 2, "channels": [{"model": "hamming", "intensity_efficiencies": [1] * 4}]},
            {"qubits": 2, "measurement": {"adapted_angles_rad": [0, 0], "outcome_mask_bits": [0, 0]}},
            {"qubits": 2, "herald_bits": [0, 0], "measurement": {"adapted_angles_rad": [0, 0]}},
        ):
            with self.subTest(cfg=cfg):
                with self.assertRaises(ValueError):
                    compile_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            for text in ('{"qubits": 2, "qubits": 3}', '{"qubits": NaN}'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    load_json(p)
            result = self.run_cli("compile", p, "--out", Path(tmp) / "should_not_exist")
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(tmp) / "should_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
