import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose

from rgsp_toolkit import (Client, bitstrings, client_phase_update, compensation,
                         decode_outcomes, graph_phases, graph_state, hamming_efficiencies,
                         measurement_instructions, prepare, random_hiding, xor_edges)


def independent_state(n, edges, angles):
    # Tensor product of equatorial states followed by explicit CZ sign changes.
    state = np.array([1.0 + 0j])
    for angle in angles:
        state = np.kron(state, np.array([1, np.exp(1j * angle)]) / np.sqrt(2))
    for index in range(2 ** n):
        bits = tuple(int(c) for c in format(index, f"0{n}b"))
        for i, j in edges:
            if bits[i] and bits[j]:
                state[index] *= -1
    return state


def measurement_probabilities(state, angles):
    rotation = np.array([[1.0 + 0j]])
    for angle in angles:
        rotation = np.kron(rotation, np.array([[1, np.exp(-1j * angle)],
                                              [1, -np.exp(-1j * angle)]]) / np.sqrt(2))
    return abs(rotation @ state) ** 2


class PhysicsTests(unittest.TestCase):
    def test_paper_two_qubit_phases_and_endianness(self):
        a, b = 0.2, 0.7
        assert_allclose(graph_phases(2, [(0, 1)], [a, b]), [0, b, a, a + b + np.pi])
        self.assertEqual(bitstrings(2).tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_reference_cz_and_product_states(self):
        for n in range(1, 7):
            theta = np.linspace(-1, 2, n)
            for edges in ([], [(i, i + 1) for i in range(n - 1)]):
                assert_allclose(graph_state(n, edges, theta), independent_state(n, edges, theta), atol=2e-15)

    def test_inverse_loss_matches_original_formula(self):
        eta = np.array([0.9, 0.7, 0.4, 0.2])
        amplitude, p = compensation(eta)
        assert_allclose(amplitude, np.sqrt((1 / eta) / np.sum(1 / eta)), rtol=1e-15)
        self.assertAlmostEqual(p, len(eta) / np.sum(1 / eta))
        self.assertAlmostEqual(np.sum(amplitude ** 2 * eta), p)
        assert_allclose(amplitude * np.sqrt(eta), np.full(4, np.sqrt(p / 4)))

    def test_uniform_loss_only_affects_probability(self):
        amplitude, p = compensation([0.02] * 8)
        assert_allclose(amplitude, np.full(8, 1 / np.sqrt(8)))
        self.assertAlmostEqual(p, 0.02)

    def test_extreme_but_representable_efficiencies(self):
        amplitude, p = compensation([1e-300, 2e-300, 1e-200, 1])
        self.assertTrue(np.all(np.isfinite(amplitude)))
        self.assertGreater(p, 0)
        plan = prepare(2, [(0, 1)], intensity_efficiencies=[1e-300, 2e-300, 1e-200, 1])
        assert_allclose(plan.output_state(), plan.target_state(), atol=1e-14)

    def test_hamming_model_scalar_and_per_qubit(self):
        eta = hamming_efficiencies(2, eta0=[0.8, 0.9], eta1=[0.5, 0.6], common=0.7)
        assert_allclose(eta, 0.7 * np.array([0.8 * 0.9, 0.8 * 0.6, 0.5 * 0.9, 0.5 * 0.6]))
        for n in range(1, 6):
            expected = [0.7 * 0.8 ** sum(x) * 0.95 ** (n - sum(x)) for x in itertools.product([0, 1], repeat=n)]
            assert_allclose(hamming_efficiencies(n, 0.95, 0.8, 0.7), expected)

    def test_n_clients_loss_phase_calibration_and_xor(self):
        rng = np.random.default_rng(1203)
        for n in range(2, 7):
            for count in (1, 2, 5):
                edges = [(i, i + 1) for i in range(n - 1)]
                clients = [Client([(0, n - 1)], rng.uniform(-4, 4, n),
                                  rng.uniform(-4, 4, n), rng.integers(0, 2, n)) for _ in range(count)]
                eta = rng.uniform(0.01, 0.9, 2 ** n)
                chi = rng.uniform(-np.pi, np.pi, 2 ** n)
                plan = prepare(n, edges, clients, eta, chi)
                expected_edges = set(edges)
                if count % 2:
                    expected_edges.symmetric_difference_update([(0, n - 1)])
                theta = sum(np.asarray(c.local_angles_rad) + np.asarray(c.hiding_angles_rad)
                            + np.pi * np.asarray(c.z_bits) for c in clients)
                expected = independent_state(n, expected_edges, theta)
                assert_allclose(plan.output_state(), expected, atol=2e-14)
                self.assertEqual(set(plan.output_edges), expected_edges)
                # No renormalization can disguise an incorrect success probability.
                raw = plan.input_coefficients * np.sqrt(eta) * np.exp(1j * chi)
                for update in plan.client_updates_rad[1:]:
                    raw *= np.exp(1j * update)
                self.assertAlmostEqual(float(np.vdot(raw, raw).real), plan.survival_probability)

    def test_all_herald_outcomes_and_inverse_hiding(self):
        n = 3
        client = Client(local_angles_rad=[0.1, 0.2, 0.3], hiding_angles_rad=[0.4, 0.5, 0.6], z_bits=[1, 0, 1])
        plan = prepare(n, [(0, 1), (1, 2)], [client], np.linspace(0.1, 0.9, 8))
        for m in itertools.product([0, 1], repeat=n):
            state = plan.output_state(m)
            correction = plan.correction_record(m)["inverse_hiding_and_herald_rad"]
            corrected = state * np.exp(1j * (bitstrings(n) @ correction))
            assert_allclose(corrected, plan.target_state(include_hiding=False), atol=2e-14)
            # Optical Hadamard row has magnitude 1/sqrt(d).
            raw = plan.input_coefficients * np.sqrt(plan.intensity_efficiencies)
            raw *= (-1.0) ** (bitstrings(n) @ np.asarray(m)) / np.sqrt(8)
            self.assertAlmostEqual(float(np.vdot(raw, raw).real), plan.survival_probability / 8)

    def test_measurement_angles_preserve_born_probabilities_with_decoding(self):
        n = 3
        edges = [(0, 1), (1, 2)]
        theta = np.array([0.37, -1.3, 4.9])
        phi = np.array([-0.9, 0.6, 1.8])
        reference = measurement_probabilities(independent_state(n, edges, [0] * n), phi)
        for m in itertools.product([0, 1], repeat=n):
            for r in itertools.product([0, 1], repeat=n):
                message = measurement_instructions(phi, theta, m, r)
                state = independent_state(n, edges, theta + np.pi * np.asarray(m))
                actual = measurement_probabilities(state, message["measurement_angles_rad"])
                for b in itertools.product([0, 1], repeat=n):
                    physical_index = int("".join(map(str, b)), 2)
                    decoded = decode_outcomes(b, r)
                    logical_index = int("".join(map(str, decoded)), 2)
                    self.assertAlmostEqual(actual[physical_index], reference[logical_index], places=13)
                self.assertNotIn("encryption_angles_rad", message)

    def test_phase_updates_compose_commutatively(self):
        a = Client([(0, 1)], [0.1, 0.2], [0.3, 0.4])
        b = Client([(0, 1)], [0.5, 0.6], [0.7, 0.8])
        ab, ba = prepare(2, clients=[a, b]), prepare(2, clients=[b, a])
        self.assertEqual(ab.output_edges, ())
        assert_allclose(ab.output_state(), ba.output_state(), atol=1e-14)
        assert_allclose(ab.client_updates_rad[1], client_phase_update(2, b))
        self.assertEqual(xor_edges(3, [(0, 1), (1, 2)], [(0, 1), (0, 2)]), ((0, 2), (1, 2)))

    def test_invalid_inputs(self):
        invalid = [lambda: prepare(0), lambda: prepare(17), lambda: prepare(True),
                   lambda: prepare(2, [(0, 0)]), lambda: prepare(2, [(0, 2)]),
                   lambda: prepare(2, [(0, 1), (1, 0)]), lambda: prepare(2, [(0.0, 1)]),
                   lambda: prepare(2, clients=[]),
                   lambda: prepare(2, clients=[Client(local_angles_rad=[1])]),
                   lambda: prepare(2, clients=[Client(z_bits=[0, 2])]),
                   lambda: prepare(2, clients=[Client(local_angles_rad=[np.nan, 0])]),
                   lambda: prepare(2, clients=[Client(local_angles_rad=[1j, 0])]),
                   lambda: compensation([0, 1]), lambda: compensation([1.1, 1]),
                   lambda: compensation([np.inf, 1]), lambda: compensation([]),
                   lambda: hamming_efficiencies(2, common=-1)]
        for fn in invalid:
            with self.subTest(fn=fn):
                with self.assertRaises(ValueError):
                    fn()

    def test_random_masks_allowed_alphabet(self):
        mask = random_hiding(16)
        angles = np.asarray(mask["hiding_angles_rad"]) / (np.pi / 4)
        assert_allclose(angles, np.round(angles), atol=1e-14)
        self.assertTrue(np.all((angles >= 0) & (angles < 8)))
        self.assertTrue(set(mask["z_bits"]) <= {0, 1})


if __name__ == "__main__":
    unittest.main()
