"""Deterministic diagonal-loss RGSP model; angles are radians, qubits zero-based."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import secrets

import numpy as np

TAU = 2 * np.pi
MAX_QUBITS = 16  # Explicit mode enumeration costs O(n * 2**n).


def validate_qubits(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= MAX_QUBITS:
        raise ValueError(f"qubits must be an integer in [1, {MAX_QUBITS}]")
    return int(n)


def vector(value, size: int, name: str, default=None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = np.full(size, default)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (size,):
        raise ValueError(f"{name} must be a real numeric vector of length {size}")
    result = raw.astype(float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def bits_vector(value, n: int, name: str) -> np.ndarray:
    result = vector(value, n, name, 0)
    if not np.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must contain only 0 or 1")
    return result.astype(np.int8)


def bitstrings(n: int) -> np.ndarray:
    """Rows 00...0 to 11...1; qubit 0 is the most significant bit."""
    n = validate_qubits(n)
    return ((np.arange(1 << n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)


def canonical_edges(n: int, edges=()) -> tuple[tuple[int, int], ...]:
    """Simple undirected graph; duplicate edges in one list are rejected."""
    validate_qubits(n)
    result = set()
    for edge in edges:
        if not isinstance(edge, (list, tuple, np.ndarray)) or len(edge) != 2:
            raise ValueError("each edge must contain two zero-based qubit indices")
        i, j = edge
        if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) for k in (i, j)):
            raise ValueError("edge indices must be integers")
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError("edges must join distinct qubits in range [0, qubits)")
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        if pair in result:
            raise ValueError("duplicate edge in one graph/toggle list; combine clients with XOR")
        result.add(pair)
    return tuple(sorted(result))


def xor_edges(n: int, *edge_lists) -> tuple[tuple[int, int], ...]:
    result = set()
    for edges in edge_lists:
        result.symmetric_difference_update(canonical_edges(n, edges))
    return tuple(sorted(result))


def graph_phases(n: int, edges=(), local_angles_rad=None) -> np.ndarray:
    """phi_x = sum_i theta_i x_i + pi sum_(i,j in E) x_i x_j, mod 2pi."""
    x = bitstrings(n)
    theta = np.remainder(vector(local_angles_rad, n, "local_angles_rad", 0), TAU)
    phase = x @ theta
    for i, j in canonical_edges(n, edges):
        phase += np.pi * x[:, i] * x[:, j]
    return np.remainder(phase, TAU)


def graph_state(n: int, edges=(), local_angles_rad=None) -> np.ndarray:
    return np.exp(1j * graph_phases(n, edges, local_angles_rad)) / np.sqrt(1 << n)


def efficiencies(value, count: int, name="intensity_efficiencies") -> np.ndarray:
    eta = vector(value, count, name)
    if np.any((eta <= 0) | (eta > 1)):
        raise ValueError(f"{name} must be in (0, 1]; zero transmission cannot be inverted")
    return eta


def hamming_efficiencies(n: int, eta0=1.0, eta1=1.0, common=1.0) -> np.ndarray:
    """Intensity eta_x = common * product_i eta[x_i,i]; scalars or n-vectors."""
    x = bitstrings(n)
    a = efficiencies(np.full(n, eta0) if np.ndim(eta0) == 0 else eta0, n, "eta0")
    b = efficiencies(np.full(n, eta1) if np.ndim(eta1) == 0 else eta1, n, "eta1")
    c = efficiencies([common], 1, "common")[0]
    result = np.exp(np.log(c) + np.sum(np.where(x, np.log(b), np.log(a)), axis=1))
    if np.any(result == 0):
        raise ValueError("efficiency underflow; channel cannot be represented in float64")
    return result


def compensation(intensity_efficiencies) -> tuple[np.ndarray, float]:
    """Unit-norm input amplitudes and all-outcome survival probability.

    a_x = eta_x**(-1/2) / sqrt(sum_y 1/eta_y); P = d / sum_y 1/eta_y.
    Computation rescales by min(eta), avoiding overflow in inverse eta.
    """
    raw = np.asarray(intensity_efficiencies)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("intensity_efficiencies must be a nonempty vector")
    eta = efficiencies(raw, raw.size)
    relative = np.min(eta) / eta
    amplitude = np.sqrt(relative / np.sum(relative))
    survival = float(np.min(eta) * (eta.size / np.sum(relative)))
    if np.any(amplitude == 0) or survival == 0:
        raise ValueError("compensation underflow; increase numerical precision")
    return amplitude, survival


@dataclass(frozen=True)
class Client:
    """Local phase-only instruction. Only client 0 also programs source amplitudes."""
    graph_toggle_edges: tuple = ()
    local_angles_rad: object = None
    hiding_angles_rad: object = None
    z_bits: object = None


def client_vectors(n: int, client: Client):
    logical = np.remainder(vector(client.local_angles_rad, n, "local_angles_rad", 0), TAU)
    hiding = np.remainder(vector(client.hiding_angles_rad, n, "hiding_angles_rad", 0), TAU)
    z = bits_vector(client.z_bits, n, "z_bits")
    return logical, np.remainder(hiding + np.pi * z, TAU)


def client_phase_update(n: int, client: Client) -> np.ndarray:
    """Can run locally: no other client's graph, coefficients or angles are needed."""
    logical, mask = client_vectors(n, client)
    return graph_phases(n, client.graph_toggle_edges, logical + mask)


def random_hiding(n: int) -> dict:
    """Fresh OS-random eight-angle masks and independent Pauli Z bits.

    This generates local randomness, not a cryptographic network protocol.
    """
    validate_qubits(n)
    return {"hiding_angles_rad": [secrets.randbelow(8) * np.pi / 4 for _ in range(n)],
            "z_bits": [secrets.randbelow(2) for _ in range(n)]}


@dataclass
class Preparation:
    qubits: int
    output_edges: tuple
    input_amplitudes: np.ndarray
    source_phases_rad: np.ndarray
    client_updates_rad: tuple
    intensity_efficiencies: np.ndarray
    channel_phases_rad: np.ndarray
    logical_angles_rad: np.ndarray
    encryption_angles_rad: np.ndarray
    preparation_angles_rad: np.ndarray
    survival_probability: float

    @property
    def input_coefficients(self):
        return self.input_amplitudes * np.exp(1j * self.source_phases_rad)

    def output_state(self, herald_bits=None):
        """Forward propagation, conditioned on the specified H^n detector outcome."""
        state = self.input_coefficients.copy()
        for update in self.client_updates_rad[1:]:
            state *= np.exp(1j * update)
        state *= np.sqrt(self.intensity_efficiencies) * np.exp(1j * self.channel_phases_rad)
        state *= np.exp(1j * np.pi * (bitstrings(self.qubits) @
                                      bits_vector(herald_bits, self.qubits, "herald_bits")))
        # Scaling before norm avoids underflow for very lossy calibrated channels.
        state /= np.max(np.abs(state))
        return state / np.linalg.norm(state)

    def target_state(self, include_hiding=True):
        theta = self.preparation_angles_rad if include_hiding else self.logical_angles_rad
        return graph_state(self.qubits, self.output_edges, theta)

    def correction_record(self, herald_bits=None):
        m = bits_vector(herald_bits, self.qubits, "herald_bits")
        return {
            "qubit_order": list(range(self.qubits)),
            "logical_angles_rad": self.logical_angles_rad.tolist(),
            "encryption_angles_rad": self.encryption_angles_rad.tolist(),
            "preparation_angles_rad": self.preparation_angles_rad.tolist(),
            "herald_bits": m.tolist(),
            "inverse_hiding_and_herald_rad": np.remainder(
                -self.encryption_angles_rad - np.pi * m, TAU).tolist(),
            "warning": "PRIVATE: do not send this record to an untrusted server.",
        }


def prepare(n: int, source_graph_edges=(), clients=None, intensity_efficiencies=None,
            channel_phases_rad=None) -> Preparation:
    """Compile calibrated loss compensation at source plus phase-only clients.

    Every supplied efficiency is intensity transmission along the entire path.
    The source pre-cancels the known total channel phase. Unknown noise is not corrected.
    """
    n = validate_qubits(n)
    source_edges = canonical_edges(n, source_graph_edges)
    clients = tuple([Client()] if clients is None else clients)
    if not clients or not all(isinstance(c, Client) for c in clients):
        raise ValueError("clients must be a nonempty sequence of Client objects")
    d = 1 << n
    eta = efficiencies(np.ones(d) if intensity_efficiencies is None else intensity_efficiencies, d)
    chi = np.remainder(vector(channel_phases_rad, d, "channel_phases_rad", 0), TAU)
    amplitude, survival = compensation(eta)
    updates = tuple(client_phase_update(n, c) for c in clients)
    logical = np.zeros(n)
    mask = np.zeros(n)
    for client in clients:
        a, b = client_vectors(n, client)
        logical = np.remainder(logical + a, TAU)
        mask = np.remainder(mask + b, TAU)
    edges = xor_edges(n, source_edges, *(c.graph_toggle_edges for c in clients))
    phase = np.remainder(graph_phases(n, source_edges) + updates[0] - chi, TAU)
    return Preparation(n, edges, amplitude, phase, updates, eta, chi,
                       logical, mask, np.remainder(logical + mask, TAU), survival)


def measurement_instructions(adapted_angles_rad, preparation_angles_rad, herald_bits=None,
                             outcome_mask_bits=None) -> dict:
    """delta = phi' + theta + pi*m + pi*r, relative to the unrotated final graph.

    Caller supplies already-adapted phi'. Only delta is sent to the server.
    Interpret returned server bits as b XOR r. No MBQC flow is inferred here.
    """
    theta_raw = np.asarray(preparation_angles_rad)
    if theta_raw.ndim != 1:
        raise ValueError("preparation_angles_rad must be one-dimensional")
    n = validate_qubits(theta_raw.size)
    theta = np.remainder(vector(theta_raw, n, "preparation_angles_rad"), TAU)
    phi = np.remainder(vector(adapted_angles_rad, n, "adapted_angles_rad"), TAU)
    m = bits_vector(herald_bits, n, "herald_bits")
    r = bits_vector(outcome_mask_bits, n, "outcome_mask_bits")
    return {"qubit_order": list(range(n)),
            "measurement_angles_rad": np.remainder(phi + theta + np.pi * (m + r), TAU).tolist(),
            "basis": "(|0> +/- exp(i*delta)|1>)/sqrt(2)",
            "angle_reference": "unrotated final graph; phi_prime must already include MBQC adaptation"}


def decode_outcomes(server_bits, outcome_mask_bits) -> list[int]:
    raw = np.asarray(server_bits)
    if raw.ndim != 1:
        raise ValueError("server_bits must be a one-dimensional bit vector")
    n = validate_qubits(raw.size)
    return (bits_vector(server_bits, n, "server_bits") ^
            bits_vector(outcome_mask_bits, n, "outcome_mask_bits")).tolist()
