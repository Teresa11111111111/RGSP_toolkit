"""Quantum-state construction and pure-state fidelity utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .phase_encoding import Bitstring, computational_bitstrings, phase_values

ComplexArray = NDArray[np.complex128]


def equatorial_plus_state(angle: float) -> ComplexArray:
    """Return the equatorial single-qubit state with the given phase.

    Parameters
    ----------
    angle
        Relative phase between the computational-basis amplitudes.

    Returns
    -------
    numpy.ndarray
        Normalized state ``(|0> + exp(i angle)|1>) / sqrt(2)``.
    """
    return np.asarray(
        [1.0, np.exp(1j * angle)],
        dtype=complex,
    ) / np.sqrt(2.0)


def tensor_product_state(states: Sequence[ArrayLike]) -> ComplexArray:
    """Return the Kronecker product of an ordered state sequence.

    Parameters
    ----------
    states
        State vectors in tensor-product order.

    Returns
    -------
    numpy.ndarray
        Tensor-product state vector.
    """
    if not states:
        raise ValueError("states must contain at least one state vector.")
    product_state = np.asarray(states[0], dtype=complex)
    for state in states[1:]:
        product_state = np.kron(product_state, np.asarray(state, dtype=complex))
    return np.asarray(product_state, dtype=complex)


def normalized_state(state: ArrayLike) -> ComplexArray:
    """Return a normalized copy of a state vector.

    Parameters
    ----------
    state
        Complex state vector.

    Returns
    -------
    numpy.ndarray
        State vector with Euclidean norm one.
    """
    state_vector = np.asarray(state, dtype=complex)
    state_norm = np.linalg.norm(state_vector)
    if state_norm == 0.0:
        raise ValueError("A zero state cannot be normalized.")
    return np.asarray(state_vector / state_norm, dtype=complex)


def pure_state_fidelity(
    first_state: ArrayLike,
    second_state: ArrayLike,
) -> float:
    """Return the fidelity between two pure states.

    Parameters
    ----------
    first_state
        First state vector.
    second_state
        Second state vector.

    Returns
    -------
    float
        Squared magnitude of the normalized state overlap.
    """
    first_normalized = normalized_state(first_state)
    second_normalized = normalized_state(second_state)
    return float(np.abs(np.vdot(first_normalized, second_normalized)) ** 2)


def pulse_state(
    phases: ArrayLike,
    amplitudes: ArrayLike | None = None,
) -> ComplexArray:
    """Return a normalized temporal-mode pulse state.

    Parameters
    ----------
    phases
        Phase assigned to each temporal mode.
    amplitudes
        Optional real or complex amplitudes. Uniform amplitudes are used
        when this argument is omitted.

    Returns
    -------
    numpy.ndarray
        Normalized complex pulse amplitudes.
    """
    phase_array = np.asarray(phases, dtype=float)
    mode_count = len(phase_array)
    if amplitudes is None:
        amplitude_array = np.ones(mode_count, dtype=float) / np.sqrt(mode_count)
    else:
        amplitude_array = np.asarray(amplitudes, dtype=complex)
    if amplitude_array.shape != phase_array.shape:
        raise ValueError("amplitudes and phases must have equal shape.")
    return normalized_state(amplitude_array * np.exp(1j * phase_array))


def graph_phase_state(
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> ComplexArray:
    """Return a graph state constructed from its basis-state phases.

    Parameters
    ----------
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Normalized graph-state vector in computational-basis order.
    """
    ordered_bitstrings, encoded_phases = phase_values(
        local_angles,
        adjacency_matrix,
    )
    if len(ordered_bitstrings) != len(encoded_phases):
        raise RuntimeError("Phase encoding returned inconsistent mode data.")
    return pulse_state(encoded_phases)


def controlled_z_graph_state(
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> ComplexArray:
    """Return a graph state formed by controlled-Z edge phases.

    Parameters
    ----------
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Graph-state vector in computational-basis order.
    """
    angles = np.asarray(local_angles, dtype=float)
    adjacency = np.asarray(adjacency_matrix, dtype=float)
    qubit_count = len(angles)
    if adjacency.shape != (qubit_count, qubit_count):
        raise ValueError("adjacency_matrix has an incompatible shape.")

    product_state = tensor_product_state(
        [equatorial_plus_state(angle) for angle in angles]
    )
    graph_state = np.zeros_like(product_state, dtype=complex)
    for basis_index, bitstring in enumerate(
        computational_bitstrings(qubit_count)
    ):
        edge_sign = 1.0
        for i in range(qubit_count):
            for j in range(i + 1, qubit_count):
                if adjacency[i, j] == 1:
                    edge_sign *= (-1) ** (bitstring[i] * bitstring[j])
        graph_state[basis_index] = edge_sign * product_state[basis_index]
    return normalized_state(graph_state)


def state_from_phase_mapping(
    phases_by_state: Mapping[Bitstring, float],
) -> ComplexArray:
    """Return a uniform-amplitude state from a phase mapping.

    Parameters
    ----------
    phases_by_state
        Mapping from bitstrings to phases in radians.

    Returns
    -------
    numpy.ndarray
        Normalized state vector in lexicographic bitstring order.
    """
    if not phases_by_state:
        raise ValueError("phases_by_state cannot be empty.")
    qubit_count = len(next(iter(phases_by_state)))
    ordered_phases = [
        phases_by_state[bitstring]
        for bitstring in computational_bitstrings(qubit_count)
    ]
    return pulse_state(ordered_phases)


def weighted_state_from_phase_mapping(
    phases_by_state: Mapping[Bitstring, float],
    amplitudes_by_state: Mapping[Bitstring, float],
) -> ComplexArray:
    """Return a weighted state from bitstring phases and amplitudes.

    Parameters
    ----------
    phases_by_state
        Mapping from bitstrings to phases in radians.
    amplitudes_by_state
        Mapping from bitstrings to state amplitudes.

    Returns
    -------
    numpy.ndarray
        Weighted complex state vector in lexicographic order.
    """
    if phases_by_state.keys() != amplitudes_by_state.keys():
        raise ValueError("Phase and amplitude mappings must share all keys.")
    qubit_count = len(next(iter(phases_by_state)))
    ordered_bitstrings = computational_bitstrings(qubit_count)
    return np.asarray(
        [
            amplitudes_by_state[bitstring]
            * np.exp(1j * phases_by_state[bitstring])
            for bitstring in ordered_bitstrings
        ],
        dtype=complex,
    )
