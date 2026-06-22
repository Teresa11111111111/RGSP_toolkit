"""Graph-state phase encoding and phase-decomposition utilities."""

from __future__ import annotations

from itertools import product
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

Bitstring = tuple[int, ...]
FloatArray = NDArray[np.float64]


class PhaseComponents(TypedDict):
    """Components of a computational-basis graph-state phase."""

    phase: float
    local_phase: float
    graph_phase: float


def computational_bitstrings(qubit_count: int) -> list[Bitstring]:
    """Return computational-basis bitstrings in lexicographic order.

    Parameters
    ----------
    qubit_count
        Number of qubits represented by each bitstring.

    Returns
    -------
    list of tuple of int
        All binary bitstrings from all-zero to all-one.
    """
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least one.")
    return list(product((0, 1), repeat=qubit_count))


def phase_input_arrays(
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return validated local-angle and adjacency arrays.

    Parameters
    ----------
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Square graph adjacency matrix.

    Returns
    -------
    tuple of numpy.ndarray
        Local angles and adjacency matrix as floating-point arrays.
    """
    angles = np.asarray(local_angles, dtype=float)
    adjacency = np.asarray(adjacency_matrix, dtype=float)
    qubit_count = len(angles)
    if angles.ndim != 1:
        raise ValueError("local_angles must be one-dimensional.")
    if adjacency.shape != (qubit_count, qubit_count):
        raise ValueError(
            "adjacency_matrix must have shape "
            f"({qubit_count}, {qubit_count})."
        )
    return angles, adjacency


def phase_components(
    bitstring: Bitstring,
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> PhaseComponents:
    """Return local, graph, and total phases for one basis state.

    Parameters
    ----------
    bitstring
        Computational-basis bitstring.
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    PhaseComponents
        Local, graph, and total phase values in radians.
    """
    angles, adjacency = phase_input_arrays(local_angles, adjacency_matrix)
    if len(bitstring) != len(angles):
        raise ValueError("bitstring and local_angles must have equal length.")

    local_phase = sum(
        angles[i] * bitstring[i] for i in range(len(bitstring))
    )
    graph_phase_factor = 0.0
    for i in range(len(bitstring)):
        for j in range(i + 1, len(bitstring)):
            graph_phase_factor += (
                adjacency[i, j] * bitstring[i] * bitstring[j]
            )
    graph_phase = float(np.pi * graph_phase_factor)
    total_phase = float(local_phase + graph_phase)
    return {
        "phase": total_phase,
        "local_phase": float(local_phase),
        "graph_phase": graph_phase,
    }


def bitstring_phase(
    bitstring: Bitstring,
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> float:
    """Return the encoded phase of one computational-basis bitstring.

    Parameters
    ----------
    bitstring
        Computational-basis bitstring.
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    float
        Encoded phase in radians.
    """
    return phase_components(
        bitstring,
        local_angles,
        adjacency_matrix,
    )["phase"]


def phase_decomposition_by_bitstring(
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> dict[Bitstring, PhaseComponents]:
    """Return phase decompositions for every computational-basis state.

    Parameters
    ----------
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    dict
        Mapping from bitstrings to local, graph, and total phases.
    """
    angles, adjacency = phase_input_arrays(local_angles, adjacency_matrix)
    return {
        bitstring: phase_components(bitstring, angles, adjacency)
        for bitstring in computational_bitstrings(len(angles))
    }


def phases_by_bitstring(
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> dict[Bitstring, float]:
    """Return encoded phases keyed by computational-basis bitstring.

    Parameters
    ----------
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    dict
        Mapping from bitstrings to encoded phases in radians.
    """
    angles, adjacency = phase_input_arrays(local_angles, adjacency_matrix)
    return {
        bitstring: bitstring_phase(bitstring, angles, adjacency)
        for bitstring in computational_bitstrings(len(angles))
    }


def phase_values(
    local_angles: ArrayLike,
    adjacency_matrix: ArrayLike,
) -> tuple[list[Bitstring], FloatArray]:
    """Return ordered bitstrings and their encoded phase values.

    Parameters
    ----------
    local_angles
        One local phase angle per qubit.
    adjacency_matrix
        Symmetric graph adjacency matrix.

    Returns
    -------
    tuple
        Lexicographically ordered bitstrings and phase values in radians.
    """
    angles, adjacency = phase_input_arrays(local_angles, adjacency_matrix)
    bitstrings = computational_bitstrings(len(angles))
    values = np.asarray(
        [
            bitstring_phase(bitstring, angles, adjacency)
            for bitstring in bitstrings
        ],
        dtype=float,
    )
    return bitstrings, values
