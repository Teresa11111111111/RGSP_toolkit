"""Amplitude weighting and temporal-mode ordering strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .loss_models import hamming_weight
from .phase_encoding import Bitstring

FloatArray = NDArray[np.float64]


def compensation_coefficients(
    efficiencies_by_state: Mapping[Bitstring, float],
) -> tuple[dict[Bitstring, float], float]:
    """Return normalized inverse-loss amplitudes and their normalization.

    Parameters
    ----------
    efficiencies_by_state
        Mapping from bitstrings to nonzero mode efficiencies.

    Returns
    -------
    tuple
        Amplitude mapping and the sum of inverse efficiencies.
    """
    if not efficiencies_by_state:
        raise ValueError("efficiencies_by_state cannot be empty.")
    inverse_efficiencies = {
        bitstring: 1.0 / efficiency
        for bitstring, efficiency in efficiencies_by_state.items()
    }
    normalization = float(sum(inverse_efficiencies.values()))
    amplitudes = {
        bitstring: float(np.sqrt(value / normalization))
        for bitstring, value in inverse_efficiencies.items()
    }
    return amplitudes, normalization


def loss_compensated_amplitudes(efficiencies: ArrayLike) -> FloatArray:
    """Return normalized amplitudes proportional to inverse square-root loss.

    Parameters
    ----------
    efficiencies
        Positive transmission efficiency for each temporal mode.

    Returns
    -------
    numpy.ndarray
        Real amplitudes whose squared magnitudes are proportional to
        inverse efficiency.
    """
    efficiency_array = np.asarray(efficiencies, dtype=float)
    if np.any(efficiency_array <= 0.0):
        raise ValueError("All efficiencies must be positive.")
    inverse_efficiencies = 1.0 / efficiency_array
    probabilities = inverse_efficiencies / np.sum(inverse_efficiencies)
    return np.asarray(np.sqrt(probabilities), dtype=float)


def uniform_amplitudes(mode_count: int) -> FloatArray:
    """Return normalized uniform amplitudes for a mode sequence.

    Parameters
    ----------
    mode_count
        Number of temporal modes.

    Returns
    -------
    numpy.ndarray
        Uniform real amplitudes with unit total probability.
    """
    if mode_count < 1:
        raise ValueError("mode_count must be at least one.")
    return np.ones(mode_count, dtype=float) / np.sqrt(mode_count)


def high_hamming_first_order(
    bitstrings: Sequence[Bitstring],
    phases: ArrayLike,
    amplitudes: ArrayLike,
) -> tuple[list[Bitstring], FloatArray, FloatArray]:
    """Return modes ordered by descending Hamming weight.

    Parameters
    ----------
    bitstrings
        Ordered binary mode labels.
    phases
        Phase value for each mode.
    amplitudes
        Amplitude for each mode.

    Returns
    -------
    tuple
        Reordered bitstrings, phases, and amplitudes.
    """
    phase_array = np.asarray(phases, dtype=float)
    amplitude_array = np.asarray(amplitudes, dtype=float)
    if len(bitstrings) != len(phase_array) or len(bitstrings) != len(
        amplitude_array
    ):
        raise ValueError("bitstrings, phases, and amplitudes must align.")
    ordered_indices = sorted(
        range(len(bitstrings)),
        key=lambda index: (
            -hamming_weight(bitstrings[index]),
            bitstrings[index],
        ),
    )
    ordered_bitstrings = [bitstrings[index] for index in ordered_indices]
    return (
        ordered_bitstrings,
        phase_array[ordered_indices],
        amplitude_array[ordered_indices],
    )


def unchanged_mode_order(
    bitstrings: Sequence[Bitstring],
    phases: ArrayLike,
    amplitudes: ArrayLike,
) -> tuple[list[Bitstring], FloatArray, FloatArray]:
    """Return copied mode data without changing its order.

    Parameters
    ----------
    bitstrings
        Ordered binary mode labels.
    phases
        Phase value for each mode.
    amplitudes
        Amplitude for each mode.

    Returns
    -------
    tuple
        Copied bitstrings, phases, and amplitudes.
    """
    return (
        list(bitstrings),
        np.asarray(phases, dtype=float).copy(),
        np.asarray(amplitudes, dtype=float).copy(),
    )


def descending_weights(weights: ArrayLike) -> FloatArray:
    """Return mode weights sorted from largest to smallest.

    Parameters
    ----------
    weights
        Mode probabilities.

    Returns
    -------
    numpy.ndarray
        Descending copy of the input weights.
    """
    weight_array = np.asarray(weights, dtype=float)
    return np.asarray(weight_array[np.argsort(-weight_array)], dtype=float)


def average_weights_by_hamming_class(
    bitstrings: Sequence[Bitstring],
    weights: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return normalized Hamming weights and class-average mode weights.

    Parameters
    ----------
    bitstrings
        Binary labels associated with the mode weights.
    weights
        Mode probabilities.

    Returns
    -------
    tuple
        Normalized Hamming-class coordinates and average class weights.
    """
    if not bitstrings:
        raise ValueError("bitstrings cannot be empty.")
    weight_array = np.asarray(weights, dtype=float)
    if len(bitstrings) != len(weight_array):
        raise ValueError("bitstrings and weights must have equal length.")
    qubit_count = len(bitstrings[0])
    hamming_weights = np.asarray(
        [hamming_weight(bitstring) for bitstring in bitstrings],
        dtype=int,
    )
    normalized_classes = np.arange(qubit_count + 1, dtype=float) / qubit_count
    class_averages = np.asarray(
        [
            np.mean(weight_array[hamming_weights == class_weight])
            for class_weight in range(qubit_count + 1)
        ],
        dtype=float,
    )
    return normalized_classes, class_averages
