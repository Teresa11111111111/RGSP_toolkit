"""Mode-dependent and Hamming-weight-dependent transmission models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase_encoding import Bitstring, computational_bitstrings

FloatArray = NDArray[np.float64]


def hamming_weight(bitstring: Bitstring) -> int:
    """Return the Hamming weight of a bitstring.

    Parameters
    ----------
    bitstring
        Binary computational-basis label.

    Returns
    -------
    int
        Number of entries equal to one.
    """
    return int(sum(bitstring))


def bitstring_efficiency(
    bitstring: Bitstring,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> float:
    """Return the transmission efficiency of one temporal mode.

    Parameters
    ----------
    bitstring
        Binary mode label.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for each logical-one component.
    zero_efficiency
        Internal efficiency for each logical-zero component.

    Returns
    -------
    float
        Mode transmission efficiency.
    """
    qubit_count = len(bitstring)
    weight = hamming_weight(bitstring)
    return float(
        transmission_efficiency
        * detector_efficiency
        * one_efficiency**weight
        * zero_efficiency ** (qubit_count - weight)
    )


def efficiencies_by_bitstring(
    qubit_count: int,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> dict[Bitstring, float]:
    """Return mode efficiencies keyed by computational-basis bitstring.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for each logical-one component.
    zero_efficiency
        Internal efficiency for each logical-zero component.

    Returns
    -------
    dict
        Bitstring-to-efficiency mapping in lexicographic order.
    """
    return {
        bitstring: bitstring_efficiency(
            bitstring,
            transmission_efficiency,
            detector_efficiency,
            one_efficiency,
            zero_efficiency,
        )
        for bitstring in computational_bitstrings(qubit_count)
    }


def efficiency_values(
    bitstrings: Sequence[Bitstring],
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> FloatArray:
    """Return ordered transmission efficiencies for temporal modes.

    Parameters
    ----------
    bitstrings
        Ordered binary mode labels.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for each logical-one component.
    zero_efficiency
        Internal efficiency for each logical-zero component.

    Returns
    -------
    numpy.ndarray
        Efficiency associated with each input bitstring.
    """
    return np.asarray(
        [
            bitstring_efficiency(
                bitstring,
                transmission_efficiency,
                detector_efficiency,
                one_efficiency,
                zero_efficiency,
            )
            for bitstring in bitstrings
        ],
        dtype=float,
    )
