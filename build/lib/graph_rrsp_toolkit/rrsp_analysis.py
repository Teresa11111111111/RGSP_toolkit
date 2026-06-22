"""Remote state-preparation success probabilities and scaling analyses."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .amplitude_compensation import (
    descending_weights,
    high_hamming_first_order,
    loss_compensated_amplitudes,
    uniform_amplitudes,
)
from .graph_topologies import TopologyFactory, brickwork_adjacency
from .loss_models import efficiency_values
from .phase_encoding import Bitstring, phase_values
from .phase_noise import (
    FidelityStatistics,
    NoiseModel,
    cumulative_fidelity_curve,
    independent_fidelity_curve,
    phase_noise_fidelity_statistics,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FidelityGainSeries:
    """Uniform, loss-compensated, and difference fidelity series."""

    uniform: FloatArray
    loss_compensated: FloatArray
    gain: FloatArray


def rrsp_success_probability(
    qubit_count: int,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.7,
    zero_efficiency: float = 0.95,
) -> float:
    """Return the heralded remote-state-preparation success probability.

    Parameters
    ----------
    qubit_count
        Number of logical qubits encoded in the optical resource.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for a logical-one interaction.
    zero_efficiency
        Internal efficiency for a logical-zero interaction.

    Returns
    -------
    float
        Closed-form success probability.
    """
    internal_factor = (
        2.0
        * zero_efficiency
        * one_efficiency
        / (zero_efficiency + one_efficiency)
    )
    return float(
        transmission_efficiency
        * detector_efficiency
        * internal_factor**qubit_count
    )


def rrsp_success_probabilities(
    qubit_counts: Iterable[int],
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> FloatArray:
    """Return success probabilities for a sequence of qubit counts.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for a logical-one interaction.
    zero_efficiency
        Internal efficiency for a logical-zero interaction.

    Returns
    -------
    numpy.ndarray
        Success probability for each input qubit count.
    """
    return np.asarray(
        [
            rrsp_success_probability(
                qubit_count,
                transmission_efficiency,
                detector_efficiency,
                one_efficiency,
                zero_efficiency,
            )
            for qubit_count in qubit_counts
        ],
        dtype=float,
    )


def compensated_mode_data(
    qubit_count: int,
    topology_factory: TopologyFactory = brickwork_adjacency,
    seed: int = 42,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> tuple[list[Bitstring], FloatArray, FloatArray]:
    """Return bitstrings, phases, and loss-compensated amplitudes.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    topology_factory
        Adjacency-matrix factory.
    seed
        Seed used to draw local phase angles.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    tuple
        Ordered bitstrings, encoded phases, and compensated amplitudes.
    """
    generator = np.random.default_rng(seed + qubit_count)
    local_angles = generator.uniform(0.0, 2.0 * np.pi, qubit_count)
    bitstrings, encoded_phases = phase_values(
        local_angles,
        topology_factory(qubit_count),
    )
    efficiencies = efficiency_values(
        bitstrings,
        transmission_efficiency,
        detector_efficiency,
        one_efficiency,
        zero_efficiency,
    )
    amplitudes = loss_compensated_amplitudes(efficiencies)
    return bitstrings, encoded_phases, amplitudes


def mode_weights(
    qubit_count: int,
    amplitude_mode: str = "loss-compensated",
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> tuple[list[Bitstring], FloatArray]:
    """Return temporal-mode weights for a selected amplitude strategy.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    amplitude_mode
        Either ``"uniform"`` or ``"loss-compensated"``.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    tuple
        Lexicographic bitstrings and normalized mode probabilities.
    """
    local_angles = np.zeros(qubit_count, dtype=float)
    bitstrings, encoded_phases = phase_values(
        local_angles,
        brickwork_adjacency(qubit_count),
    )
    if amplitude_mode == "uniform":
        amplitudes = uniform_amplitudes(len(encoded_phases))
    elif amplitude_mode == "loss-compensated":
        efficiencies = efficiency_values(
            bitstrings,
            transmission_efficiency,
            detector_efficiency,
            one_efficiency,
            zero_efficiency,
        )
        amplitudes = loss_compensated_amplitudes(efficiencies)
    else:
        raise ValueError(
            "amplitude_mode must be 'uniform' or 'loss-compensated'."
        )
    return bitstrings, np.asarray(amplitudes**2, dtype=float)


def fidelity_for_setting(
    qubit_count: int,
    noise_scale: float,
    amplitude_mode: str,
    noise_model: NoiseModel,
    order_mode: str = "natural",
    topology_factory: TopologyFactory = brickwork_adjacency,
    sample_count: int = 1600,
    seed: int = 42,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> float:
    """Return mean Monte Carlo fidelity for one experimental setting.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    noise_scale
        Gaussian phase-increment standard deviation.
    amplitude_mode
        Either ``"uniform"`` or ``"loss-compensated"``.
    noise_model
        Either independent or cumulative phase noise.
    order_mode
        Either ``"natural"`` or ``"high-hamming-first"``.
    topology_factory
        Adjacency-matrix factory.
    sample_count
        Number of Monte Carlo realizations.
    seed
        Base random seed.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    float
        Mean fidelity over the requested realizations.
    """
    generator = np.random.default_rng(seed + qubit_count)
    local_angles = generator.uniform(0.0, 2.0 * np.pi, qubit_count)
    bitstrings, encoded_phases = phase_values(
        local_angles,
        topology_factory(qubit_count),
    )

    if amplitude_mode == "uniform":
        amplitudes = uniform_amplitudes(len(encoded_phases))
    elif amplitude_mode == "loss-compensated":
        efficiencies = efficiency_values(
            bitstrings,
            transmission_efficiency,
            detector_efficiency,
            one_efficiency,
            zero_efficiency,
        )
        amplitudes = loss_compensated_amplitudes(efficiencies)
    else:
        raise ValueError("Unknown amplitude_mode.")

    if order_mode == "high-hamming-first":
        _, encoded_phases, amplitudes = high_hamming_first_order(
            bitstrings,
            encoded_phases,
            amplitudes,
        )
    elif order_mode != "natural":
        raise ValueError(
            "order_mode must be 'natural' or 'high-hamming-first'."
        )

    statistics = phase_noise_fidelity_statistics(
        encoded_phases,
        np.asarray([noise_scale], dtype=float),
        noise_model,
        amplitudes,
        sample_count,
        seed + 999 * qubit_count,
    )
    return float(statistics.means[0])


def fidelity_gains_by_qubit_count(
    qubit_counts: Iterable[int],
    noise_scale: float = np.pi / 10.0,
    topology_factory: TopologyFactory = brickwork_adjacency,
    sample_count: int = 2000,
    seed: int = 42,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> dict[str, FidelityGainSeries]:
    """Return loss-compensation fidelity gains across system sizes.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    noise_scale
        Fixed phase-noise standard deviation.
    topology_factory
        Adjacency-matrix factory.
    sample_count
        Number of Monte Carlo realizations per setting.
    seed
        Base random seed.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    dict
        Uniform, compensated, and gain series for three noise settings.
    """
    settings: Mapping[str, tuple[NoiseModel, str]] = {
        "Independent": ("independent", "natural"),
        "Cumulative natural": ("cumulative", "natural"),
        "Cumulative high-H first": (
            "cumulative",
            "high-hamming-first",
        ),
    }
    counts = list(qubit_counts)
    results: dict[str, FidelityGainSeries] = {}

    for label, (noise_model, order_mode) in settings.items():
        uniform_values: list[float] = []
        compensated_values: list[float] = []
        for qubit_count in counts:
            uniform_values.append(
                fidelity_for_setting(
                    qubit_count,
                    noise_scale,
                    "uniform",
                    noise_model,
                    order_mode,
                    topology_factory,
                    sample_count,
                    seed,
                    transmission_efficiency,
                    detector_efficiency,
                    one_efficiency,
                    zero_efficiency,
                )
            )
            compensated_values.append(
                fidelity_for_setting(
                    qubit_count,
                    noise_scale,
                    "loss-compensated",
                    noise_model,
                    order_mode,
                    topology_factory,
                    sample_count,
                    seed + 12345,
                    transmission_efficiency,
                    detector_efficiency,
                    one_efficiency,
                    zero_efficiency,
                )
            )
        uniform_array = np.asarray(uniform_values, dtype=float)
        compensated_array = np.asarray(compensated_values, dtype=float)
        results[label] = FidelityGainSeries(
            uniform=uniform_array,
            loss_compensated=compensated_array,
            gain=compensated_array - uniform_array,
        )
    return results


def topology_noise_statistics(
    qubit_count: int,
    topology_map: Mapping[str, TopologyFactory],
    noise_scales: FloatArray,
    noise_model: NoiseModel = "independent",
    amplitude_mode: str = "uniform",
    sample_count: int = 1000,
    seed: int = 42,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> dict[str, FidelityStatistics]:
    """Return fidelity statistics for several graph topologies.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    topology_map
        Labels mapped to adjacency factories.
    noise_scales
        Phase-noise standard deviations.
    noise_model
        Either independent or cumulative phase noise.
    amplitude_mode
        Either ``"uniform"`` or ``"loss-compensated"``.
    sample_count
        Number of Monte Carlo realizations per point.
    seed
        Base random seed.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    dict
        Topology labels mapped to Monte Carlo statistics.
    """
    generator = np.random.default_rng(seed)
    local_angles = generator.uniform(0.0, 2.0 * np.pi, qubit_count)
    results: dict[str, FidelityStatistics] = {}
    for label, topology_factory in topology_map.items():
        bitstrings, encoded_phases = phase_values(
            local_angles,
            topology_factory(qubit_count),
        )
        if amplitude_mode == "uniform":
            amplitudes = uniform_amplitudes(len(encoded_phases))
        elif amplitude_mode == "loss-compensated":
            efficiencies = efficiency_values(
                bitstrings,
                transmission_efficiency,
                detector_efficiency,
                one_efficiency,
                zero_efficiency,
            )
            amplitudes = loss_compensated_amplitudes(efficiencies)
        else:
            raise ValueError(
                "amplitude_mode must be 'uniform' or 'loss-compensated'."
            )
        results[label] = phase_noise_fidelity_statistics(
            encoded_phases,
            noise_scales,
            noise_model,
            amplitudes,
            sample_count,
            seed,
        )
    return results


def qubit_count_noise_statistics(
    qubit_counts: Iterable[int],
    topology_factory: TopologyFactory,
    noise_scales: FloatArray,
    noise_model: NoiseModel,
    amplitude_mode: str,
    order_mode: str = "natural",
    sample_count: int = 1000,
    seed: int = 42,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> dict[int, FidelityStatistics]:
    """Return fidelity statistics indexed by logical-qubit count.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    topology_factory
        Adjacency-matrix factory.
    noise_scales
        Phase-noise standard deviations.
    noise_model
        Either independent or cumulative phase noise.
    amplitude_mode
        Either ``"uniform"`` or ``"loss-compensated"``.
    order_mode
        Either ``"natural"`` or ``"high-hamming-first"``.
    sample_count
        Number of Monte Carlo realizations per point.
    seed
        Base random seed.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    dict
        Qubit counts mapped to Monte Carlo statistics.
    """
    results: dict[int, FidelityStatistics] = {}
    for qubit_count in qubit_counts:
        generator = np.random.default_rng(seed + qubit_count)
        local_angles = generator.uniform(0.0, 2.0 * np.pi, qubit_count)
        bitstrings, encoded_phases = phase_values(
            local_angles,
            topology_factory(qubit_count),
        )
        if amplitude_mode == "uniform":
            amplitudes = uniform_amplitudes(len(encoded_phases))
        elif amplitude_mode == "loss-compensated":
            efficiencies = efficiency_values(
                bitstrings,
                transmission_efficiency,
                detector_efficiency,
                one_efficiency,
                zero_efficiency,
            )
            amplitudes = loss_compensated_amplitudes(efficiencies)
        else:
            raise ValueError(
                "amplitude_mode must be 'uniform' or 'loss-compensated'."
            )
        if order_mode == "high-hamming-first":
            _, encoded_phases, amplitudes = high_hamming_first_order(
                bitstrings,
                encoded_phases,
                amplitudes,
            )
        elif order_mode != "natural":
            raise ValueError(
                "order_mode must be 'natural' or 'high-hamming-first'."
            )
        results[qubit_count] = phase_noise_fidelity_statistics(
            encoded_phases,
            noise_scales,
            noise_model,
            amplitudes,
            sample_count,
            seed + 1000 * qubit_count,
        )
    return results


def final_comparison_curves(
    qubit_counts: Iterable[int],
    noise_scales: FloatArray,
    amplitude_mode: str = "loss-compensated",
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.3,
    zero_efficiency: float = 0.95,
) -> dict[int, dict[str, FloatArray]]:
    """Return analytic independent and cumulative comparison curves.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    noise_scales
        Phase-noise standard deviations.
    amplitude_mode
        Either ``"uniform"`` or ``"loss-compensated"``.
    transmission_efficiency
        Shared channel transmission efficiency.
    detector_efficiency
        Shared detector efficiency.
    one_efficiency
        Internal efficiency for logical-one components.
    zero_efficiency
        Internal efficiency for logical-zero components.

    Returns
    -------
    dict
        Qubit counts mapped to three analytic fidelity curves.
    """
    curves: dict[int, dict[str, FloatArray]] = {}
    for qubit_count in qubit_counts:
        _, weights = mode_weights(
            qubit_count,
            amplitude_mode,
            transmission_efficiency,
            detector_efficiency,
            one_efficiency,
            zero_efficiency,
        )
        ordered_weights = descending_weights(weights)
        curves[qubit_count] = {
            "Independent phase noise": independent_fidelity_curve(
                weights,
                noise_scales,
            ),
            "Cumulative, highest weight first": cumulative_fidelity_curve(
                ordered_weights,
                noise_scales,
            ),
            "Cumulative noise": cumulative_fidelity_curve(
                weights,
                noise_scales,
            ),
        }
    return curves
