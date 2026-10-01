"""Consistency checks for graph states, noise models, and loss formulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .amplitude_compensation import compensation_coefficients
from .graph_states import (
    controlled_z_graph_state,
    graph_phase_state,
    normalized_state,
    pure_state_fidelity,
    weighted_state_from_phase_mapping,
)
from .graph_topologies import (
    brickwork_adjacency,
    clique_adjacency,
    line_adjacency,
    star_adjacency,
)
from .loss_models import efficiencies_by_bitstring
from .phase_encoding import phases_by_bitstring
from .phase_noise import (
    independent_average_fidelity,
    phase_noise_fidelity_statistics,
)
from .rrsp_analysis import rrsp_success_probability


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of one numerical consistency check."""

    name: str
    passed: bool
    value: float
    reference: float
    tolerance: float


def phase_construction_fidelity(
    qubit_count: int = 5,
    seed: int = 42,
) -> float:
    """Return fidelity between phase and controlled-Z constructions.

    Parameters
    ----------
    qubit_count
        Number of qubits in the brickwork graph.
    seed
        Random seed for local phase angles.

    Returns
    -------
    float
        Fidelity between independently constructed graph states.
    """
    generator = np.random.default_rng(seed)
    local_angles = generator.uniform(0.0, 2.0 * np.pi, qubit_count)
    adjacency = brickwork_adjacency(qubit_count)
    phase_state = graph_phase_state(local_angles, adjacency)
    controlled_z_state = controlled_z_graph_state(local_angles, adjacency)
    return pure_state_fidelity(phase_state, controlled_z_state)


def loss_success_probability_pair(
    qubit_count: int = 6,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.7,
    zero_efficiency: float = 0.95,
) -> tuple[float, float]:
    """Return numerical and closed-form success probabilities.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
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
        Probability from normalized inverse-loss coefficients and the
        closed-form probability.
    """
    efficiencies = efficiencies_by_bitstring(
        qubit_count,
        transmission_efficiency,
        detector_efficiency,
        one_efficiency,
        zero_efficiency,
    )
    normalization = compensation_coefficients(efficiencies)[1]
    numerical_probability = 2**qubit_count / normalization
    closed_form_probability = rrsp_success_probability(
        qubit_count,
        transmission_efficiency,
        detector_efficiency,
        one_efficiency,
        zero_efficiency,
    )
    return float(numerical_probability), closed_form_probability


def compensated_state_fidelity(
    qubit_count: int = 6,
    seed: int = 42,
    transmission_efficiency: float = 0.8,
    detector_efficiency: float = 0.9,
    one_efficiency: float = 0.7,
    zero_efficiency: float = 0.95,
) -> float:
    """Return ideal-to-compensated state fidelity for a line graph.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    seed
        Legacy NumPy seed used by the original notebook.
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
        Fidelity reproduced from the phase-noise notebook.
    """
    legacy_generator = np.random.RandomState(seed)
    local_angles = legacy_generator.uniform(0.0, 2.0 * np.pi, qubit_count)
    phase_mapping = phases_by_bitstring(
        local_angles,
        line_adjacency(qubit_count),
    )
    ideal_state = graph_phase_state(
        local_angles,
        line_adjacency(qubit_count),
    )
    efficiencies = efficiencies_by_bitstring(
        qubit_count,
        transmission_efficiency,
        detector_efficiency,
        one_efficiency,
        zero_efficiency,
    )
    amplitudes, compensation_normalization = compensation_coefficients(
        efficiencies
    )
    if compensation_normalization <= 0.0:
        raise RuntimeError("Compensation normalization must be positive.")
    compensated_state = normalized_state(
        weighted_state_from_phase_mapping(phase_mapping, amplitudes)
    )
    return pure_state_fidelity(ideal_state, compensated_state)


def topology_fidelity_spread(
    qubit_count: int = 6,
    noise_scale: float = 0.5,
    sample_count: int = 500,
    seed: int = 123,
) -> float:
    """Return the Monte Carlo mean-fidelity spread across topologies.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    noise_scale
        Independent phase-noise standard deviation.
    sample_count
        Number of Monte Carlo samples per topology.
    seed
        Seed reused for each topology.

    Returns
    -------
    float
        Difference between largest and smallest mean fidelity.
    """
    legacy_generator = np.random.RandomState(42)
    local_angles = legacy_generator.uniform(0.0, 2.0 * np.pi, qubit_count)
    adjacency_matrices = (
        line_adjacency(qubit_count),
        star_adjacency(qubit_count),
        clique_adjacency(qubit_count),
        brickwork_adjacency(qubit_count),
    )
    means = []
    for adjacency in adjacency_matrices:
        phase_mapping = phases_by_bitstring(local_angles, adjacency)
        phase_array = np.asarray(list(phase_mapping.values()), dtype=float)
        statistics = phase_noise_fidelity_statistics(
            phase_array,
            np.asarray([noise_scale], dtype=float),
            "independent",
            sample_count=sample_count,
            seed=seed,
        )
        means.append(float(statistics.means[0]))
    return float(np.max(means) - np.min(means))


def validation_results() -> list[ValidationResult]:
    """Return the package validation suite results.

    Returns
    -------
    list of ValidationResult
        Numerical checks against identities and notebook reference values.
    """
    construction_value = phase_construction_fidelity()
    numerical_probability, closed_form_probability = (
        loss_success_probability_pair()
    )
    compensated_fidelity = compensated_state_fidelity()
    topology_spread = topology_fidelity_spread()
    analytic_reference = independent_average_fidelity(6, 0.5)

    legacy_generator = np.random.RandomState(42)
    local_angles = legacy_generator.uniform(0.0, 2.0 * np.pi, 6)
    phase_mapping = phases_by_bitstring(local_angles, line_adjacency(6))
    phase_array = np.asarray(list(phase_mapping.values()), dtype=float)
    notebook_noise_scales = np.linspace(0.0, 1.0, 21)
    monte_carlo_statistics = phase_noise_fidelity_statistics(
        phase_array,
        notebook_noise_scales,
        "independent",
        sample_count=500,
        seed=42,
    )
    midpoint_index = int(np.argmin(np.abs(notebook_noise_scales - 0.5)))

    return [
        ValidationResult(
            "phase construction fidelity",
            abs(construction_value - 1.0) <= 1e-12,
            construction_value,
            1.0,
            1e-12,
        ),
        ValidationResult(
            "loss success probability identity",
            abs(numerical_probability - closed_form_probability) <= 1e-14,
            numerical_probability,
            closed_form_probability,
            1e-14,
        ),
        ValidationResult(
            "compensated-state fidelity",
            abs(compensated_fidelity - 0.9658608070218452) <= 1e-12,
            compensated_fidelity,
            0.9658608070218452,
            1e-12,
        ),
        ValidationResult(
            "topology-independent Monte Carlo fidelity",
            topology_spread <= 1e-12,
            topology_spread,
            0.0,
            1e-12,
        ),
        ValidationResult(
            "independent analytic fidelity at delta=0.5",
            abs(analytic_reference - 0.7822570208361428) <= 1e-12,
            analytic_reference,
            0.7822570208361428,
            1e-12,
        ),
        ValidationResult(
            "notebook Monte Carlo mean at delta=0.5",
            abs(
                monte_carlo_statistics.means[midpoint_index] - 0.782640
            )
            <= 5e-7,
            float(monte_carlo_statistics.means[midpoint_index]),
            0.782640,
            5e-7,
        ),
        ValidationResult(
            "notebook Monte Carlo std at delta=0.5",
            abs(
                monte_carlo_statistics.standard_deviations[midpoint_index]
                - 0.035892
            )
            <= 5e-7,
            float(
                monte_carlo_statistics.standard_deviations[midpoint_index]
            ),
            0.035892,
            5e-7,
        ),
    ]


def validation_passes(results: list[ValidationResult] | None = None) -> bool:
    """Return whether every validation result passes.

    Parameters
    ----------
    results
        Optional pre-evaluated results.

    Returns
    -------
    bool
        True only when all checks pass.
    """
    evaluated_results = validation_results() if results is None else results
    return all(result.passed for result in evaluated_results)
