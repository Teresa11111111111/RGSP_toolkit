"""Independent and cumulative phase-noise models and fidelity predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .graph_states import normalized_state, pulse_state, pure_state_fidelity

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
NoiseModel = Literal["independent", "cumulative"]


@dataclass(frozen=True)
class FidelityStatistics:
    """Monte Carlo fidelity means and standard deviations."""

    noise_scales: FloatArray
    means: FloatArray
    standard_deviations: FloatArray


def phase_error_values(
    mode_count: int,
    noise_scale: float,
    noise_model: NoiseModel,
    random_generator: np.random.Generator,
) -> FloatArray:
    """Return phase errors for one temporal-mode realization.

    Parameters
    ----------
    mode_count
        Number of temporal modes.
    noise_scale
        Standard deviation of independent Gaussian increments.
    noise_model
        Either ``"independent"`` or ``"cumulative"``.
    random_generator
        NumPy random generator used for reproducible sampling.

    Returns
    -------
    numpy.ndarray
        Phase error assigned to each temporal mode.
    """
    increments = random_generator.normal(
        loc=0.0,
        scale=noise_scale,
        size=mode_count,
    )
    if noise_model == "independent":
        return np.asarray(increments, dtype=float)
    if noise_model == "cumulative":
        return np.asarray(np.cumsum(increments), dtype=float)
    raise ValueError("noise_model must be 'independent' or 'cumulative'.")


def noisy_pulse_state(
    pulse: ArrayLike,
    noise_scale: float,
    noise_model: NoiseModel = "independent",
    random_generator: np.random.Generator | None = None,
) -> ComplexArray:
    """Return a normalized pulse after Gaussian phase noise.

    Parameters
    ----------
    pulse
        Ideal complex temporal-mode amplitudes.
    noise_scale
        Standard deviation of independent Gaussian increments.
    noise_model
        Either independent mode noise or cumulative random-walk noise.
    random_generator
        Optional NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Noisy normalized pulse state.
    """
    generator = (
        np.random.default_rng()
        if random_generator is None
        else random_generator
    )
    pulse_array = np.asarray(pulse, dtype=complex)
    phase_errors = phase_error_values(
        len(pulse_array),
        noise_scale,
        noise_model,
        generator,
    )
    return normalized_state(pulse_array * np.exp(1j * phase_errors))


def phase_noise_fidelity_sample(
    phases: ArrayLike,
    noise_scale: float,
    noise_model: NoiseModel = "independent",
    amplitudes: ArrayLike | None = None,
    random_generator: np.random.Generator | None = None,
) -> float:
    """Return one noisy-pulse fidelity sample.

    Parameters
    ----------
    phases
        Ideal phase assigned to each temporal mode.
    noise_scale
        Standard deviation of independent Gaussian increments.
    noise_model
        Either independent mode noise or cumulative random-walk noise.
    amplitudes
        Optional ideal temporal-mode amplitudes.
    random_generator
        Optional NumPy random generator.

    Returns
    -------
    float
        Fidelity between ideal and noisy pulse states.
    """
    ideal_pulse = pulse_state(phases, amplitudes)
    noisy_pulse = noisy_pulse_state(
        ideal_pulse,
        noise_scale,
        noise_model,
        random_generator,
    )
    return pure_state_fidelity(ideal_pulse, noisy_pulse)


def phase_noise_fidelity_statistics(
    phases: ArrayLike,
    noise_scales: ArrayLike,
    noise_model: NoiseModel = "independent",
    amplitudes: ArrayLike | None = None,
    sample_count: int = 1000,
    seed: int = 42,
) -> FidelityStatistics:
    """Return Monte Carlo fidelity statistics over noise strengths.

    Parameters
    ----------
    phases
        Ideal phase assigned to each temporal mode.
    noise_scales
        Gaussian increment standard deviations.
    noise_model
        Either independent mode noise or cumulative random-walk noise.
    amplitudes
        Optional ideal temporal-mode amplitudes.
    sample_count
        Number of Monte Carlo realizations per noise strength.
    seed
        Random seed shared across the full sweep.

    Returns
    -------
    FidelityStatistics
        Noise values, sample means, and sample standard deviations.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be at least one.")
    phase_array = np.asarray(phases, dtype=float)
    noise_array = np.asarray(noise_scales, dtype=float)
    generator = np.random.default_rng(seed)
    means: list[float] = []
    standard_deviations: list[float] = []

    for noise_scale in noise_array:
        samples = np.asarray(
            [
                phase_noise_fidelity_sample(
                    phase_array,
                    float(noise_scale),
                    noise_model,
                    amplitudes,
                    generator,
                )
                for sample_index in range(sample_count)
            ],
            dtype=float,
        )
        means.append(float(np.mean(samples)))
        standard_deviations.append(float(np.std(samples)))

    return FidelityStatistics(
        noise_scales=noise_array,
        means=np.asarray(means, dtype=float),
        standard_deviations=np.asarray(standard_deviations, dtype=float),
    )


def independent_average_fidelity(
    qubit_count: int,
    noise_scale: float,
) -> float:
    """Return analytic average fidelity for uniform independent noise.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    noise_scale
        Standard deviation of each independent phase error.

    Returns
    -------
    float
        Expected fidelity for ``2**qubit_count`` uniform modes.
    """
    mode_count = 2**qubit_count
    return float(
        1.0 / mode_count
        + (1.0 - 1.0 / mode_count) * np.exp(-(noise_scale**2))
    )


def cumulative_average_fidelity(
    qubit_count: int,
    noise_scale: float,
) -> float:
    """Return analytic average fidelity for uniform cumulative noise.

    Parameters
    ----------
    qubit_count
        Number of logical qubits.
    noise_scale
        Standard deviation of each random-walk increment.

    Returns
    -------
    float
        Expected fidelity for uniformly weighted temporal modes.
    """
    mode_count = 2**qubit_count
    total = float(mode_count)
    for lag in range(1, mode_count):
        total += (
            2.0
            * (mode_count - lag)
            * np.exp(-0.5 * lag * noise_scale**2)
        )
    return float(total / mode_count**2)


def independent_weighted_fidelity(
    weights: ArrayLike,
    noise_scale: float,
) -> float:
    """Return analytic independent-noise fidelity for arbitrary weights.

    Parameters
    ----------
    weights
        Normalized temporal-mode probabilities.
    noise_scale
        Standard deviation of each independent phase error.

    Returns
    -------
    float
        Expected weighted fidelity.
    """
    weight_array = np.asarray(weights, dtype=float)
    purity = float(np.sum(weight_array**2))
    decay = float(np.exp(-(noise_scale**2)))
    return float(decay + (1.0 - decay) * purity)


def weight_autocorrelation(weights: ArrayLike) -> FloatArray:
    """Return nonnegative-lag autocorrelation of mode weights.

    Parameters
    ----------
    weights
        Temporal-mode probabilities.

    Returns
    -------
    numpy.ndarray
        Autocorrelation values from zero lag onward.
    """
    weight_array = np.asarray(weights, dtype=float)
    full_correlation = np.correlate(weight_array, weight_array, mode="full")
    mode_count = len(weight_array)
    return np.asarray(full_correlation[mode_count - 1 :], dtype=float)


def cumulative_weighted_fidelity(
    weights: ArrayLike,
    noise_scale: float,
) -> float:
    """Return analytic cumulative-noise fidelity for arbitrary weights.

    Parameters
    ----------
    weights
        Normalized temporal-mode probabilities in pulse order.
    noise_scale
        Standard deviation of each random-walk increment.

    Returns
    -------
    float
        Expected weighted fidelity.
    """
    correlations = weight_autocorrelation(weights)
    lags = np.arange(len(correlations), dtype=float)
    decays = np.exp(-0.5 * lags * noise_scale**2)
    return float(
        correlations[0] + 2.0 * np.dot(correlations[1:], decays[1:])
    )


def independent_fidelity_curve(
    weights: ArrayLike,
    noise_scales: ArrayLike,
) -> FloatArray:
    """Return the analytic independent-noise fidelity curve.

    Parameters
    ----------
    weights
        Normalized temporal-mode probabilities.
    noise_scales
        Gaussian phase-error standard deviations.

    Returns
    -------
    numpy.ndarray
        Expected fidelity at every noise strength.
    """
    weight_array = np.asarray(weights, dtype=float)
    noise_array = np.asarray(noise_scales, dtype=float)
    purity = np.sum(weight_array**2)
    variances = noise_array**2
    decays = np.exp(-variances)
    return np.asarray(decays + (1.0 - decays) * purity, dtype=float)


def cumulative_fidelity_curve(
    weights: ArrayLike,
    noise_scales: ArrayLike,
) -> FloatArray:
    """Return the analytic cumulative-noise fidelity curve.

    Parameters
    ----------
    weights
        Normalized temporal-mode probabilities in pulse order.
    noise_scales
        Random-walk increment standard deviations.

    Returns
    -------
    numpy.ndarray
        Expected fidelity at every noise strength.
    """
    correlations = weight_autocorrelation(weights)
    lags = np.arange(len(correlations), dtype=float)
    variances = np.asarray(noise_scales, dtype=float) ** 2
    decay_matrix = np.exp(-0.5 * np.outer(variances, lags[1:]))
    return np.asarray(
        correlations[0] + 2.0 * (decay_matrix @ correlations[1:]),
        dtype=float,
    )
