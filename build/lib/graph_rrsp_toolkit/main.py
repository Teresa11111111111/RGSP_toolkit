"""Command-line entry point for validation and figure reproduction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .amplitude_compensation import (
    average_weights_by_hamming_class,
    descending_weights,
)
from .graph_topologies import (
    TopologyFactory,
    brickwork_adjacency,
    topology_factories,
)
from .phase_noise import (
    FidelityStatistics,
    cumulative_fidelity_curve,
    independent_average_fidelity,
    independent_fidelity_curve,
)
from .plotting import (
    analytic_qubit_count_figure,
    dual_noise_topology_figure,
    fidelity_gain_figure,
    final_phase_noise_comparison_figure,
    graph_figure,
    hamming_weight_distribution_figure,
    ordering_comparison_figure,
    phase_noise_fidelity_figure,
    qubit_count_monte_carlo_figure,
    rrsp_probability_figure,
    saved_figure_path,
    show_figures,
    three_case_scaling_figure,
    topology_comparison_figure,
)
from .rrsp_analysis import (
    FidelityGainSeries,
    fidelity_gains_by_qubit_count,
    final_comparison_curves,
    mode_weights,
    qubit_count_noise_statistics,
    rrsp_success_probabilities,
    topology_noise_statistics,
)
from .validation import ValidationResult, validation_results

FloatArray = NDArray[np.float64]


def command_line_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the research workflow.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate graph-state identities and reproduce research figures."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "thesis"),
        default="quick",
        help=(
            "Use a fast smoke-test profile or the notebook sampling settings."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run numerical checks without producing figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving them.",
    )
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="Append timestamps to output filenames.",
    )
    return parser


def printed_validation_report(results: Sequence[ValidationResult]) -> None:
    """Print a compact numerical validation report.

    Parameters
    ----------
    results
        Evaluated validation results.

    Returns
    -------
    None
        The report is written to standard output.
    """
    print("\nValidation report")
    print("-----------------")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status:4s}  {result.name}: "
            f"value={result.value:.12g}, "
            f"reference={result.reference:.12g}"
        )


def profile_settings(profile: str) -> dict[str, int]:
    """Return simulation sizes for a named execution profile.

    Parameters
    ----------
    profile
        Either ``"quick"`` or ``"thesis"``.

    Returns
    -------
    dict
        Monte Carlo sample counts and dense-curve point counts.
    """
    if profile == "quick":
        return {
            "phase_samples": 40,
            "general_samples": 40,
            "gain_samples": 80,
            "dense_points": 31,
        }
    if profile == "thesis":
        return {
            "phase_samples": 500,
            "general_samples": 1000,
            "gain_samples": 2000,
            "dense_points": 301,
        }
    raise ValueError("profile must be 'quick' or 'thesis'.")


def independent_curves_by_qubit_count(
    qubit_counts: Sequence[int],
    noise_scales: FloatArray,
) -> dict[int, FloatArray]:
    """Return uniform independent-noise curves indexed by qubit count.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    noise_scales
        Independent phase-noise standard deviations.

    Returns
    -------
    dict
        Qubit counts mapped to analytic fidelity curves.
    """
    return {
        qubit_count: np.asarray(
            [
                independent_average_fidelity(qubit_count, noise_scale)
                for noise_scale in noise_scales
            ],
            dtype=float,
        )
        for qubit_count in qubit_counts
    }


def analytic_curves_for_case(
    qubit_counts: Sequence[int],
    noise_scales: FloatArray,
    amplitude_mode: str,
    noise_model: str,
    order_mode: str,
) -> dict[int, FloatArray]:
    """Return weighted analytic curves for a noise-and-ordering case.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    noise_scales
        Phase-noise standard deviations.
    amplitude_mode
        Either uniform or loss-compensated.
    noise_model
        Either independent or cumulative.
    order_mode
        Either natural or high-Hamming-first.

    Returns
    -------
    dict
        Qubit counts mapped to analytic fidelity curves.
    """
    curves: dict[int, FloatArray] = {}
    for qubit_count in qubit_counts:
        _, weights = mode_weights(qubit_count, amplitude_mode)
        if order_mode == "high-hamming-first":
            weights = descending_weights(weights)
        if noise_model == "independent":
            curves[qubit_count] = independent_fidelity_curve(
                weights,
                noise_scales,
            )
        elif noise_model == "cumulative":
            curves[qubit_count] = cumulative_fidelity_curve(
                weights,
                noise_scales,
            )
        else:
            raise ValueError(
                "noise_model must be 'independent' or 'cumulative'."
            )
    return curves


def hamming_weight_series(
    qubit_counts: Sequence[int],
    normalized_axis: bool,
) -> dict[int, tuple[FloatArray, FloatArray]]:
    """Return average compensated weights grouped by Hamming class.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    normalized_axis
        Use Hamming weight divided by qubit count when true.

    Returns
    -------
    dict
        Qubit counts mapped to class coordinates and average weights.
    """
    series: dict[int, tuple[FloatArray, FloatArray]] = {}
    for qubit_count in qubit_counts:
        bitstrings, weights = mode_weights(
            qubit_count,
            "loss-compensated",
        )
        normalized_classes, class_averages = (
            average_weights_by_hamming_class(bitstrings, weights)
        )
        if normalized_axis:
            coordinates = normalized_classes
        else:
            coordinates = np.arange(qubit_count + 1, dtype=float)
        series[qubit_count] = coordinates, class_averages
    return series


def ordering_curves(
    qubit_counts: Sequence[int],
    noise_scales: FloatArray,
    sample_count: int,
    seed: int = 42,
) -> dict[int, dict[str, FloatArray]]:
    """Return numerical and analytic ordering-comparison curves.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    noise_scales
        Phase-noise standard deviations.
    sample_count
        Number of Monte Carlo realizations per point.
    seed
        Base random seed.

    Returns
    -------
    dict
        Qubit counts mapped to six labeled fidelity curves.
    """
    independent_statistics = qubit_count_noise_statistics(
        qubit_counts,
        brickwork_adjacency,
        noise_scales,
        "independent",
        "loss-compensated",
        "natural",
        sample_count,
        seed,
    )
    cumulative_natural_statistics = qubit_count_noise_statistics(
        qubit_counts,
        brickwork_adjacency,
        noise_scales,
        "cumulative",
        "loss-compensated",
        "natural",
        sample_count,
        seed,
    )
    cumulative_ordered_statistics = qubit_count_noise_statistics(
        qubit_counts,
        brickwork_adjacency,
        noise_scales,
        "cumulative",
        "loss-compensated",
        "high-hamming-first",
        sample_count,
        seed,
    )
    independent_analytic = analytic_curves_for_case(
        qubit_counts,
        noise_scales,
        "loss-compensated",
        "independent",
        "natural",
    )
    cumulative_natural_analytic = analytic_curves_for_case(
        qubit_counts,
        noise_scales,
        "loss-compensated",
        "cumulative",
        "natural",
    )
    cumulative_ordered_analytic = analytic_curves_for_case(
        qubit_counts,
        noise_scales,
        "loss-compensated",
        "cumulative",
        "high-hamming-first",
    )
    return {
        qubit_count: {
            "Independent": independent_statistics[qubit_count].means,
            "Independent analytic": independent_analytic[qubit_count],
            "Cumulative natural": (
                cumulative_natural_statistics[qubit_count].means
            ),
            "Cumulative natural analytic": (
                cumulative_natural_analytic[qubit_count]
            ),
            "Cumulative high-H first": (
                cumulative_ordered_statistics[qubit_count].means
            ),
            "Cumulative high-H analytic": (
                cumulative_ordered_analytic[qubit_count]
            ),
        }
        for qubit_count in qubit_counts
    }


def saved_path(
    figure: Figure,
    output_directory: Path,
    filename: str,
    timestamped: bool,
    figures: list[Figure] | None,
    paths: list[Path],
) -> None:
    """Save and register one generated figure.

    Parameters
    ----------
    figure
        Figure to save.
    output_directory
        Destination directory.
    filename
        Output filename stem.
    timestamped
        Append a timestamp when true.
    figures
        Optional mutable figure registry for later display.
    paths
        Mutable output-path registry.

    Returns
    -------
    None
        The registries are updated in place.
    """
    paths.append(
        saved_figure_path(
            figure,
            output_directory,
            filename,
            timestamped,
        )
    )
    if figures is None:
        plt.close(figure)
    else:
        figures.append(figure)


def research_figure_paths(
    output_directory: Path,
    profile: str,
    timestamped: bool = False,
    keep_figures: bool = False,
) -> tuple[list[Path], list[Figure], dict[str, FidelityGainSeries]]:
    """Generate all notebook-equivalent figures and return their paths.

    Parameters
    ----------
    output_directory
        Destination directory for PNG files.
    profile
        Either the fast ``"quick"`` profile or ``"thesis"`` settings.
    timestamped
        Append timestamps to filenames when true.
    keep_figures
        Retain live figures for interactive display when true.

    Returns
    -------
    tuple
        Saved paths, live figures, and fidelity-gain data.
    """
    settings = profile_settings(profile)
    paths: list[Path] = []
    figures: list[Figure] | None = [] if keep_figures else None
    topology_map = dict(topology_factories())
    selected_topologies: Mapping[str, TopologyFactory] = {
        label: topology_map[label]
        for label in ("Line", "Star", "Brickwork")
    }

    graph_qubit_count = 6
    for topology_name, topology_factory in topology_map.items():
        saved_path(
            graph_figure(
                topology_factory(graph_qubit_count),
                f"{topology_name} Graph",
            ),
            output_directory,
            f"graph_{topology_name.lower()}_n6",
            timestamped,
            figures,
            paths,
        )

    phase_noise_scales = np.linspace(0.0, 1.0, 21)
    phase_statistics = topology_noise_statistics(
        graph_qubit_count,
        topology_map,
        phase_noise_scales,
        "independent",
        "uniform",
        settings["phase_samples"],
        123,
    )
    phase_analytic = independent_curves_by_qubit_count(
        [graph_qubit_count],
        phase_noise_scales,
    )[graph_qubit_count]
    for topology_name, statistics in phase_statistics.items():
        saved_path(
            phase_noise_fidelity_figure(
                statistics,
                phase_analytic,
                topology_name,
            ),
            output_directory,
            f"phase_noise_{topology_name.lower()}_n6",
            timestamped,
            figures,
            paths,
        )

    saved_path(
        topology_comparison_figure(
            phase_statistics,
            phase_analytic,
            graph_qubit_count,
        ),
        output_directory,
        "topology_comparison_n6",
        timestamped,
        figures,
        paths,
    )

    analytic_noise_scales = np.linspace(0.0, 1.0, 100)
    analytic_qubit_counts = [3, 4, 5, 6, 7]
    analytic_curves = independent_curves_by_qubit_count(
        analytic_qubit_counts,
        analytic_noise_scales,
    )
    for topology_name in topology_map:
        saved_path(
            analytic_qubit_count_figure(
                analytic_noise_scales,
                analytic_curves,
                f"{topology_name} Graph",
            ),
            output_directory,
            f"analytic_qubit_scaling_{topology_name.lower()}",
            timestamped,
            figures,
            paths,
        )

    monte_carlo_counts = [3, 4, 5, 6]
    monte_carlo_noise_scales = np.linspace(0.0, 1.0, 21)
    monte_carlo_analytic = independent_curves_by_qubit_count(
        monte_carlo_counts,
        monte_carlo_noise_scales,
    )
    for topology_name, topology_factory in topology_map.items():
        count_statistics = qubit_count_noise_statistics(
            monte_carlo_counts,
            topology_factory,
            monte_carlo_noise_scales,
            "independent",
            "uniform",
            "natural",
            settings["phase_samples"],
            42,
        )
        saved_path(
            qubit_count_monte_carlo_figure(
                count_statistics,
                monte_carlo_analytic,
                f"{topology_name} Graph",
            ),
            output_directory,
            f"monte_carlo_qubit_scaling_{topology_name.lower()}",
            timestamped,
            figures,
            paths,
        )

    fiber_noise_scales = np.linspace(0.0, np.pi / 5.0, 31)
    for amplitude_mode, amplitude_label in (
        ("uniform", "uniform amplitudes"),
        ("loss-compensated", "loss-compensated amplitudes"),
    ):
        model_statistics: dict[str, dict[str, FidelityStatistics]] = {}
        for noise_label, noise_model in (
            ("Independent", "independent"),
            ("Cumulative", "cumulative"),
        ):
            model_statistics[noise_label] = topology_noise_statistics(
                graph_qubit_count,
                selected_topologies,
                fiber_noise_scales,
                noise_model,
                amplitude_mode,
                settings["general_samples"],
                42,
            )
        analytic_by_model = None
        if amplitude_mode == "uniform":
            uniform_weights = mode_weights(
                graph_qubit_count,
                "uniform",
            )[1]
            analytic_by_model = {
                "Independent": independent_fidelity_curve(
                    uniform_weights,
                    fiber_noise_scales,
                ),
                "Cumulative": cumulative_fidelity_curve(
                    uniform_weights,
                    fiber_noise_scales,
                ),
            }
        saved_path(
            dual_noise_topology_figure(
                model_statistics,
                analytic_by_model,
                graph_qubit_count,
                amplitude_label,
            ),
            output_directory,
            f"graphs_vs_delta_n6_{amplitude_mode}",
            timestamped,
            figures,
            paths,
        )

    scaling_counts = [3, 4, 5, 6, 7, 8]
    for amplitude_mode in ("uniform", "loss-compensated"):
        case_settings = {
            "Independent phase noise": ("independent", "natural"),
            "Cumulative as-is": ("cumulative", "natural"),
            "Cumulative high-H first": (
                "cumulative",
                "high-hamming-first",
            ),
        }
        statistics_by_case: dict[
            str,
            dict[int, FidelityStatistics],
        ] = {}
        analytic_by_case: dict[str, dict[int, FloatArray]] = {}
        for case_label, (noise_model, order_mode) in case_settings.items():
            statistics_by_case[case_label] = (
                qubit_count_noise_statistics(
                    scaling_counts,
                    brickwork_adjacency,
                    fiber_noise_scales,
                    noise_model,
                    amplitude_mode,
                    order_mode,
                    settings["general_samples"],
                    42,
                )
            )
            analytic_by_case[case_label] = analytic_curves_for_case(
                scaling_counts,
                fiber_noise_scales,
                amplitude_mode,
                noise_model,
                order_mode,
            )
        saved_path(
            three_case_scaling_figure(
                statistics_by_case,
                analytic_by_case,
                "Brickwork",
                amplitude_mode,
            ),
            output_directory,
            f"n_behavior_three_cases_brickwork_{amplitude_mode}",
            timestamped,
            figures,
            paths,
        )

    saved_path(
        hamming_weight_distribution_figure(
            hamming_weight_series([3, 4, 5, 6, 7], False),
            normalized_axis=False,
            smooth_curves=False,
        ),
        output_directory,
        "weights_vs_hamming",
        timestamped,
        figures,
        paths,
    )
    saved_path(
        hamming_weight_distribution_figure(
            hamming_weight_series([3, 4, 5, 6, 7, 8], True),
            normalized_axis=True,
            smooth_curves=True,
        ),
        output_directory,
        "loss_compensated_weight_distribution",
        timestamped,
        figures,
        paths,
    )

    dense_noise_scales = np.linspace(
        0.0,
        np.pi / 5.0,
        settings["dense_points"],
    )
    saved_path(
        ordering_comparison_figure(
            ordering_curves(
                scaling_counts,
                dense_noise_scales,
                settings["general_samples"],
            ),
            dense_noise_scales,
        ),
        output_directory,
        "ordering_comparison_loss_comp_n3_to_n8",
        timestamped,
        figures,
        paths,
    )

    gain_counts = [3, 4, 5, 6, 7]
    gain_series = fidelity_gains_by_qubit_count(
        gain_counts,
        np.pi / 10.0,
        brickwork_adjacency,
        settings["gain_samples"],
        42,
    )
    saved_path(
        fidelity_gain_figure(
            np.asarray(gain_counts, dtype=float),
            gain_series,
            r"\pi/10",
            "Brickwork",
        ),
        output_directory,
        "fidelity_gain_vs_n",
        timestamped,
        figures,
        paths,
    )

    rate_counts = np.arange(3, 9, dtype=int)
    probabilities = rrsp_success_probabilities(rate_counts)
    saved_path(
        rrsp_probability_figure(rate_counts, probabilities),
        output_directory,
        "rrsp_rate_vs_n",
        timestamped,
        figures,
        paths,
    )

    final_noise_scales = np.linspace(0.0, np.pi / 5.0, 401)
    for amplitude_mode in ("loss-compensated", "uniform"):
        final_curves = final_comparison_curves(
            [6, 12],
            final_noise_scales,
            amplitude_mode,
        )
        saved_path(
            final_phase_noise_comparison_figure(
                final_noise_scales,
                final_curves,
                amplitude_mode,
            ),
            output_directory,
            f"final_n6_n12_{amplitude_mode}",
            timestamped,
            figures,
            paths,
        )

    return paths, [] if figures is None else figures, gain_series


def printed_gain_table(
    qubit_counts: Sequence[int],
    gains_by_case: Mapping[str, FidelityGainSeries],
) -> None:
    """Print the fixed-noise fidelity gain table.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    gains_by_case
        Case labels mapped to uniform, compensated, and gain series.

    Returns
    -------
    None
        The table is written to standard output.
    """
    print("\nFidelity gain table at delta = pi/10")
    print("------------------------------------")
    for index, qubit_count in enumerate(qubit_counts):
        print(f"n = {qubit_count}")
        for label, series in gains_by_case.items():
            print(
                f"  {label}: "
                f"F_uniform={series.uniform[index]:.6f}, "
                f"F_loss={series.loss_compensated[index]:.6f}, "
                f"DeltaF={series.gain[index]:+.6f}"
            )


def printed_probability_table() -> None:
    """Print the default remote-state-preparation probability table.

    Returns
    -------
    None
        The table is written to standard output.
    """
    qubit_counts = np.arange(3, 9, dtype=int)
    probabilities = rrsp_success_probabilities(qubit_counts)
    print("\nR-RSP success probability table")
    print("--------------------------------")
    for qubit_count, probability in zip(qubit_counts, probabilities):
        print(f"n = {qubit_count}, P_B = {probability:.6f}")


def main(arguments: Sequence[str] | None = None) -> int:
    """Run validation and optionally reproduce all research figures.

    Parameters
    ----------
    arguments
        Optional command-line argument sequence.

    Returns
    -------
    int
        Zero when validation passes, otherwise one.
    """
    options = command_line_parser().parse_args(arguments)
    results = validation_results()
    printed_validation_report(results)
    if not all(result.passed for result in results):
        return 1
    if options.validation_only:
        return 0

    paths, figures, gain_series = research_figure_paths(
        options.output_dir,
        options.profile,
        options.timestamped,
        options.show,
    )
    printed_gain_table([3, 4, 5, 6, 7], gain_series)
    printed_probability_table()
    print(f"\nSaved {len(paths)} figures to {options.output_dir.resolve()}")
    for path in paths:
        print(f"  {path}")

    if options.show:
        show_figures(figures)
    else:
        for figure in figures:
            plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
