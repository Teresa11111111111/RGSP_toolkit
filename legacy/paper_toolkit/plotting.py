"""Publication-quality plotting functions for pre-evaluated research data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from scipy.interpolate import make_interp_spline

from .graph_topologies import networkx_graph
from .phase_noise import FidelityStatistics
from .rrsp_analysis import FidelityGainSeries


def saved_figure_path(
    figure: Figure,
    output_directory: str | Path,
    filename: str,
    timestamped: bool = False,
    resolution: int = 300,
) -> Path:
    """Save a figure and return its output path.

    Parameters
    ----------
    figure
        Matplotlib figure to save.
    output_directory
        Destination directory.
    filename
        Filename stem or filename ending in ``.png``.
    timestamped
        Append a wall-clock timestamp when true.
    resolution
        Raster resolution in dots per inch.

    Returns
    -------
    pathlib.Path
        Path of the saved PNG file.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    if timestamped:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{stem}_{timestamp}"
    output_path = destination / f"{stem}.png"
    figure.savefig(output_path, dpi=resolution, bbox_inches="tight")
    return output_path


def graph_figure(
    adjacency_matrix: ArrayLike,
    title: str = "Graph",
) -> Figure:
    """Return a visualization of an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Square graph adjacency matrix.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Graph visualization.
    """
    graph = networkx_graph(adjacency_matrix)
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    positions = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        positions,
        ax=axis,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
    )
    axis.set_title(title)
    return figure


def phase_noise_fidelity_figure(
    statistics: FidelityStatistics,
    analytic_values: ArrayLike,
    title: str = "",
) -> Figure:
    """Return a Monte Carlo and analytic phase-noise figure.

    Parameters
    ----------
    statistics
        Monte Carlo means and standard deviations.
    analytic_values
        Analytic fidelity values at the same noise strengths.
    title
        Graph or experiment label.

    Returns
    -------
    matplotlib.figure.Figure
        Fidelity comparison figure.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(
        statistics.noise_scales,
        statistics.means,
        yerr=statistics.standard_deviations,
        marker="o",
        capsize=3,
        label="Monte Carlo mean ± std",
    )
    axis.plot(
        statistics.noise_scales,
        np.asarray(analytic_values, dtype=float),
        linestyle="--",
        label=r"Analytic $\mathbb{E}[F]$",
    )
    axis.set_xlabel(r"Phase noise standard deviation $\delta$")
    axis.set_ylabel("Fidelity with ideal graph state")
    axis.set_title(f"Independent Gaussian phase noise: {title}")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    return figure


def analytic_qubit_count_figure(
    noise_scales: ArrayLike,
    fidelity_by_qubit_count: Mapping[int, ArrayLike],
    topology_name: str,
) -> Figure:
    """Return analytic fidelity curves for several qubit counts.

    Parameters
    ----------
    noise_scales
        Phase-noise standard deviations.
    fidelity_by_qubit_count
        Qubit counts mapped to analytic fidelity values.
    topology_name
        Topology label used in the title.

    Returns
    -------
    matplotlib.figure.Figure
        Analytic qubit-scaling figure.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(
        np.linspace(0.15, 0.85, len(fidelity_by_qubit_count))
    )
    for (qubit_count, fidelities), color in zip(
        fidelity_by_qubit_count.items(),
        colors,
    ):
        axis.plot(
            noise_scales,
            fidelities,
            label=f"n = {qubit_count}",
            color=color,
            linewidth=2,
        )
    axis.set_xlabel(r"Phase noise standard deviation $\delta$", fontsize=12)
    axis.set_ylabel(r"Analytic $\mathbb{E}[F]$", fontsize=12)
    axis.set_title(
        f"Fidelity vs qubit count: {topology_name}\n"
        r"$\mathbb{E}[F]=2^{-n}+(1-2^{-n})e^{-\delta^2}$",
        fontsize=11,
    )
    axis.legend(fontsize=10)
    axis.grid(True)
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    return figure


def qubit_count_monte_carlo_figure(
    statistics_by_qubit_count: Mapping[int, FidelityStatistics],
    analytic_by_qubit_count: Mapping[int, ArrayLike],
    topology_name: str,
) -> Figure:
    """Return Monte Carlo panels for several qubit counts.

    Parameters
    ----------
    statistics_by_qubit_count
        Qubit counts mapped to Monte Carlo statistics.
    analytic_by_qubit_count
        Qubit counts mapped to analytic fidelity curves.
    topology_name
        Topology label used in the title.

    Returns
    -------
    matplotlib.figure.Figure
        Multi-panel qubit-scaling figure.
    """
    panel_count = len(statistics_by_qubit_count)
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(4 * panel_count, 4),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    for axis, (qubit_count, statistics) in zip(
        flat_axes,
        statistics_by_qubit_count.items(),
    ):
        axis.errorbar(
            statistics.noise_scales,
            statistics.means,
            yerr=statistics.standard_deviations,
            marker="o",
            markersize=4,
            capsize=2,
            label="MC mean ± std",
        )
        axis.plot(
            statistics.noise_scales,
            analytic_by_qubit_count[qubit_count],
            linestyle="--",
            color="red",
            label=r"Analytic $\mathbb{E}[F]$",
        )
        axis.set_title(f"n = {qubit_count}", fontsize=11)
        axis.set_xlabel(r"$\delta$", fontsize=11)
        axis.grid(True)
        axis.set_ylim(0, 1.05)
    flat_axes[0].set_ylabel("Fidelity", fontsize=11)
    flat_axes[0].legend(fontsize=8)
    figure.suptitle(
        f"Fidelity vs qubit count - {topology_name}\n"
        r"$\phi_x\to\phi_x+\varepsilon_x,\ "
        r"\varepsilon_x\sim\mathcal{N}(0,\delta^2)$",
        fontsize=12,
    )
    figure.tight_layout()
    return figure


def topology_comparison_figure(
    statistics_by_topology: Mapping[str, FidelityStatistics],
    analytic_values: ArrayLike,
    qubit_count: int,
) -> Figure:
    """Return a fixed-size graph-topology fidelity comparison.

    Parameters
    ----------
    statistics_by_topology
        Topology labels mapped to Monte Carlo statistics.
    analytic_values
        Shared analytic fidelity curve.
    qubit_count
        Number of logical qubits.

    Returns
    -------
    matplotlib.figure.Figure
        Topology comparison figure.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.arange(len(statistics_by_topology)))
    for (label, statistics), color in zip(
        statistics_by_topology.items(),
        colors,
    ):
        axis.errorbar(
            statistics.noise_scales,
            statistics.means,
            yerr=statistics.standard_deviations,
            marker="o",
            markersize=4,
            capsize=2,
            label=label,
            color=color,
        )
    first_statistics = next(iter(statistics_by_topology.values()))
    axis.plot(
        first_statistics.noise_scales,
        analytic_values,
        linestyle="--",
        color="black",
        linewidth=2,
        label=r"Analytic $\mathbb{E}[F]$ (all topologies)",
    )
    axis.set_xlabel(r"Phase noise standard deviation $\delta$", fontsize=12)
    axis.set_ylabel("Fidelity with ideal graph state", fontsize=12)
    axis.set_title(
        f"Topology comparison at n = {qubit_count}\n"
        r"$\phi_x\to\phi_x+\varepsilon_x,\ "
        r"\varepsilon_x\sim\mathcal{N}(0,\delta^2)$",
        fontsize=12,
    )
    axis.legend(fontsize=10)
    axis.grid(True)
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    return figure


def dual_noise_topology_figure(
    statistics_by_noise_model: Mapping[
        str,
        Mapping[str, FidelityStatistics],
    ],
    analytic_by_noise_model: Mapping[str, ArrayLike] | None,
    qubit_count: int,
    amplitude_label: str,
) -> Figure:
    """Return side-by-side topology comparisons for two noise models.

    Parameters
    ----------
    statistics_by_noise_model
        Noise labels mapped to topology Monte Carlo statistics.
    analytic_by_noise_model
        Optional analytic curve for each noise label.
    qubit_count
        Number of logical qubits.
    amplitude_label
        Description of the temporal-mode amplitudes.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel graph comparison.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, (noise_label, topology_statistics) in zip(
        axes,
        statistics_by_noise_model.items(),
    ):
        for topology_label, statistics in topology_statistics.items():
            axis.errorbar(
                statistics.noise_scales,
                statistics.means,
                yerr=statistics.standard_deviations,
                marker="o",
                markersize=4,
                capsize=2,
                label=topology_label,
            )
        if analytic_by_noise_model is not None:
            first_statistics = next(iter(topology_statistics.values()))
            axis.plot(
                first_statistics.noise_scales,
                analytic_by_noise_model[noise_label],
                linestyle="--",
                color="black",
                linewidth=2.2,
                label="Analytic",
            )
        axis.set_title(f"{noise_label} phase noise")
        axis.set_xlabel(r"Phase noise $\delta$ [rad]")
        axis.grid(True)
        axis.set_ylim(0, 1.05)
    axes[0].set_ylabel("Output qubit graph-state fidelity")
    axes[1].legend(fontsize=9)
    figure.suptitle(
        f"Output qubit fidelity vs fiber phase noise, n={qubit_count}\n"
        f"{amplitude_label}, "
        r"$\delta\in[0,\pi/5]$",
        fontsize=14,
    )
    figure.tight_layout()
    return figure


def three_case_scaling_figure(
    statistics_by_case: Mapping[str, Mapping[int, FidelityStatistics]],
    analytic_by_case: Mapping[str, Mapping[int, ArrayLike]] | None,
    topology_name: str,
    amplitude_label: str,
) -> Figure:
    """Return three noise-and-ordering panels across qubit counts.

    Parameters
    ----------
    statistics_by_case
        Case labels mapped to qubit-count Monte Carlo statistics.
    analytic_by_case
        Optional case and qubit-count analytic curves.
    topology_name
        Topology label.
    amplitude_label
        Description of the amplitude strategy.

    Returns
    -------
    matplotlib.figure.Figure
        Three-panel scaling figure.
    """
    figure, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    analytic_colors = [
        "black",
        "dimgray",
        "darkred",
        "darkblue",
        "darkgreen",
        "darkviolet",
    ]
    for axis, (case_label, statistics_by_count) in zip(
        axes,
        statistics_by_case.items(),
    ):
        for series_index, (qubit_count, statistics) in enumerate(
            statistics_by_count.items()
        ):
            axis.plot(
                statistics.noise_scales,
                statistics.means,
                linewidth=2.4,
                label=f"n={qubit_count}",
            )
            if analytic_by_case is not None:
                axis.plot(
                    statistics.noise_scales,
                    analytic_by_case[case_label][qubit_count],
                    linestyle="--",
                    linewidth=1,
                    color=analytic_colors[
                        series_index % len(analytic_colors)
                    ],
                    alpha=0.95,
                )
        axis.set_title(case_label)
        axis.set_xlabel(r"Phase noise $\delta$ [rad]")
        axis.grid(True)
        axis.set_ylim(0, 1.05)
    axes[0].set_ylabel("Output qubit graph-state fidelity")
    axes[2].legend(fontsize=9)
    figure.suptitle(
        f"n-behavior of weighted pulse amplitudes: {topology_name}\n"
        f"amplitude mode = {amplitude_label}, "
        r"$\delta\in[0,\pi/5]$",
        fontsize=15,
    )
    figure.tight_layout()
    return figure


def hamming_weight_distribution_figure(
    series_by_qubit_count: Mapping[int, tuple[ArrayLike, ArrayLike]],
    normalized_axis: bool = False,
    smooth_curves: bool = False,
) -> Figure:
    """Return average mode weight versus Hamming class.

    Parameters
    ----------
    series_by_qubit_count
        Qubit counts mapped to x coordinates and average weights.
    normalized_axis
        Label the horizontal axis as normalized Hamming weight.
    smooth_curves
        Draw cubic spline interpolants when true.

    Returns
    -------
    matplotlib.figure.Figure
        Hamming-weight distribution figure.
    """
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for qubit_count, (coordinates, average_weights) in (
        series_by_qubit_count.items()
    ):
        coordinate_array = np.asarray(coordinates, dtype=float)
        weight_array = np.asarray(average_weights, dtype=float)
        if smooth_curves:
            smooth_coordinates = np.linspace(
                coordinate_array.min(),
                coordinate_array.max(),
                200,
            )
            spline = make_interp_spline(
                coordinate_array,
                weight_array,
                k=min(3, len(coordinate_array) - 1),
            )
            axis.plot(
                smooth_coordinates,
                spline(smooth_coordinates),
                linewidth=2.5,
                label=f"n={qubit_count}",
            )
        else:
            axis.plot(
                coordinate_array,
                weight_array,
                marker="o",
                linewidth=1.5,
                label=f"n={qubit_count}",
            )
    if normalized_axis:
        axis.set_xlabel(r"Normalized Hamming weight $H(x)/n$", fontsize=12)
        axis.set_title("Loss-compensated mode weights", fontsize=14)
    else:
        axis.set_xlabel("Hamming weight H(x)")
        axis.set_title(
            "Loss-compensated weights vs Hamming weight\n"
            r"$|c_x|^2\propto1/\eta_x$"
        )
    axis.set_ylabel(r"Average mode weight $w_x=|c_x|^2$", fontsize=12)
    axis.grid(True)
    axis.legend(fontsize=9, frameon=True)
    figure.tight_layout()
    return figure


def ordering_comparison_figure(
    curves_by_qubit_count: Mapping[int, Mapping[str, ArrayLike]],
    noise_scales: ArrayLike,
    topology_name: str = "Brickwork",
) -> Figure:
    """Return per-size panels comparing noise and ordering strategies.

    Parameters
    ----------
    curves_by_qubit_count
        Qubit counts mapped to labeled fidelity curves.
    noise_scales
        Phase-noise standard deviations shared by all curves.
    topology_name
        Topology label.

    Returns
    -------
    matplotlib.figure.Figure
        Multi-panel ordering comparison.
    """
    column_count = 3
    row_count = int(np.ceil(len(curves_by_qubit_count) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.3 * column_count, 4.2 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    for panel_index, (qubit_count, labeled_curves) in enumerate(
        curves_by_qubit_count.items()
    ):
        axis = flat_axes[panel_index]
        for label, fidelities in labeled_curves.items():
            linestyle = "--" if "analytic" in label.lower() else "-"
            linewidth = 1.0 if linestyle == "--" else 2.0
            axis.plot(
                noise_scales,
                fidelities,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label if panel_index == 0 else None,
            )
        axis.set_title(f"{topology_name}, n={qubit_count}")
        axis.set_xlabel(r"$\delta$ [rad]")
        axis.set_ylabel("Fidelity")
        axis.grid(True)
        axis.set_ylim(0, 1.05)
    for axis in flat_axes[len(curves_by_qubit_count) :]:
        axis.axis("off")
    flat_axes[0].legend(fontsize=8)
    figure.suptitle(
        "Ordering comparison under loss-compensated amplitudes\n"
        "independent vs cumulative natural vs cumulative high-H first",
        fontsize=15,
    )
    figure.tight_layout()
    return figure


def fidelity_gain_figure(
    qubit_counts: ArrayLike,
    gains_by_case: Mapping[str, FidelityGainSeries],
    noise_scale_label: str = r"$\pi/10$",
    topology_name: str = "Brickwork",
) -> Figure:
    """Return fidelity gain from loss-compensated amplitudes.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    gains_by_case
        Case labels mapped to uniform, compensated, and gain series.
    noise_scale_label
        Display label for the fixed noise strength.
    topology_name
        Topology label.

    Returns
    -------
    matplotlib.figure.Figure
        Fidelity-gain figure.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, series in gains_by_case.items():
        axis.plot(
            qubit_counts,
            series.gain,
            marker="o",
            linewidth=1.8,
            label=label,
        )
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Number of qubits n")
    axis.set_ylabel(
        r"$\Delta F=F_{\rm loss-comp}-F_{\rm uniform}$"
    )
    axis.set_title(
        f"Fidelity gain from loss compensation at "
        f"$\\delta={noise_scale_label}$\n{topology_name}"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    return figure


def rrsp_probability_figure(
    qubit_counts: ArrayLike,
    probabilities: ArrayLike,
) -> Figure:
    """Return remote-state-preparation success probability versus size.

    Parameters
    ----------
    qubit_counts
        Logical-qubit counts.
    probabilities
        Success probability for each qubit count.

    Returns
    -------
    matplotlib.figure.Figure
        Success-probability scaling figure.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(qubit_counts, probabilities, marker="o", linewidth=2)
    axis.set_xlabel("Number of qubits n")
    axis.set_ylabel(r"R-RSP success probability $P_B$")
    axis.set_title(
        "R-RSP success probability\n"
        r"$P_B=\eta_t\eta_d"
        r"\left(\frac{2\eta_0\eta_1}{\eta_0+\eta_1}\right)^n$"
    )
    axis.grid(True)
    figure.tight_layout()
    return figure


def final_phase_noise_comparison_figure(
    noise_scales: ArrayLike,
    curves_by_qubit_count: Mapping[int, Mapping[str, ArrayLike]],
    amplitude_mode: str,
    zero_efficiency: float = 0.95,
    one_efficiency: float = 0.3,
) -> Figure:
    """Return the final two-panel analytic phase-noise comparison.

    Parameters
    ----------
    noise_scales
        Phase-noise standard deviations.
    curves_by_qubit_count
        Qubit counts mapped to labeled analytic curves.
    amplitude_mode
        Either uniform or loss-compensated.
    zero_efficiency
        Internal logical-zero efficiency shown in the caption.
    one_efficiency
        Internal logical-one efficiency shown in the caption.

    Returns
    -------
    matplotlib.figure.Figure
        Final publication-style comparison figure.
    """
    figure, axes = plt.subplots(
        1,
        len(curves_by_qubit_count),
        figsize=(6 * len(curves_by_qubit_count), 5),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    for axis, (qubit_count, labeled_curves) in zip(
        flat_axes,
        curves_by_qubit_count.items(),
    ):
        for label, fidelities in labeled_curves.items():
            axis.plot(noise_scales, fidelities, linewidth=2.5, label=label)
        axis.set_title(f"n={qubit_count} qubits", fontsize=14)
        axis.set_xlabel(r"Phase noise $\delta$ [rad]")
        axis.grid(True)
        axis.set_ylim(0, 1.05)
    flat_axes[0].set_ylabel("Output qubit graph-state fidelity")
    flat_axes[-1].legend(fontsize=9)

    if amplitude_mode == "loss-compensated":
        caption = (
            "Phase-rotated graph states generated with loss-compensated "
            f"temporal-mode amplitudes, eta_0={zero_efficiency} and "
            f"eta_1={one_efficiency}. The fidelity is independent of graph "
            "connectivity under this phase-noise model."
        )
    else:
        caption = (
            "Phase-rotated graph states generated with uniform temporal-mode "
            "amplitudes. Highest-weight-first ordering is equivalent to the "
            "natural order because every mode has equal weight."
        )
    figure.subplots_adjust(bottom=0.22)
    figure.text(0.5, 0.04, caption, ha="center", fontsize=9, wrap=True)
    return figure


def show_figures(figures: Iterable[Figure]) -> None:
    """Display a collection of prepared Matplotlib figures.

    Parameters
    ----------
    figures
        Figures that should remain alive until the display call.

    Returns
    -------
    None
        Figures are displayed using the active Matplotlib backend.
    """
    tuple(figures)
    plt.show()
