"""Adjacency matrices and graph-conversion utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TopologyFactory = Callable[[int], FloatArray]


def line_adjacency(qubit_count: int) -> FloatArray:
    """Return the adjacency matrix of a line graph.

    Parameters
    ----------
    qubit_count
        Number of graph vertices.

    Returns
    -------
    numpy.ndarray
        Symmetric line-graph adjacency matrix.
    """
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least one.")
    adjacency = np.zeros((qubit_count, qubit_count), dtype=float)
    for i in range(qubit_count - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    return adjacency


def star_adjacency(qubit_count: int) -> FloatArray:
    """Return the adjacency matrix of a star graph.

    Parameters
    ----------
    qubit_count
        Number of graph vertices.

    Returns
    -------
    numpy.ndarray
        Symmetric star-graph adjacency matrix.
    """
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least one.")
    adjacency = np.zeros((qubit_count, qubit_count), dtype=float)
    for i in range(1, qubit_count):
        adjacency[0, i] = 1.0
        adjacency[i, 0] = 1.0
    return adjacency


def clique_adjacency(qubit_count: int) -> FloatArray:
    """Return the adjacency matrix of a complete graph.

    Parameters
    ----------
    qubit_count
        Number of graph vertices.

    Returns
    -------
    numpy.ndarray
        Symmetric complete-graph adjacency matrix without self-loops.
    """
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least one.")
    return np.ones((qubit_count, qubit_count), dtype=float) - np.eye(
        qubit_count,
        dtype=float,
    )


def brickwork_adjacency(qubit_count: int) -> FloatArray:
    """Return the brickwork adjacency used in the research notebooks.

    Parameters
    ----------
    qubit_count
        Number of graph vertices.

    Returns
    -------
    numpy.ndarray
        Symmetric brickwork-style adjacency matrix.
    """
    adjacency = line_adjacency(qubit_count)
    for i in range(0, qubit_count - 2, 2):
        adjacency[i, i + 2] = 1.0
        adjacency[i + 2, i] = 1.0
    return adjacency


def topology_factories() -> Mapping[str, TopologyFactory]:
    """Return the supported graph names and adjacency factories.

    Returns
    -------
    mapping
        Human-readable topology names mapped to adjacency factories.
    """
    return {
        "Line": line_adjacency,
        "Star": star_adjacency,
        "Clique": clique_adjacency,
        "Brickwork": brickwork_adjacency,
    }


def networkx_graph(adjacency_matrix: ArrayLike) -> nx.Graph:
    """Return a NetworkX graph from an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix
        Square graph adjacency matrix.

    Returns
    -------
    networkx.Graph
        Undirected graph with integer node labels.
    """
    adjacency = np.asarray(adjacency_matrix, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_matrix must be square.")
    return nx.from_numpy_array(adjacency)
