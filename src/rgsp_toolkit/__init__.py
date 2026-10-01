"""Remote graph-state preparation under calibrated diagonal optical loss."""
from .core import (Client, Preparation, bitstrings, client_phase_update, compensation,
                   decode_outcomes, graph_phases, graph_state, hamming_efficiencies,
                   measurement_instructions, prepare, random_hiding, xor_edges)

__version__ = "0.1.0"
__all__ = ["Client", "Preparation", "bitstrings", "client_phase_update", "compensation",
           "decode_outcomes", "graph_phases", "graph_state", "hamming_efficiencies",
           "measurement_instructions", "prepare", "random_hiding", "xor_edges"]
