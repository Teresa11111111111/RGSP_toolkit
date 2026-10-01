"""Run after installing: python examples/python_api.py. Masks here are demonstrations."""
import numpy as np
from rgsp_toolkit import Client, prepare, hamming_efficiencies, measurement_instructions

plan = prepare(
    3, source_graph_edges=[(0, 1), (1, 2)],
    clients=[
        Client(local_angles_rad=[0, np.pi / 4, 0], hiding_angles_rad=[np.pi / 4, 0, np.pi / 2]),
        Client(graph_toggle_edges=[(0, 1), (0, 2)], hiding_angles_rad=[0, np.pi / 4, 0]),
    ],
    intensity_efficiencies=hamming_efficiencies(3, eta0=0.97, eta1=0.85, common=0.8),
)
print("Input amplitudes:", plan.input_amplitudes)
print("Source phases (radians):", plan.source_phases_rad)
print("Client 2 phase-only update:", plan.client_updates_rad[1])
print("Private encryption vector:", plan.encryption_angles_rad)
print("Output graph edges:", plan.output_edges)
print("All-herald probability:", plan.survival_probability)
print("Conditioned fidelity:", abs(np.vdot(plan.target_state(), plan.output_state())) ** 2)
print("Server message:", measurement_instructions(
    [0, 0, 0], plan.preparation_angles_rad, herald_bits=[0, 1, 0], outcome_mask_bits=[1, 0, 1]))
