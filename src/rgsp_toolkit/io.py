"""Strict JSON configuration and explicit private/source/client export boundaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .core import (Client, TAU, bitstrings, efficiencies, hamming_efficiencies,
                   measurement_instructions, prepare, validate_qubits, vector, bits_vector)


def keys(obj, allowed, required=(), context="configuration"):
    if not isinstance(obj, dict):
        raise ValueError(f"{context} must be an object")
    extra = set(obj) - set(allowed)
    missing = set(required) - set(obj)
    if extra or missing:
        raise ValueError(f"{context}: unknown keys {sorted(extra)}; missing keys {sorted(missing)}")


def parse_client(obj):
    keys(obj, {"graph_toggle_edges", "local_angles_rad", "hiding_angles_rad", "z_bits"},
         context="client (phase-only)")
    return Client(**obj)


def channel_arrays(n, segments):
    if not isinstance(segments, list) or not segments:
        raise ValueError("channels must be a nonempty list (omit it for ideal transmission)")
    eta = np.ones(1 << n)
    chi = np.zeros(1 << n)
    for segment in segments:
        keys(segment, {"model", "intensity_efficiencies", "eta0", "eta1", "common", "phases_rad"},
             {"model"}, "channel")
        if segment["model"] == "measured":
            keys(segment, {"model", "intensity_efficiencies", "phases_rad"},
                 {"intensity_efficiencies"}, "measured channel")
            e = efficiencies(segment["intensity_efficiencies"], 1 << n)
        elif segment["model"] == "hamming":
            keys(segment, {"model", "eta0", "eta1", "common", "phases_rad"}, context="hamming channel")
            e = hamming_efficiencies(n, segment.get("eta0", 1), segment.get("eta1", 1),
                                     segment.get("common", 1))
        else:
            raise ValueError("channel model must be 'measured' or 'hamming'")
        eta *= e
        chi = np.remainder(chi + np.remainder(
            vector(segment.get("phases_rad"), 1 << n, "channel phases_rad", 0), TAU), TAU)
    return efficiencies(eta, 1 << n), chi


def compile_config(config):
    keys(config, {"qubits", "source_graph_edges", "clients", "channels", "herald_bits", "measurement"},
         {"qubits"})
    n = validate_qubits(config["qubits"])
    records = config.get("clients", [{}])
    if not isinstance(records, list) or not records:
        raise ValueError("clients must be a nonempty list")
    clients = [parse_client(c) for c in records]
    eta, chi = channel_arrays(n, config.get("channels", [{"model": "hamming"}]))
    plan = prepare(n, config.get("source_graph_edges", ()), clients, eta, chi)
    # Validate all optional inputs before writing anything.
    if "herald_bits" in config and config["herald_bits"] is None:
        raise ValueError("explicit herald_bits cannot be null")
    m = bits_vector(config.get("herald_bits"), n, "herald_bits")
    if "measurement" in config:
        if "herald_bits" not in config:
            raise ValueError("measurement requires explicit observed herald_bits")
        measurement = config["measurement"]
        keys(measurement, {"adapted_angles_rad", "outcome_mask_bits"},
             {"adapted_angles_rad", "outcome_mask_bits"}, "measurement")
        if measurement["outcome_mask_bits"] is None:
            raise ValueError("measurement outcome_mask_bits cannot be null")
        measurement_instructions(measurement["adapted_angles_rad"], plan.preparation_angles_rad,
                                 m, measurement["outcome_mask_bits"])
    return plan


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_output_dir(path):
    """Never overwrite a previous experimental run; restrict directory on POSIX."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def write_modes(path, n, phases, amplitudes=None, eta=None):
    header = ["mode_index", "bitstring_q0_first", "phase_rad", "phase_over_pi"]
    if amplitudes is not None:
        header += ["input_amplitude", "input_power_fraction", "coefficient_real", "coefficient_imag",
                   "path_intensity_efficiency"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k, bits in enumerate(bitstrings(n)):
            row = [k, "".join(map(str, bits)), float(phases[k]), float(phases[k] / np.pi)]
            if amplitudes is not None:
                c = amplitudes[k] * np.exp(1j * phases[k])
                row += [float(amplitudes[k]), float(amplitudes[k] ** 2), float(c.real),
                        float(c.imag), float(eta[k])]
            writer.writerow(row)


def export_plan(plan, config, out):
    """Export trusted integration plan, with server message separate from secrets."""
    out = new_output_dir(out)
    write_modes(out / "source_modes.csv", plan.qubits, plan.source_phases_rad,
                plan.input_amplitudes, plan.intensity_efficiencies)
    for k, phase in enumerate(plan.client_updates_rad[1:], 1):
        write_modes(out / f"client_{k + 1:02d}_phases.csv", plan.qubits, phase)
    m = config.get("herald_bits")
    private = plan.correction_record(m)
    private["herald_outcome_status"] = "observed" if m is not None else "assumed_all_zero_for_preview"
    if "measurement" in config:
        measurement = config["measurement"]
        private["outcome_mask_bits"] = measurement["outcome_mask_bits"]
        write_json(out / "server_measurement_instructions.json", measurement_instructions(
            measurement["adapted_angles_rad"], plan.preparation_angles_rad, m,
            measurement["outcome_mask_bits"]))
    write_json(out / "private_corrections.json", private)
    overlap = np.vdot(plan.target_state(), plan.output_state())
    write_json(out / "summary.json", {
        "schema_version": 1, "tool_version": "0.1.0", "qubits": plan.qubits,
        "clients": len(plan.client_updates_rad), "modes": 1 << plan.qubits,
        "output_graph_edges": plan.output_edges,
        "all_herald_outcomes_probability": plan.survival_probability,
        "one_specified_herald_outcome_probability": plan.survival_probability / (1 << plan.qubits),
        "predicted_conditioned_fidelity": float(abs(overlap) ** 2),
        "input_power_sum": float(np.sum(plan.input_amplitudes ** 2)),
        "input_amplitude_dynamic_range": float(np.max(plan.input_amplitudes) / np.min(plan.input_amplitudes)),
        "model": "calibrated diagonal intensity loss and phase; ideal memory mapping and H^n erasure",
        "privacy": "This whole output directory is a trusted local artifact, not a server payload.",
    })
    write_json(out / "private_input_config.json", config)
    return out
