# RGSP Toolkit

A small experimental calculator for **loss-compensated remote preparation of
phase-rotated graph states and equatorial product states**. It turns a graph,
local angles and calibrated channel transmissions into normalized source
coefficients, per-mode phase settings, a private encryption-angle vector and
optional classical measurement instructions.

Multiple clients are supported: **only the source client sets amplitudes**;
subsequent clients program diagonal phases. Their graph-edge toggles compose by
XOR and their local rotations add modulo 2π. Each phase-only client can generate
its own table without knowing the other clients' settings.

This is a numerical model and programming-table exporter, not a hardware driver
or a complete blind-computation/security protocol. Version 0.1.0 implements the
conventions described in [the model notes](docs/model.md).

## Install and run

Python 3.10 or later and NumPy are the only runtime requirements. From a terminal:

```sh
git clone https://github.com/Teresa11111111111/RGSP_toolkit.git
cd RGSP_toolkit
python -m venv .venv
# macOS/Linux; Windows PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install .
rgsp compile examples/lossy_line.json --out runs/lossy_line
rgsp compile examples/product_state.json --out runs/product
rgsp compile examples/three_clients.json --out runs/multi
```

`python -m rgsp_toolkit` is equivalent to `rgsp`. Output directories must be new;
previous experimental runs are never overwritten. This project is installable
from GitHub or the supplied wheel; it has **not** been published to PyPI.

## A minimal configuration

```json
{
  "qubits": 2,
  "source_graph_edges": [[0, 1]],
  "clients": [{"local_angles_rad": [0, 1.5707963267948966]}],
  "channels": [{"model": "measured", "intensity_efficiencies": [0.9, 0.7, 0.6, 0.4]}]
}
```

Save as `my_experiment.json` and run
`rgsp compile my_experiment.json --out runs/my_experiment`.
Use `"source_graph_edges": []` for a product of equatorial states.
All angles are **radians**, all efficiencies are **intensity probabilities**,
qubits are **zero-based**, and mode labels use **qubit 0 first / most significant**.
For two qubits the modes are `00, 01, 10, 11`. Keep this order on the hardware.

| Output | Meaning / intended recipient |
| --- | --- |
| `source_modes.csv` | Source client: normalized field amplitudes, power fractions, complex coefficients and source phases |
| `client_02_phases.csv`, … | Each later client: its own phase table only; no amplitude control |
| `private_corrections.json` | Trusted controller: aggregate hiding vector, logical rotations, herald bits and inverse corrections |
| `server_measurement_instructions.json` | Optional server payload: corrected measurement angles only |
| `summary.json` | Predicted success probability, conditioned fidelity and amplitude dynamic range; contains graph information, keep local |
| `private_input_config.json` | Exact input for reproducing this run, including private angles |

Do not send the entire output directory to an untrusted server. The source table
already includes client 1's phase update and pre-cancels calibrated channel phase;
do **not** apply that client's update a second time. CSV bitstrings should be
imported as text in spreadsheets to preserve leading zeros.

## Multiple clients and local hiding

See [three_clients.json](examples/three_clients.json). `source_graph_edges` is the
initial graph. Every client's `graph_toggle_edges` is a **change**, not the final
graph: applying the same edge twice cancels it. To choose a final graph, use the
symmetric difference of its edge set and the initial graph as the total toggle.
Local logical rotations, hiding angles and optional `z_bits` are separate inputs.

A client can generate its update using only its own configuration:

```sh
rgsp client examples/local_client.json --out runs/client_2
rgsp random-mask --qubits 3 --out runs/fresh_mask
```

The mask command uses operating-system randomness. Copy the generated fields
into that client's private config. The examples use **fixed demonstration masks**;
they are not secret and must not be reused for a private experiment. Local
mask generation is not itself a proof of blindness or collusion resistance.
The central `compile` command sees all supplied masks; use it only in a trusted
integration context. `client` permits local table generation without aggregation.

## Classical corrections

For bitwise Hadamard erasure outcome `m`, the memory has a local `Z^m` byproduct.
The private vector `encryption_angles_rad` contains only the sum of clients'
hiding angles and π times their `z_bits`; `preparation_angles_rad` also includes
intentional logical rotations.

The optional measurement block supplies **already-adapted** angles `phi_prime`
relative to the **unrotated final graph** and fresh outcome-mask bits `r`:

```json
"herald_bits": [0, 1],
"measurement": {
  "adapted_angles_rad": [0, 0.7853981633974483],
  "outcome_mask_bits": [1, 0]
}
```

The server receives `delta = phi_prime + preparation_angle + pi*m + pi*r (mod 2*pi)`.
Decode its outcomes with `decode_outcomes(server_bits, r)`. The software does not
infer a computation's flow, measurement order or adaptive dependencies. For an
adaptive computation, call the Python helper as each adapted angle becomes
known; the batch example is not a general MBQC compiler. See
[the conventions and derivation](docs/model.md#measurement-conventions).

## Python API

```python
from rgsp_toolkit import Client, prepare, hamming_efficiencies

plan = prepare(
    3,
    source_graph_edges=[(0, 1), (1, 2)],
    clients=[Client(), Client(graph_toggle_edges=[(0, 1), (0, 2)])],
    intensity_efficiencies=hamming_efficiencies(3, eta0=0.96, eta1=0.8, common=0.7),
)
print(plan.input_amplitudes)
print(plan.source_phases_rad)
print(plan.client_updates_rad[1])
print(plan.encryption_angles_rad)  # private
```

Run `python examples/python_api.py` for a complete example.
See [configuration reference](docs/configuration.md) and [validation](docs/validation.md).

## Model boundaries

- The target family is `prod_i diag(1, exp(i*theta_i)) |G>`: simple undirected graph
  states, including equatorial product states. Arbitrary Bloch polar amplitudes
  and non-diagonal/noisy channels are not implemented.
- The channel is calibrated diagonal attenuation and deterministic phase.
  Stochastic phase noise, dark counts, memory errors and drift require a separate model.
- Compensation restores the **conditioned** state; it does not undo loss events.
  Total success probability and a selected detector outcome probability are separate.
- The source must know the whole path's intensity calibration. Unknown future loss
  cannot be compensated by phase-only clients. Inputs with any zero transmission
  are rejected rather than silently dropping a mode.
- Phase and amplitude settings are ideal normalized targets. Convert them to
  modulator voltages with your own calibration and check the exported dynamic range.
- Explicit enumeration is limited to 1–16 qubits (`2**n` modes), independent of the
  number of clients. Larger problems need another representation.

## Development and provenance

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip install build
python -m build
```

The implementation follows *High-Fidelity Remote Graph State Preparation for
Blind Quantum Computation*, Jiawei Cai, Rex Fleur, Benedikt Tissot, Wolfgang
Löffler and Tzula B. Propp, author manuscript (V3): Appendices A, B, D, E and H.
The manuscript PDF is not redistributed. Measurement masking conventions also
follow [Broadbent, Fitzsimons and Kashefi, arXiv:0807.4154](https://arxiv.org/abs/0807.4154).

The previous paper-plotting implementation is preserved in
[`legacy/paper_toolkit/`](legacy/paper_toolkit/), excluded from the installed package.
Older root-level plotting modules are retained for provenance and are not part
of the new installation. Original build products remain in Git history. The original local
paper and figure workspace is unchanged.

MIT licensed; see [LICENSE](LICENSE).

## 中文快速说明

输入目标图（从 0 编号）、局域旋转、各光学模式的强度透射率；输出归一化振幅、
相位设置和私有加密角度向量。只有第一个客户端调振幅，其他客户端仅调相位，
边的增删按 XOR 合成。空图对应赤道面乘积态。`runs/` 内含私有信息，不能整目录发给服务器。
所有示例已可直接运行，详见以上命令；示例中的固定隐藏角度仅供演示。
