# Remote Graph State Preparation(RGSP) Toolkit

Research software for GRSP.
The implementation covers:

- graph-state phase encoding and controlled-Z reference states;
- line, star, clique, and brickwork graph topologies;
- independent and cumulative Gaussian phase noise;
- Monte Carlo and analytic fidelity predictions;
- Hamming-weight-dependent loss and inverse-loss compensation;
- temporal-mode ordering strategies;
- GRSP success-probability scaling;
- numerical consistency checks; and
- publication-quality figure reproduction.

## Scientific motivation

An encrypted graph state can be written in the computational basis as

\[
|\Psi_G\rangle =
\frac{1}{\sqrt{2^n}}
\sum_{x\in\{0,1\}^n}
e^{i\phi_x}|x\rangle,
\qquad
\phi_x =
\sum_i \theta_i x_i +
\pi\sum_{i<j}A_{ij}x_ix_j.
\]

This representation moves graph connectivity into a structured phase pattern.
For temporal-mode weights \(w_x=|c_x|^2\), phase-noise fidelity becomes

\[
F = \left|\sum_x w_x e^{i\delta\phi_x}\right|^2.
\]

The ideal graph phases cancel, so the modeled transmission-noise fidelity is
independent of graph topology. The package verifies this identity numerically
and compares independent residual phase errors with cumulative random-walk
drift.

## Project structure

```text
graph_rrsp_toolkit/
├── __init__.py
├── graph_states.py
├── graph_topologies.py
├── phase_encoding.py
├── phase_noise.py
├── amplitude_compensation.py
├── loss_models.py
├── rrsp_analysis.py
├── plotting.py
├── validation.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <repository-url>
cd graph_rrsp_toolkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For an editable development installation, replace the final command with
`python -m pip install -e .`. Installing from `requirements.txt` is also
supported.

## Validation

The validation suite checks the phase construction against an explicit
controlled-Z state, verifies the loss-success formula, confirms topology
independence, and compares seeded results with notebook reference values.

```bash
graph-rrsp-toolkit --validation-only
```

## Usage examples

### Construct and verify a graph state

```python
import numpy as np

from graph_rrsp_toolkit.graph_states import (
    controlled_z_graph_state,
    graph_phase_state,
    pure_state_fidelity,
)
from graph_rrsp_toolkit.graph_topologies import brickwork_adjacency

qubit_count = 6
local_angles = np.random.default_rng(42).uniform(
    0.0,
    2.0 * np.pi,
    qubit_count,
)
adjacency = brickwork_adjacency(qubit_count)

phase_state = graph_phase_state(local_angles, adjacency)
reference_state = controlled_z_graph_state(local_angles, adjacency)

print(pure_state_fidelity(phase_state, reference_state))
```

### Compare Monte Carlo data with the independent-noise formula

```python
import numpy as np

from graph_rrsp_toolkit.graph_topologies import line_adjacency
from graph_rrsp_toolkit.phase_encoding import phase_values
from graph_rrsp_toolkit.phase_noise import (
    independent_average_fidelity,
    phase_noise_fidelity_statistics,
)

qubit_count = 6
local_angles = np.random.default_rng(42).uniform(
    0.0,
    2.0 * np.pi,
    qubit_count,
)
_, phases = phase_values(local_angles, line_adjacency(qubit_count))
noise_scales = np.linspace(0.0, 1.0, 21)

statistics = phase_noise_fidelity_statistics(
    phases,
    noise_scales,
    noise_model="independent",
    sample_count=500,
    seed=123,
)
analytic = np.array(
    [
        independent_average_fidelity(qubit_count, noise_scale)
        for noise_scale in noise_scales
    ]
)
```

### Evaluate loss compensation and GRSP probability

```python
from graph_rrsp_toolkit.amplitude_compensation import (
    loss_compensated_amplitudes,
)
from graph_rrsp_toolkit.loss_models import (
    efficiencies_by_bitstring,
)
from graph_rrsp_toolkit.rrsp_analysis import rrsp_success_probability

efficiencies = efficiencies_by_bitstring(
    qubit_count=6,
    transmission_efficiency=0.8,
    detector_efficiency=0.9,
    one_efficiency=0.3,
    zero_efficiency=0.95,
)
amplitudes = loss_compensated_amplitudes(list(efficiencies.values()))
probability = rrsp_success_probability(
    qubit_count=6,
    transmission_efficiency=0.8,
    detector_efficiency=0.9,
    one_efficiency=0.3,
    zero_efficiency=0.95,
)
```

## Reproducing the thesis figures

A quick profile exercises every figure pipeline with fewer Monte Carlo
realizations:

```bash
graph-rrsp-toolkit \
    --profile quick \
    --output-dir results/quick
```

The thesis profile uses the original notebook sampling settings, including
500- or 1000-sample sweeps, 2000 samples for the fidelity-gain table, and dense
ordering curves:

```bash
graph-rrsp-toolkit \
    --profile thesis \
    --output-dir results/thesis
```

Add `--show` to display the figures after saving. Add `--timestamped` to retain
multiple runs without overwriting files.

The thesis profile reproduces:

- one graph visualization for each supported topology;
- independent phase-noise curves and analytic predictions;
- topology comparisons at fixed graph size;
- analytic and Monte Carlo qubit-count sweeps;
- independent/cumulative graph comparisons with uniform and compensated
  amplitudes;
- weighted-amplitude scaling for natural and high-Hamming-first ordering;
- Hamming-class weight distributions;
- ordering comparisons for \(n=3,\ldots,8\);
- fidelity gain from loss compensation; and
- final \(n=6\) and \(n=12\) phase-noise comparisons.

## Module descriptions

| Module | Responsibility |
|---|---|
| `graph_states.py` | State vectors, tensor products, normalization, graph-state construction, and pure-state fidelity |
| `graph_topologies.py` | Line, star, clique, and brickwork adjacency matrices plus NetworkX conversion |
| `phase_encoding.py` | Computational-basis bitstrings, \(\phi_x\), ordered phase arrays, and phase decomposition |
| `phase_noise.py` | Independent/cumulative noise, Monte Carlo statistics, analytic fidelity, and weight autocorrelation |
| `amplitude_compensation.py` | Inverse-loss amplitudes, uniform amplitudes, Hamming ordering, and class averages |
| `loss_models.py` | Mode efficiencies and Hamming-weight-dependent loss |
| `rrsp_analysis.py` | Success probabilities, scaling sweeps, fidelity gains, and final comparison data |
| `plotting.py` | Plot construction and explicit figure saving; no physics simulations |
| `validation.py` | Numerical identities and regression checks against notebook outputs |
| `main.py` | Command-line orchestration and full figure reproduction |

## Adjustable parameters

| Parameter | Typical value | Meaning |
|---|---:|---|
| `qubit_count` | 3-12 | Number of logical qubits; temporal modes scale as \(2^n\) |
| `noise_scale` | \(0\) to \(\pi/5\) rad | Standard deviation of phase errors or random-walk increments |
| `noise_model` | `independent`, `cumulative` | Residual mode noise or accumulated phase drift |
| `sample_count` | 500-2000 | Monte Carlo realizations per point |
| `seed` | 42 or 123 | Reproducible random seed |
| `transmission_efficiency` | 0.8 | Shared optical transmission efficiency \(\eta_t\) |
| `detector_efficiency` | 0.9 | Detector efficiency \(\eta_d\) |
| `one_efficiency` | 0.3 or 0.7 | Internal logical-one efficiency \(\eta_1\) |
| `zero_efficiency` | 0.95 | Internal logical-zero efficiency \(\eta_0\) |
| `amplitude_mode` | `uniform`, `loss-compensated` | Temporal-mode amplitude strategy |
| `order_mode` | `natural`, `high-hamming-first` | Temporal-mode ordering strategy |

The default loss-analysis figures use `one_efficiency=0.3`; the phase notebook's
loss-consistency example uses `one_efficiency=0.7`.

## Renamed function mapping

| Notebook function | Package replacement |
|---|---|
| `plus_theta` | `equatorial_plus_state` |
| `tensor_product` | `tensor_product_state` |
| `normalize_state` | `normalized_state` |
| `fidelity` | `pure_state_fidelity` |
| `compute_phi_single` | `bitstring_phase` |
| `compute_all_phi` | `phases_by_bitstring` |
| `compute_all_phi_list` | `phase_values` |
| `compute_phi_decomposition` | `phase_decomposition_by_bitstring` |
| `build_phi_state` | `graph_phase_state` |
| `build_target_state` | `controlled_z_graph_state` |
| `build_target_state_for_check` | `controlled_z_graph_state` |
| `build_ideal_pulse` | `pulse_state` |
| `map_pulse_to_qubit_state` | `normalized_state` |
| `build_state_from_phi_dict` | `state_from_phase_mapping` |
| `build_lossy_state` | `weighted_state_from_phase_mapping` |
| `add_independent_phi_noise` | `phase_error_values` and `noisy_pulse_state` |
| `apply_fiber_phase_noise_to_pulse` | `noisy_pulse_state` |
| `noisy_phi_fidelity_single_sample` | `phase_noise_fidelity_sample` |
| `qubit_fidelity_from_fiber_noise` | `phase_noise_fidelity_sample` |
| `monte_carlo_phi_noise_fidelity` | `phase_noise_fidelity_statistics` |
| `monte_carlo_fidelity_vs_delta` | `phase_noise_fidelity_statistics` |
| `analytic_average_fidelity_phi_noise` | `independent_average_fidelity` |
| `analytic_independent_average_fidelity` | `independent_average_fidelity` |
| `analytic_cumulative_average_fidelity` | `cumulative_average_fidelity` |
| `analytic_independent_weighted_fidelity` | `independent_weighted_fidelity` |
| `analytic_cumulative_weighted_fidelity` | `cumulative_weighted_fidelity` |
| `iid_fidelity_curve_from_delta` | `independent_fidelity_curve` |
| `cumulative_fidelity_curve_from_delta` | `cumulative_fidelity_curve` |
| `autocorrelation_by_lag` | `weight_autocorrelation` |
| `line_graph` | `line_adjacency` |
| `star_graph` | `star_adjacency` |
| `clique_graph` | `clique_adjacency` |
| `brickwork_graph` | `brickwork_adjacency` |
| `visualize_graph` | `graph_figure` |
| `generate_eta_dict` | `efficiencies_by_bitstring` |
| `generate_eta_list` | `efficiency_values` |
| `compute_coefficients` | `compensation_coefficients` |
| `loss_compensated_amplitudes` | `loss_compensated_amplitudes` |
| `uniform_amplitudes` | `uniform_amplitudes` |
| `reorder_by_high_hamming_first` | `high_hamming_first_order` |
| `no_reorder` | `unchanged_mode_order` |
| `reorder_weights_highest_first` | `descending_weights` |
| `get_loss_compensated_weights_for_n` | `mode_weights(..., "loss-compensated")` |
| `get_uniform_weights_for_n` | `mode_weights(..., "uniform")` |
| `compute_mean_fidelity_for_setting` | `fidelity_for_setting` |
| `rrsp_success_probability` | `rrsp_success_probability` |
| `save_current_plot` | `saved_figure_path` |
| `sanity_check` | `phase_construction_fidelity` |
| notebook plotting functions | corresponding `*_figure` functions in `plotting.py` |
| notebook execution blocks | `python -m graph_rrsp_toolkit.main` |

## Numerical behavior

- Bitstrings retain the notebooks' lexicographic ordering.
- The graph phase uses the same upper-triangular edge sum
  \(\pi\sum_{i<j}A_{ij}x_ix_j\).
- Monte Carlo sweeps use one seeded NumPy generator across each noise sweep.
- Population standard deviation (`numpy.std` with its default settings) is
  retained.
- Importing the package does not create directories, save files, show plots, or
  alter NumPy's global random state.
