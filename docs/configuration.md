# Configuration reference

`rgsp compile CONFIG.json --out NEW_DIRECTORY` accepts one JSON object.
Unknown keys, repeated JSON keys, nonfinite values and invalid dimensions are
rejected. Omitted fields have the defaults below; `null` is allowed for optional
angle/bit vectors and means their zero default. Efficiencies, explicitly supplied herald bits and required measurement fields cannot be null.

| Field | Type / default |
| --- | --- |
| `qubits` | Required integer 1–16 |
| `source_graph_edges` | List of `[i,j]`; default `[]`; indices 0 through n−1 |
| `clients` | Nonempty list of client objects; default `[{}]` |
| `channels` | Nonempty list of calibrated segments; omitted means ideal transmission |
| `herald_bits` | n-vector of 0/1; omitted means all-zero **preview only** |
| `measurement` | Optional object; requires explicit observed `herald_bits` |

Each client has only these fields (all optional):

| Field | Type / default |
| --- | --- |
| `graph_toggle_edges` | Simple undirected edge list; default `[]` |
| `local_angles_rad` | n-vector of logical rotations; default zeros |
| `hiding_angles_rad` | n-vector of private hiding angles; default zeros |
| `z_bits` | n-vector of private Pauli Z bits; default zeros |

The first client is physically the source. Its update is already included in
`source_modes.csv`. Subsequent clients are named `client_02`, `client_03`, etc.
No client object accepts coefficients or amplitudes; the source's amplitudes are
computed from the full path calibration.

For an independently operating client, `rgsp client CONFIG.json --out DIR`
accepts exactly `{"qubits": n, "client": {...}}`. It does not require the initial
graph, the target graph, the loss model or any other client's input.

## Channels

A measured segment:

```json
{
  "model": "measured",
  "intensity_efficiencies": [0.9, 0.7, 0.6, 0.4],
  "phases_rad": [0.1, -0.2, 0.05, 0.3]
}
```

There must be exactly `2**n` entries. Efficiencies must be in `(0,1]` and are
intensity/power transmission, **not** field transmission or loss in dB.
Convert attenuation `loss_dB >= 0` using `eta = 10**(-loss_dB/10)`.
`phases_rad` defaults to zeros and represents calibrated deterministic phase.

A factorized Hamming segment:

```json
{"model": "hamming", "eta0": 0.96, "eta1": 0.8, "common": 0.7}
```

`eta0`, `eta1` may each be scalars or n-vectors; default 1. `common` is a scalar,
default 1. An optional `phases_rad` still has `2**n` entries. The total intensity
is the product over all segments; the source compensates it once. Segment order
is immaterial only under the diagonal, mode-preserving model assumed here.

## Optional measurements

```json
{
  "adapted_angles_rad": [0, 0.7853981633974483],
  "outcome_mask_bits": [1, 0]
}
```

Both fields are required and have n entries. Supply actual herald bits at the
root. No server message is exported unless this block exists. Changing outcomes
or MBQC adaptation requires new instructions; an assumed zero herald vector is
not an experimental correction record.

## Outputs and reproducibility

All phase outputs lie in `[0, 2*pi)` up to floating-point rounding. Coefficients
are normalized complex field amplitudes. The summary's `predicted_conditioned_fidelity`
is an ideal-model self-consistency diagnostic, not an experimental measurement.
It may differ from 1 by floating-point roundoff and is not artificially clipped.

A private copy of the exact config is saved alongside the exports. On POSIX the
new output directory is created with mode 0700; access control and secure storage
on other systems remain the operator's responsibility. Existing directories are
rejected. No network requests, uploads or hardware actions occur when running
the package. Keep local experiments under ignored `runs/` directories.
