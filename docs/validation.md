# Validation record — v0.1.0

Local validation used Python 3.12.4 and NumPy 1.26.4. The package was installed
into a dedicated virtual environment (reusing the existing NumPy installation),
then tested through its installed API and CLI, not only from source imports.

Run from a repository checkout after installation:

```sh
python -m unittest discover -s tests -v
```

The 17 tests cover:

- The manuscript's two-qubit phases and the explicit mode/bit order.
- Independent tensor-product + CZ reference states, including empty-graph products.
- Inverse-loss normalization, survival probability, uniform and highly unequal loss.
- Scalar and per-qubit Hamming losses and cascaded calibrated channel phases.
- 1, 2 and 5 clients at 2–6 qubits, independently constructed final states and graph XOR.
- All 8 herald outcomes for a three-qubit graph, including inverse hiding.
- All 64 combinations of three-bit herald and outcome masks, compared using
  joint Born probabilities in explicitly constructed measurement bases.
- Client-order invariance, local-only phase generation, valid random-mask alphabet.
- Invalid/ambiguous inputs, zero efficiency, nonfinite values, duplicate edges/JSON
  keys, unsupported coefficients on later clients, and refusal to overwrite runs.
- Export separation: later clients have no amplitude fields, and server instructions
  contain no standalone private encryption vector or outcome-mask vector.
- Direct parity against the archived original phase and amplitude functions at
  n=2 through 8 (seed 9012026). Maximum unit-phasor discrepancy was approximately
  `2.18e-14`; maximum amplitude discrepancy was `1.11e-16`.
  Phase comparisons use phasors because modulo-2π representatives differ. The
  tolerance `1e-12` allows ordinary floating-point summation/reduction roundoff.

Randomized numerical fixtures use fixed seeds (1203 and 9012026), distinct from
production mask generation, which uses Python's OS-backed `secrets` module.
There is no resampling of paper simulation data in this package.

The following CLI examples were actually run:

| Example | n | Clients | Total modeled survival probability |
| --- | ---: | ---: | ---: |
| `lossy_line.json` | 3 | 1 | 0.465300676183321 |
| `product_state.json` | 2 | 1 | 0.596449704142012 |
| `three_clients.json` | 3 | 3 | 0.46120575656357 |

All produced unit-norm inputs and conditioned fidelities equal to one within
floating-point roundoff. `local_client.json` was also run and compared against
the second client's update in the combined three-client example.

These checks establish correctness under the declared ideal diagonal-channel
model. They do not validate laboratory calibration, noise immunity, achievable
modulator dynamic range or cryptographic security. The GitHub Actions workflow
runs the tests, examples and build on Python 3.10, 3.12 and 3.13; its live status
is authoritative for those environments.

A pure-Python wheel and source archive were built successfully using setuptools.
The wheel was installed without fetching dependencies and the CLI examples ran
from the installed package. The wheel includes only the new package; no original
plotting code, manuscript, experimental output or private configuration is bundled.
