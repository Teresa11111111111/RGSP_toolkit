# Changelog

## 0.1.0

- Added a NumPy-only, installable experimental calculator with CLI and Python API.
- Added measured and factorized channel loss, calibrated phase pre-compensation,
  normalized input coefficients, and total/per-detector success probabilities.
- Added arbitrary numbers of phase-only clients, XOR graph transformations and
  local hiding; only the source controls amplitudes.
- Added private encryption/correction vectors, separate server measurement
  instructions and outcome decoding under explicit Hadamard/basis conventions.
- Added independent state and Born-probability tests, original-code parity checks,
  runnable configurations, model notes and package/CI checks.
- Archived the previous plotting source unchanged under `legacy/paper_toolkit`;
  excluded all legacy modules and build products from the new distribution.

Scope: equatorial phase-rotated graph/product states under calibrated diagonal
loss, ideal memory mapping and bitwise Hadamard erasure. Full MBQC adaptation,
hardware drivers and multi-party security protocols remain outside this version.
