# Physical model and conventions

## Target family and optical mode mapping

Let `n` be the number of memory qubits and `M` the number of clients. They are
independent parameters. There are `d = 2^n` optical modes, labeled by the binary
string `x = x_0 ... x_(n-1)`, in increasing binary order with `x_0` most significant.
Qubit indices in edge lists are zero-based. There is no mode reordering.

For a simple undirected graph with edge set `E`:

```text
|G_theta> = (1/sqrt(d)) sum_x exp(i phi_x) |x>
phi_x = sum_i theta_i x_i + pi sum_(i,j in E) x_i x_j.
```

The local phase gate is `P(theta) = diag(1, exp(i theta))`. It agrees with a
conventional `Rz(theta)` up to global phase. An empty graph gives the product
of `( |0> + exp(i theta_i)|1> )/sqrt(2)`; arbitrary polar angles are outside v0.1.
This implements Appendix A and the phase encoding in Appendix B of the V3 manuscript.

## Compensation and probability

For calibrated intensity transmission `eta_x` and phase shift `chi_x`, the
surviving field in a mode is multiplied by `sqrt(eta_x) exp(i chi_x)`. The source uses

```text
S = sum_x 1/eta_x
 a_x = 1 / sqrt(eta_x * S)
 source coefficient = a_x * exp(i phi_source,x)
 sum_x a_x^2 = 1.
```

These are **field amplitudes**, not power fractions. The CSV exports both.
The source subtracts the total known `chi_x` from its phase. After all clients
and channels, each surviving mode has magnitude `1/sqrt(S)`, so the normalized
memory state is the target. The total modeled survival probability is `d/S`.
Stable scaling by `min(eta)` avoids explicitly forming overflowing inverses.

For the paper's Hamming model (Appendix D):

```text
eta_x = common * product_i eta0_i^(1-x_i) eta1_i^x_i.
```

Scalar `eta0, eta1` reproduce `common * eta0^(n-H(x)) * eta1^H(x)`.
For multiple channel segments multiply intensity transmissions and add calibrated
phases. Include each physical loss factor only once. Detector efficiency may be
included in this calibration if the assumed detector response is diagonal in the
chosen optical-mode model. More general output-port-dependent efficiencies need
an explicit erasure/detector model and are not captured by one input-mode vector.
The predicted probability excludes any physical process not included in the input
calibration or ideal mapping assumptions.

No input modes are discarded, clipped, resampled or smoothed. Zero transmission
makes preparation of a full-support graph state impossible in this inverse-loss
model and is rejected. Large dynamic range remains an experimental limitation.

## Many clients

Let `E_source` be the initial source graph. Client `k` contributes edge toggles
`Delta_k`, intended local rotation `alpha^(k)`, private hiding angle `h^(k)`, and
optional private Pauli bits `z^(k)`:

```text
phi_k(x) = sum_i [alpha_i^(k) + h_i^(k) + pi*z_i^(k)] x_i
           + pi sum_(i,j in Delta_k) x_i x_j
E_out = E_source XOR Delta_1 XOR ... XOR Delta_M
alpha = sum_k alpha^(k) mod 2pi
kappa = sum_k [h^(k) + pi*z^(k)] mod 2pi
Theta = alpha + kappa mod 2pi.
```

The source programs `graph_phase(E_source) + phi_1 - chi_total`, and is the
**only** client that programs the compensation amplitudes. Later clients program
only `phi_k`. Applying the same edge toggle in two clients cancels it, even if
that edge was initially present. Repeated edges inside a single input list are
rejected to catch mistakes. All these operators are diagonal and commute.

This is the composition rule of Appendices E/H. It does not establish a full
multi-party blindness proof, authentication, secure aggregation or collusion
resistance. A client can compute `phi_k` locally without other inputs. The central
integration exporter deliberately has all inputs and must be treated as trusted.
Private hiding vectors are not automatically transmitted anywhere.

## Heralding convention

The erasure network here is the **bitwise Walsh–Hadamard `H^n`**, with rows
`<m|H^n|x> = (-1)^(m dot x)/sqrt(d)`. It is not the cyclic `d`-dimensional QFT.
Using the ideal memory mapping in Appendix B, herald outcome `m` leaves

```text
Z^m |G_Theta>.
```

Each specified outcome has probability `1/S = (d/S)/d`. Accepting all outcomes
and tracking/applying the local Pauli corrections gives probability `d/S`.
The paper's uniform erasure projection is the `m = 00...0` case. Handling the
other outcomes here follows directly from the Hadamard row signs and is tested
explicitly. Different interferometers, mode permutations or hardware output-port
labels require their own mapping; do not substitute them silently.

To recover the intentionally rotated graph `|G_alpha>`, apply local phases
`-kappa - pi*m` modulo `2pi`. To recover the unrotated graph, also subtract `alpha`.
These physical inverse corrections reveal private angles if sent in the clear;
keep the exported private correction vector with a trusted controller.

## Measurement conventions

A measurement angle `delta` means basis
`(|0> +/- exp(i delta)|1>)/sqrt(2)`; bit 0 labels plus and bit 1 minus.
If `phi_prime` is already adapted to the computation on the unrotated final graph,
then on the prepared state `Z^m |G_Theta>` use

```text
delta = phi_prime + Theta + pi*m + pi*r mod 2pi
logical_result = server_result XOR r.
```

Here `r` is an independently chosen classical outcome-mask bit; it is distinct
from each client's preparation `z_bits`. The sign follows by evaluating
`<+_delta| P(Theta)`: it equals `<+_(delta-Theta)|`. Adding `pi*r` swaps the two
measurement outcomes. The helper's reference graph is the **unrotated** `G_out`.
If your requested measurement angles are instead relative to the intentionally
rotated `|G_alpha>`, add only `kappa + pi*m + pi*r` (pass `kappa` as the helper's
rotation vector for that chosen reference). Never count `alpha` twice.

This masking convention follows the UBQC protocol in
[Broadbent, Fitzsimons and Kashefi, arXiv:0807.4154](https://arxiv.org/abs/0807.4154),
with the additional known RGSP herald byproduct tracked explicitly. Computing
`phi_prime`, the measurement dependency structure, and any multi-party protocol
for hiding or aggregating these quantities remain the caller's responsibility.
The helper can be called on a single-qubit slice after that qubit's adaptation
has been computed.

## Source correspondence

| Implementation | Manuscript / derivation |
| --- | --- |
| `graph_phases`, `graph_state` | V3 Appendix A, Appendix B phase encoding |
| `compensation`, `hamming_efficiencies` | V3 Appendix B and Appendix D inverse-loss amplitudes |
| `xor_edges`, `client_phase_update` | V3 Appendices E and H |
| `encryption_angles_rad` | Sum of local hiding contributions in Appendix H conventions |
| `output_state(herald_bits)` | Ideal Appendix B mapping plus explicit `H^n` detector row signs |
| `measurement_instructions`, `decode_outcomes` | UBQC masking identity plus the known herald correction |

The manuscript is used as a scientific reference, not redistributed with this
software. The preparation calculator does not reuse the old plotting code's
noise/fidelity approximations. It forward-propagates the compensated amplitudes
through the declared diagonal channel and then normalizes the heralded state.
