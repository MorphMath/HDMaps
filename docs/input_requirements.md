# Input requirements

{func}`~hdmaps.hdm.run_hdm` takes three inputs besides the config: the base distances, the fiber distances and the maps between fibers. All matrices are {class}`scipy.sparse.csr_matrix`.

`run_hdm` checks these requirements before it starts and raises a `ValueError` when one is broken. The message starts with the requirement's id, shown in brackets below, e.g. `D3: fiber_dists[4]: diagonal not fully stored`. The full list, with the reasons and the quotes from the paper they follow from, is in `hdmaps/hdm/validation.py`.

## Base distances

`base_dist` is an `(n, n)` matrix of distances between the `n` data samples.

- It is a square CSR matrix (D1) without duplicate entries (D4).
- Its entries are finite and at least 0 (D2).
- The full diagonal must be stored, including its zeros (D3).
- A missing entry means the two samples are not neighbors. Only stored entries become edges of the base graph.
- The matrix is symmetrized, so an entry stored in only one direction counts with half weight.
- The joint graph over all points should be connected, which needs a connected base graph (S2). It is checked on the joint kernel before the eigenvectors are computed: a graph in more than one piece raises an error that gives the number of pieces and the sizes of the largest.

## Fiber distances

`fiber_dists[i]` is an `(n_i, n_i)` matrix of distances between the points of sample `i`.

- There is one per sample, and each has at least one point (S1).
- The same requirements as for `base_dist` apply: a square CSR matrix (D1) of finite distances at least 0 (D2), with the full diagonal stored, including its zeros (D3), and no duplicate entries (D4).
- A missing entry means the distance is unknown, not that the points are far apart. A stored `0.0` means the points coincide.
- The matrix is symmetrized, so an entry stored in only one direction counts with half weight.

## Maps

`maps[i, j]` is an `(n_i, n_j)` matrix mapping the points of sample `i` to the points of sample `j`. Row `p` holds the weights of point `p` over the points of sample `j`.

- `maps.data` has one item per sample (S1).
- Both `maps[i, j]` and `maps[j, i]` are needed for every pair of neighbors in `base_dist` (M1). That includes each sample with itself: `maps[i, i]` must be given, and is normally the identity.
- It is a CSR matrix of shape `(n_i, n_j)` (M2).
- Its entries are at least 0, and every row sums to 1, within 0.01 (M3).
- It stores no explicit zeros (M4): every stored entry in row `p` counts as a target of point `p`. Remove them with `eliminate_zeros()`.
- It has no duplicate entries (M5).
- Any two points of sample `j` that one point is mapped to have a stored distance in `fiber_dists[j]`, which may be 0 (M6). A map with one target per point always satisfies this.
- A point of sample `i` is only related to point `q` of sample `j` if every target of the point has a stored fiber distance to `q`. Otherwise the pair is dropped.

Maps are checked as they are read, and only the maps of base edges are read. A {class}`~hdmaps.mapping.MapBundle` with `DirStorage` or `PackedStorage` still computes every map its mask allows when it is created, so pass a mask that matches the neighbors in `base_dist` when maps are expensive.

## Dtypes

Inputs whose dtype differs from `config.dtype` are cast to it, with one warning for each of `base_dist`, `fiber_dists` and `maps`.

## Config

- `config.epsilon` and `config.gamma` are each `None` or greater than 0 (C1). `epsilon` is the kernel bandwidth and `gamma` the weight of the fiber distances against the base distance. When `None`, they are estimated; see [Choosing parameters](parameters.md). An estimate that is 0 or infinite, when there are no non-zero distances, raises an error asking to set the value.
- `config.t` is greater than 0 (C2).
- `config.num_eigenvectors` is at least 1 and less than the total number of points minus 1 (C3).
