# Input requirements

{func}`~hdmaps.hdm.run_hdm` takes three inputs besides the config: the base distances, the fiber distances and the maps between fibers. All matrices are {class}`scipy.sparse.csr_matrix`.

## Base distances

`base_dist` is an `(n, n)` matrix of distances between the `n` data samples.

- The full diagonal must be stored, including its zeros.
- Its dtype must equal `config.dtype`.
- A missing entry means the two samples are not neighbours. Only stored entries become edges of the base graph.
- The matrix is symmetrized, so an entry stored in only one direction counts with half weight.
- The base graph must be connected.

## Fiber distances

`fiber_dists[i]` is an `(n_i, n_i)` matrix of distances between the points of sample `i`.

- The full diagonal must be stored, including its zeros.
- A missing entry means the distance is unknown, not that the points are far apart. A stored `0.0` means the points coincide.

## Maps

`maps[i, j]` is an `(n_i, n_j)` matrix mapping the points of sample `i` to the points of sample `j`. Row `p` holds the weights of point `p` over the points of sample `j`.

- Its dtype must equal `config.dtype`.
- Both `maps[i, j]` and `maps[j, i]` are needed for every pair of neighbours in `base_dist`.
- Avoid storing explicit zeros. Every stored entry in row `p` counts as a target of point `p`.
- A point of sample `i` is only related to point `q` of sample `j` if every target of the point has a stored fiber distance to `q`. Otherwise the pair is dropped.

Every map allowed by the {class}`~hdmaps.mappings.MapBundle` mask is computed when `run_hdm` validates its inputs. With the default mask that is all `n²` pairs, so pass a mask that matches the neighbours in `base_dist` when maps are expensive.

## Config

`config.fiber_epsilon` must be set. `config.base_epsilon` is estimated from `base_dist` when left as `None`.
