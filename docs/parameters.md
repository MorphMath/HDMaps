# Choosing parameters

This page is a work in progress.

## Kernel bandwidth and fiber weight

Two points of neighboring objects `i` and `j` are compared through one squared distance,

```
d² = d_B(i, j)² + γ/2 · (mapped squared fiber distances, in both directions)
```

and the kernel is `exp(-d² / ε²)`.

- `epsilon` is the bandwidth: pairs with `d` well above it get almost no weight.
- `gamma` weighs the fiber distances against the base distance. The larger it is, the more a mismatch within the fibers counts, so the kernel concentrates on the pairs that the maps bring close together. As `gamma` goes to 0 the fibers stop mattering and only the base distance is left.

Either can be left as `None` to estimate it from the data:

- `epsilon` is the median non-zero distance in `base_dist`.
- `gamma` is `(epsilon / f)²`, where `f` is the median, over the fibers, of each fiber's median non-zero distance. It uses the `epsilon` you gave, or the estimated one. This makes a typical distance within a fiber count as much as one bandwidth.

`hdmaps.hdm.estimate_epsilon(base_dist)` and `hdmaps.hdm.estimate_gamma(epsilon, fiber_dists)` give the same values, for when you want to scale an estimate. `HDMResult.config` holds the values that were used.

## Number of eigenvectors

## Diffusion time

## Sinkhorn normalization

## Troubleshooting
