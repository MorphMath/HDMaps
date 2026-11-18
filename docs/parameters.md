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

Both must be set; there is no default.

## Number of eigenvectors

## Diffusion time

## Sinkhorn normalization

## Troubleshooting
