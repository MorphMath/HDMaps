# Installation

HDMaps requires Python 3.12 or newer.

Install the latest development version from GitHub:

```bash
pip install git+https://github.com/MorphMath/HDMaps
```

## PyTorch

HDMaps depends on PyTorch. To avoid downloading the CUDA build on a machine without a GPU, install the CPU-only build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Optional extras

| Extra | Installs | Use |
|---|---|---|
| `gpu` | CuPy for CUDA 13 | Running on a GPU, see {doc}`gpu` |
| `dev` | pytest, pyright, matplotlib | Development, see {doc}`contributing` |
| `docs` | Sphinx and theme | Building these docs |

Install an extra with:

```bash
pip install "hdmaps[gpu] @ git+https://github.com/MorphMath/HDMaps"
```
