# HDMaps
[![Test package](https://github.com/MorphMath/HDMaps/actions/workflows/test.yml/badge.svg)](https://github.com/MorphMath/HDMaps/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A Python implementation of Horizontal Diffusion Maps (HDM), a manifold learning framework for data analysis of datasets with base-fiber structure.**

## What is HDMaps?

HDM extends diffusion maps to collections of related data objects, such as a shape or image collection. A random walk on the collection (the base) is lifted through correspondence maps across each object's internal structure (the fiber). This produces a shared coordinate system that jointly embeds each object's internal structure and the pairwise relations between objects.

<p align="center">
  <img src="media/hdm_demo.gif" width="600" alt="A random walk on a neighbor graph on the base manifold, lifted through the fibres">
  <br>
  <sub><em>A random walk on a neighbor graph on the base manifold lifted through the fibres: hopping between objects moves to the corresponding point on each object's structure.</em></sub>
</p>


## Installation
To install the latest development version of `HDM_Python` run:
```bash
pip install git+https://github.com/MorphMath/HDMaps
```

## Documentation and Usage
For a short, accesible overview of the theory, see the [Introduction to Horizontal Diffusion Maps](docs.md/#theory). For the full treatment, see the [paper](https://www.sciencedirect.com/science/article/pii/S1063520318302215).

Documentation is in [docs.md](docs.md). See [examples/](examples/) for usage examples.

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
