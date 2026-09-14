# HDM_Python
[![Test package](https://github.com/frisbro303/HDM_Python/actions/workflows/test.yml/badge.svg)](https://github.com/frisbro303/HDM_Python/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A Python implementation of Horizontal Diffusion Maps (HDM), a manifold learning framework for data analysis of datasets with base-fiber structure.**

## Installation
To install the latest development version of `HDM_Python` run:
```bash
pip install git+https://github.com/MorphMath/HDMaps
```

## Introduction to Horizontal Diffusion Maps


<p align="center">
  <img src="media/hdm_demo.gif" width="600" alt="A random walk on a neighbor graph on the base manifold, lifted through the fibres">
  <br>
  <sub><em>A random walk on a neighbor graph on the base manifold (top) lifted through the fibres (bottom): hopping between objects moves to the corresponding point on each object's structure.</em></sub>
</p>


## Usage

To get started using HDM_Python, add the following import to the top of your Python file:
```python
from HDM import hdm_embed, HDMConfig
```

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.




