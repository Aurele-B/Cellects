<h1>
  <img src="https://raw.githubusercontent.com/Aurele-B/cellects/main/.github/icon.png"
       width="42"
       style="vertical-align: middle; margin-right: 12px;">
  Cellects: Cell Expansion Computer Tracking Software
</h1>

[![PyPI version](https://img.shields.io/pypi/v/cellects.svg?style=flat-square)](https://pypi.org/project/cellects/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cellects)
[![Python versions](https://img.shields.io/pypi/pyversions/cellects.svg?style=flat-square)](https://pypi.org/project/cellects/)
[![License](https://img.shields.io/pypi/l/cellects.svg?style=flat-square)](https://github.com/Aurele-B/cellects/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/Aurele-B/cellects.svg?style=flat-square)](https://github.com/Aurele-B/cellects/stargazers)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Aurele-B/Cellects/.github%2Fworkflows%2Frelease.yml)
![Coverage](https://raw.githubusercontent.com/Aurele-B/cellects/gh-pages/badges/coverage.svg)

Description
-----------

Cellects is a tracking software for organisms whose shape and size change over time. 
Cellects’ main strengths are its broad scope of action, automated computation of a variety of geometrical descriptors, 
easy installation and user-friendly interface.

<figure>
  <img src="doc/static/UserManualFigure1.png" alt="Cellects first window"
       style="display:block;float:none;margin-left:auto;margin-right:auto;width:100%">
  <figcaption><strong>Figure 1:</strong> Cellects first window</figcaption>
</figure>

---

## Installation (Short version)
Install using our Windows installer: [Cellects_installer.exe](https://github.com/Aurele-B/Cellects/releases/)

Or, install via pip:
```bash
pip install cellects
```
Any difficulties? follow our [complete installation tutorial](https://aurele-b.github.io/Cellects/installation/)

---

## Quick Start
Run in terminal:
```bash
cellects
```

---

## Documentation

Cellects' workflow is described in a [complete documentation](https://aurele-b.github.io/Cellects/). It includes:
- [**What is Cellects**](https://aurele-b.github.io/Cellects/what-is-cellects/): Purpose of the software, usable data and introduction of its user manual
- [**Setting up a first analysis**](https://aurele-b.github.io/Cellects/first-analysis/): Step-by-step workflows for data localization, image analysis and video tracking 
- [**Improving the analysis**](https://aurele-b.github.io/Cellects/advanced/): Customization options, batch processing, parameter tuning.
- [**Use cases**](https://aurele-b.github.io/Cellects/use-cases/): Real-world cases using the GUI (interface) and the API (scripts).
- [**Contributing**](https://aurele-b.github.io/Cellects/contributing/): Report bugs and feature requests; contribute; testing and documentation processes.
- [**API Reference**](https://aurele-b.github.io/Cellects/api/): Auto-generated from source code docstrings (see [Build Documentation]).

---

## Use Cases

See [use cases](https://aurele-b.github.io/Cellects/use-cases/) for real-world examples:
- Automated Physarum polycephalum tracking using GUI
- Automated Physarum polycephalum tracking using API
- Colony growth tracking

---

## Contributing

We welcome contributions!  
1. Fork the repository and create a new branch.
2. Submit issues/PRs via [GitHub](https://github.com/Aurele-B/cellects/issues).

For developer workflows, see [**Contributing**](https://aurele-b.github.io/Cellects/contributing/).

---

## License & Citation

GNU GPL3 License (see [LICENSE](https://github.com/Aurele-B/cellects/blob/main/LICENSE)).

To cite Cellects, use:
```bibtex
@article{boussard2024cellects,
  title={Cellects, a software to quantify cell expansion and motion},
  author={Boussard, Aur{\`e}le and Arrufat, Patrick and Dussutour, Audrey and P{\'e}rez-Escudero, Alfonso},
  journal={bioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

## Testing

Run unit tests with:
```bash
pip install -e ".[test]"
pytest
```

---

## Resources
- [Usage example (video)](https://www.youtube.com/watch?v=N-k4p_aSPC0)
