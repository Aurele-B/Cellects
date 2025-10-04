Cellects: Cell Expansion Computer Tracking Software
===================================================

Description
-----------

Cellects is a tracking software for organisms whose shape and size change over time. 
Cellects’ main strengths are its broad scope of action, 
automated computation of a variety of geometrical descriptors, easy installation and user-friendly interface.


---

## Quick Start


⚠️ **Note:** At this stage, Cellects is available **only from source**.  
You will need **Miniconda3** and **git** installed on your system.

- Install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)  
  (choose the installer for your operating system).  
- Install [git](https://git-scm.com/downloads)  
  (also available through package managers like `apt`, `brew`, or `choco`).

Once these prerequisites are installed, you can set up Cellects as follows:

```bash
# Clone the repository
git clone https://github.com/Aurele-B/Cellects.git
cd Cellects

# Create and activate the environment
conda env create -f conda/env.yml
conda activate cellects-dev

# Install the package in editable mode
pip install -e .
```

Launch the application:
```bash
Cellects
```

---

## Developer Guide

### Run Tests
Cellects uses `pytest` + `pytest-cov`.  
Install test dependencies:

```bash
pip install -e ".[test]"
```

Run the test suite (with coverage enabled by default via `pyproject.toml`):

```bash
pytest
```

You can access the coverage report with `coverage html` and open `htmlcov/index.html` in your browser.

```bash
open htmlcov/index.html        # macOS
xdg-open htmlcov/index.html    # Linux
start htmlcov\index.html       # Windows (PowerShell)
```

Or explicitly:
```bash
pytest --cov=src/cellects --cov-report=term-missing
```

### Build Documentation
Install doc dependencies:

```bash
pip install -e ".[doc]"
```

Serve the docs locally:
```bash
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Resources
- [User manual](https://github.com/Aurele-B/Cellects/blob/main/_old_doc/UserManual.md)  
- [Usage example (video)](https://www.youtube.com/watch?v=N-k4p_aSPC0)
