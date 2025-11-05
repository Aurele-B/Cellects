Cellects: Cell Expansion Computer Tracking Software
===================================================

Description
-----------

Cellects is a tracking software for organisms whose shape and size change over time. 
Cellects’ main strengths are its broad scope of action, 
automated computation of a variety of geometrical descriptors, easy installation and user-friendly interface.


---
## Quick Start
### 1. With Cellects_installer.exe (windows)
- Download [Cellects_installer.exe](https://drive.google.com/file/d/1v2ppaln0LJ5QhXXq1D-zduhfun5D2ZXX/view?usp=drive_link)
- Double-click on the Cellects_installer.exe file to start installation
Note 1: Windows may warn you that it's unsafe; that's normal, because we are not a registered developer. Click "More info" and "Run Anyway".
Note 2: For the same reason, some antivirus software can prevent installation.

- To run Cellects, explore the newly created folder to find and execute Cellects.exe
<br />

### 2. Using pip (Mac, Windows or Linux)
- Install [python 3.13](https://www.python.org/downloads/release/python-3139/)
Use a terminal to write:
```bash
pip install Cellects
```
Run Cellects:
```bash
Cellects
```
To uninstall, use:
```bash
pip uninstall Cellects
```

### 3. To access the source code and close this repository  (Mac, Windows or Linux)
- Install [python 3.13](https://www.python.org/downloads/release/python-3139/)
- Install [git](https://git-scm.com/downloads)
- On Mac: also install [brew](https://brew.sh/)
- Choose a place to install Cellects and use a terminal to write:
```bash
cd path/toward/an/existing/folder/
```
Note: The repository will be cloned to that folder; if updating an existing project, use a different folder name and rename it after verifying the new version.
- Clone [Cellects repository](https://github.com/Aurele-B/Cellects.git) in terminal (or any IDE) with:
```bash
git clone https://github.com/Aurele-B/Cellects.git
cd ./Cellects
pip install --upgrade pip
python -m venv ./python_virtual_env
```
On Windows, run:
```bash
python_virtual_env\Scripts\activate
```
On Unix or MacOS, run:
```bash
source python_virtual_env/bin/activate
```
Install Cellects dependencies:
```bash
pip install .
```
Run Cellects:
```bash
Cellects
```

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
