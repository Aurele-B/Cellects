# Contributing to Cellects

## How to Report Issues  
- File bugs/feature requests on [GitHub](https://github.com/Aurele-B/cellects/issues).

## Code Contributions  
1. Fork the repository and create a new branch:  
```bash
   git checkout -b feature/new-widget
```
2. Write tests for new features (see tests/).
3. Submit a pull request with a clear description.

## Testing
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

### Create windows executable
Start by creating a new python environment. 
If it already exists, save the __main__.spec file elsewhere and delete the environment to get the last Cellects version using pip.
And run:
```bash
python -m venv ./cellects_env
deactivate
cellects_env\Scripts\activate
pip install .
pip install pyinstaller
cd cellects_env/Lib/site-packages/cellects
```
Generate the .exe
```bash
pyinstaller __main__.py
```
To modify the .exe (icons, terminal...), add a .spec file in 'cellects_env/Lib/site-packages/cellects/' and run:
```bash
pyinstaller __main__.spec
```
Then, compress the dist folder into Cellects.zip and use NSIS to generate the installer

Note: When installing Cellects dependencies, do not use editable mode:
Note: Lighter installer with the "Solid" option