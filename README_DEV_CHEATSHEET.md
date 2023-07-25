


# 1. Package installation
## 1.1. Packages in a virtual environment, from a requirements.txt file
## 1.1.1. Install packages
```
pip install -r src/requirements.txt 
```
## 1.1.2. Uninstall packages
```
pip uninstall -r src\requirements.txt
```

## 1.2. Create a python package as a .whl file
How to create a package:
https://packaging.python.org/en/latest/tutorials/packaging-projects/

Open the project from the dir Cellects (above src)
Go in core/cellects_paths.py and make sure of CELLECTS_DIR = CORE_DIR.parent   (without a second .parent)

Run from terminal:
```
pip install --upgrade setuptools wheel
py -m pip install --upgrade build

python setup.py sdist bdist_wheel --dist-dir ../dist

py -m build
```
cd C:\Directory\Scripts\Python\Cellects
pip install --upgrade setuptools wheel
py -m pip install --upgrade build
py -m build

## 1.3. Install a python package from a .whl file
### 1.3.1. Install a python package on windows
```
cd C:\Users\APELab
cd C:\Directory
pip install h5py
pip install typing-extensions
pip install wheel
pip install cellects-1.0.0-py3-none-any.whl
```

### 1.3.2. Uninstall a python package on windows
B/ii) Uninstall a package
```
pip uninstall cellects-1.0.0-py3-none-any.whl
pip install cellects-1.0.0-py3-none-any.whl
```


### 1.3.3. Install a python package on mac
```
python3 -m pip install cellects-1.0.0-py3-none-any.whl
```
### 1.3.4. Uninstall a python package on mac
```
python3 -m pip uninstall cellects-1.0.0-py3-none-any.whl
```

### 1.3.4. Run a python package
```
Cellects
```

# 2. Create an executable
## 1.1. Create a windows executable

Always make sure that pythonxx (for python.exe) and pythonxx/Scripts (for pip) are in windows path

### 1.1.1. Install all packages in cmd
```
python -m pip install --upgrade pip
python -m pip install -r D:\Directory\Scripts\Python\Cellects\src\requirements.txt
```
### 1.1.2. Activate the python environment
```
C:/Directory/Scripts/Python/CellectsEnv/Scripts/activate
```
### 1.1.2. Change directory
```
cd C:/Directory/Scripts/Python/Cellects/src/cellects
```
### 1.1.3. Update two lines

1) Go in core/cellects_paths.py and make sure of
```
CELLECTS_DIR = CORE_DIR.parent.parent   # (with a second .parent)
```
2) Go in the __main__.spec file \
In Analysis/pathex : after "CELLECTS_DIR, ", put the path toward the current virtual Env, e.g.:\
```
C:/Directory/Scripts/Python/CellectsEnv
```
### 1.1.4. Create .exe from cmd
```
pyinstaller __main__.spec
```
### 1.1.5. Create an installer
1) Rename the dist folder (within src\cellects)
2) Compress it into a .zip.
3) Run NSIS (download if not installed)
4) click on "Installer based on zip file "/open/Cellects.zip
5) Click on Generate, wait and close. The installer is ready.

## 1.2. Create a mac executable
### 1.1.1. Install all packages in Terminal
```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade coloredlogs exif ExifRead llvmlite numba numpy opencv-python-headless pandas plum-py psutil pyinstaller PySide2 python-dateutil pytz rawpy scipy screeninfo six
```

B/ Create a python environment
python3 -m venv /path/to/new/virtual/environment

C/ Activate the python environment
source /Users/Annuminas/Desktop/python/Venv/bin/activate

D/ Install all packages in that env (May be only usefull for pycharm)
python -m pip install -U pip setuptools 
pip install --upgrade coloredlogs exif ExifRead llvmlite numba numpy opencv-python-headless pandas plum-py psutil pyinstaller PySide2 python-dateutil pytz rawpy scipy screeninfo six

E/ Go to project location
cd /Users/Annuminas/Desktop/python/src/cellects

F/ Update two lines
1) Go in core/cellects_paths.py and make sure of 
CELLECTS_DIR = CORE_DIR.parent.parent   (with a second .parent)

2) Change one line of the __main__.spec file
In Analysis/pathex : after "CELLECTS_DIR, ", put the path toward the current virtual Env, e.g.:
/Users/Annuminas/Desktop/python/Venv

G) Create executable:
pyinstaller --onefile cellects/__main__.py
pyinstaller onefile.spec

H/ Create an ISO image (macOS)
If Homebrew is not installed, put that link in the terminal (or get an updated link here:https://brew.sh)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install create-dmg

IF already installed proceed:
Once brew and create-dmg are installed, save the shell script in the project root
and execute it to create the iso image:
chmod +x create_dmg.sh
./create_dmg.sh

