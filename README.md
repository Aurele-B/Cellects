Cellects
=================

Description
-----------
Cellects is a tracking software for organisms whose shape and size change over time. 
Cellects’ main strengths are its broad scope of action, 
automated computation of a variety of geometrical descriptors, easy installation and user-friendly interface.

Installation
------------
<br />

To install Cellects, you can:
<br />

### 1. Install it in a click using Cellects.exe on windows.
- Download [Cellects_installer.exe](https://drive.google.com/file/d/1v2ppaln0LJ5QhXXq1D-zduhfun5D2ZXX/view?usp=drive_link)
- Double-click on the Cellects_installer.exe file to start installation
<br />
Note 1: Windows may warn you that it's unsafe; that's normal, because we are not a registered developer. Click "More info" and "Run Anyway".
<br />
Note 2: For the same reason, other antiviruses may prevent the installation.
- Finds the app in Cellects/Cellects/Cellects.exe and run it

<br />

### 2. From this repository
- Install [python 3.11](https://www.python.org/downloads/release/python-3116/)
- Install [git](https://git-scm.com/downloads)
- Clone [Cellects repository](https://github.com/Aurele-B/Cellects.git) in terminal (or any IDE) with:
```
cd path/toward/an/existing/folder/
git clone https://github.com/Aurele-B/Cellects.git
cd ./Cellects
```
- Upgrade pip:
```
pip install --upgrade pip
```
- Install poetry:
```
pip install poetry
```
- Install all necessary packages:
```
poetry install
```
- Run Cellects:
```
poetry run Cellects
```
<br />

#### Once installed, run it with:
```
cd path/toward/Cellects
poetry run Cellects
```

<br />

### 3. Install it in the python global environment with a wheel file
- Create or download "cellects-1.0.0-py3-none-any.whl" file in a directory
- Open a terminal and go in that directory
```
cd path/toward/the/wheel/file/folder/
```
- Uninstall any previous version with:
```
pip uninstall cellects-1.0.0-py3-none-any.whl
```
- Install all necessary packages and Cellects:
```
pip install --upgrade pip
pip install h5py
pip install typing-extensions
pip install wheel
pip install cellects-1.0.0-py3-none-any.whl
```
- Run Cellects:
```
Cellects
```
#### To get the wheel file:
Contact me or create the wheel file yourself by:
- Following section 2. to install Cellects 
- Creating the wheel file with:
```
poetry build
```

Usage
------------
Browse the user manual in the supporting information of the paper describing the software.