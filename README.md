Cellects
=================

Description
-----------
Cellects is a tracking software for organisms whose shape and size change over time. 
Cellects’ main strengths are its broad scope of action, 
automated computation of a variety of geometrical descriptors, easy installation and user-friendly interface.

Installation
------------
To install Cellects, you can:

### 1. Install it in a click using Cellects.exe on windows.
- Double-click on the Cellects.exe file to start installation
- Finds the app in Cellects/Cellects/Cellects.exe and run it

### 2. From this repository
- Install python 11 (https://www.python.org/downloads/release/python-3116/)
- Install git (https://git-scm.com/downloads)
- Clone this repository (https://github.com/Aurele-B/Cellects.git) in terminal (or any IDE)
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
- Make sure to activate the poetry environment:
```
poetry shell
```
- Run Cellects:
```
Cellects
```

### 3. Install it as a python package using the wheel file named "cellects-1.0.0-py3-none-any.whl"
- Uninstall any previous version with:
```
pip uninstall cellects-1.0.0-py3-none-any.whl
```
- Put the cellects-1.0.0-py3-none-any.whl file in a directory
- Open a terminal and go in that directory
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


Usage
------------
Browse the user manual in the supporting information of the paper describing the software.