Cellects: Cell Expansion Computer Tracking Software
===================================================

Description
-----------
<div style="text-align: justify">
Cellects is a tracking software for organisms whose shape and size change over time. 
Cellects’ main strengths are its broad scope of action, 
automated computation of a variety of geometrical descriptors, easy installation and user-friendly interface.


Installation
------------
<br />

The following is four methods to install Cellects.
<br />

### 1. With Cellects_installer.exe (windows)
- Download [Cellects_installer.exe](https://drive.google.com/file/d/1v2ppaln0LJ5QhXXq1D-zduhfun5D2ZXX/view?usp=drive_link)
- Double-click on the Cellects_installer.exe file to start installation
Note 1: Windows may warn you that it's unsafe; that's normal, because we are not a registered developer. Click "More info" and "Run Anyway".
Note 2: For the same reason, some antivirus software can prevent installation.

- To run Cellects, explore the newly created folder to find and execute Cellects.exe
<br />

### 2. From this repository  (Mac, Windows or Linux)
- Install [python 3.11](https://www.python.org/downloads/release/python-3116/)
- Install [git](https://git-scm.com/downloads)
- On Mac: also install [brew](https://brew.sh/)
- Choose a place to install Cellects:
```
cd path/toward/an/existing/folder/
```
Note: The repository will be cloned to that folder; if updating an existing project, use a different folder name and rename it after verifying the new version.
- Clone [Cellects repository](https://github.com/Aurele-B/Cellects.git) in terminal (or any IDE) with:
```
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
- [When changing python version only] Remove a previous environment:
```
poetry env remove python
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

### 3. With the wheel file
- Create or download [cellects-1.0.0-py3-none-any.whl](https://drive.google.com/file/d/1W3N85LSdk5NX7wYPz4WTEgtcF1Ydr32v/view?usp=drive_link) file in a directory
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

Usage
------------
Find a usage example on video [here](https://www.youtube.com/watch?v=N-k4p_aSPC0) and/or browse the [user manual](https://github.com/Aurele-B/Cellects/blob/main/UserManual.md)

</div>
