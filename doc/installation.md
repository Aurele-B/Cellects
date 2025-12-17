## Installation

Choose the method that matches your OS and how you want to use Cellects.

=== "Windows (Installer)"

    ### Install with `Cellects_installer.exe`

    1. Download **Cellects_installer.exe**:  
       [Cellects_installer.exe](https://github.com/Aurele-B/Cellects/releases/tag/0.2.1)

    2. Double-click `Cellects_installer.exe` to start the installation.

    3. To run Cellects, open the newly created folder and launch **`Cellects.exe`**.

    !!! info "Windows security warning"
        Windows may warn you that the installer is unsafe (because we are not a registered developer).  
        Click **More info** → **Run anyway**.

    !!! warning "Antivirus software"
        Some antivirus software may block or slow down the installation for the same reason.


=== "All OS (pip)"

    ### Install with `pip` (macOS / Windows / Linux)

    !!! note "Prerequisite"
        Install **Python 3.13**:  
        [Python 3.13](https://www.python.org/downloads/release/python-3139/)

    #### Optional but recommended: use a virtual environment

    ```bash
    cd path/toward/an/existing/folder/
    python -m venv ./cellects_env
    ```

    Activate it:

    === "Windows"
        ```cmd
        cellects_env\Scripts\activate
        ```

    === "macOS / Linux"
        ```cmd
        source cellects_env/bin/activate
        ```

    !!! tip "Why use a virtual environment?"
        It prevents compatibility issues with other Python projects.  
        You’ll just need to **activate it** each time before running Cellects.

    #### Install

    ```bash
    pip install cellects
    ```

    #### Run

    ```bash
    Cellects
    ```

    #### Uninstall

    ```bash
    pip uninstall cellects
    ```

=== "All OS (Source)"

    ### Install from source (macOS / Windows / Linux)

    !!! note "Prerequisites"
        - Install **Python 3.13**: [Python 3.13](https://www.python.org/downloads/release/python-3139/)  
        - Install **git**: [git](https://git-scm.com/downloads)  
        - On macOS, install **Homebrew**: [brew](https://brew.sh/)

    #### Clone the repository

    ```bash
    cd path/toward/an/existing/folder/
    ```

    !!! warning "Folder choice"
        The repository will be cloned into this folder.  
        If you are updating an existing project, clone into a **new folder name** and rename it only after verifying the new version.

    ```bash
    git clone https://github.com/Aurele-B/Cellects.git
    cd ./Cellects
    pip install --upgrade pip
    python -m venv ./cellects_env
    ```

    Activate the environment:

    === "Windows"
        ```cmd
        cellects_env\Scripts\activate
        ```

    === "macOS / Linux"
        ```cmd
        source cellects_env/bin/activate
        ```

    #### Install dependencies (editable mode)

    ```bash
    pip install -e .
    ```

    #### Run

    ```bash
    Cellects
    ```