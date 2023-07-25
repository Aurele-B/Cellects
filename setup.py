from setuptools import setup, find_packages

setup(
    name='Cellects',
    version='1.0.0',
    description='Cell Expansion Computer Tracking Software.',
    long_description='A tracking software for organisms whose shape and size change over time. Cellects’ main strengths are its broad scope of action, automated computation of a variety of geometrical descriptors, easy installation and user-friendly interface.',
    url='https://github.com/Aurele-B/Cellects',
    author='Aurèle Boussard',
    author_email='aurele.boussard@gmail.com',
    maintainer='Aurèle Boussard',
    maintainer_email='aurele.boussard@gmail.com',
    license='GNU General Public License v3.0',

    # packages=find_packages("src"), #['src/config', 'src/core', 'src/gui', 'src/image_analysis', 'src/test', 'src/utils'], # find_packages('src'),#[src/config, src/core, src/gui, src/image_analysis, src/utils]
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        "coloredlogs",
        "exif",
        "ExifRead",
        "llvmlite",
        "numba",
        "numpy",
        "opencv-python",
        "pandas",
        "plum-py",
        "psutil",
        "pyinstaller",
        "PySide6",
        "python-dateutil",
        "pytz",
        "rawpy",
        "scipy",
        "screeninfo",
        "six"
    ],
    tests_require=[
        'unittest'
    ],
    entry_points={
        'console_scripts': [
            'cellects = cellects.__main__:run_cellects'
        ],
    },
    zip_safe=True,
    package_data={'cellects': ['config/*',
                               'core/*',
                               'gui/*',
                               'icons/*',
                               'image_analysis/*',
                               'test/*',
                               'utils/*']},
)
