# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

import os
from pathlib import Path

CORE_DIR = Path(os.path.dirname(os.path.realpath(__name__)))
SRC_DIR = CORE_DIR.parent
CELLECTS_DIR = SRC_DIR.parent
ENV_DIR = CELLECTS_DIR.parent / "CellectsEnv"

print(SRC_DIR)

CONFIG_DIR = SRC_DIR / "config"
CORE_DIR = SRC_DIR / "core"
GUI_DIR = SRC_DIR / "gui"
IMAGE_ANALYSIS_DIR = SRC_DIR / "image_analysis"
TEST_DIR = SRC_DIR / "test"
UTILS_DIR = SRC_DIR / "utils"
ICONS_DIR = SRC_DIR / "icons"
ICON_FILE = ICONS_DIR / "myicon.ico"


from PyInstaller.utils.hooks import collect_submodules

hiddenimports_coloredlogs = collect_submodules('coloredlogs')
hidden_imports_exif = collect_submodules('exif')
hidden_imports_ExifRead = collect_submodules('ExifRead')
hidden_imports_opencv_python = collect_submodules('opencv-python')
hidden_imports_plum_py = collect_submodules('plum-py')
hidden_imports_python_dateutil = collect_submodules('python-dateutil')
hidden_imports_screeninfo = collect_submodules('screeninfo')
all_hidden_imports = hiddenimports_coloredlogs + hidden_imports_exif + hidden_imports_ExifRead + hidden_imports_opencv_python + hidden_imports_plum_py + hidden_imports_python_dateutil + hidden_imports_screeninfo

a = Analysis([SRC_DIR / '__main__.py'],
             pathex=[SRC_DIR, ENV_DIR],
             binaries=[],
             datas=[(CONFIG_DIR, 'config'), (CORE_DIR, 'core'), (GUI_DIR, 'gui'), (IMAGE_ANALYSIS_DIR, 'image_analysis'), (TEST_DIR, 'test'), (UTILS_DIR, 'utils'), (ICONS_DIR, 'icons')],
             hiddenimports=all_hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='Cellects',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon=["src/icons/myicon.ico"])

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Cellects')
