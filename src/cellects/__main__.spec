# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

import os
from pathlib import Path

CELLECTS_DIR = Path(os.path.dirname(os.path.realpath(__name__)))
print(CELLECTS_DIR)
#CELLECTS_DIR = CORE_DIR.parent

CORE_DIR = CELLECTS_DIR / "core"
GUI_DIR = CELLECTS_DIR / "gui"
IMAGE_ANALYSIS_DIR = CELLECTS_DIR / "image_analysis"
UTILS_DIR = CELLECTS_DIR / "utils"

ICONS_DIR = CELLECTS_DIR / "icons"
TEST_DIR = CELLECTS_DIR / "test"
ICON_FILE = ICONS_DIR / "myicon.ico"
CONFIG_DIR = CELLECTS_DIR / "config"


from PyInstaller.utils.hooks import collect_submodules

hiddenimports_coloredlogs = collect_submodules('coloredlogs')
hidden_imports_exif = collect_submodules('exif')
hidden_imports_ExifRead = collect_submodules('ExifRead')
hidden_imports_opencv_python = collect_submodules('opencv-python')
hidden_imports_plum_py = collect_submodules('plum-py')
hidden_imports_python_dateutil = collect_submodules('python-dateutil')
hidden_imports_screeninfo = collect_submodules('screeninfo')
all_hidden_imports = hiddenimports_coloredlogs + hidden_imports_exif + hidden_imports_ExifRead + hidden_imports_opencv_python + hidden_imports_plum_py + hidden_imports_python_dateutil + hidden_imports_screeninfo

a = Analysis(['__main__.py'],
             pathex=[CELLECTS_DIR, "C:/Directory/Scripts/Python/CellectsEnv"],
             binaries=[],
             datas=[(CONFIG_DIR, 'config'), (ICONS_DIR, 'icons'), (CORE_DIR, 'core'), (GUI_DIR, 'gui'), (IMAGE_ANALYSIS_DIR, 'image_analysis'), (TEST_DIR, 'test'), (UTILS_DIR, 'utils')],
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
          entitlements_file=None , icon=["icons/myicon.ico"])

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Cellects')
