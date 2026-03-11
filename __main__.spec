# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Determine platform
IS_MAC = sys.platform == 'darwin'
IS_WINDOWS = sys.platform == 'win32'

# Platform-specific icon paths
if IS_MAC:
    icon_file = "src/cellects/icons/cellects_icon.icns"
    icon_data = ("src/cellects/icons/cellects_icon.icns", "icons")
elif IS_WINDOWS:
    icon_file = "src/cellects/icons/cellects_icon.ico"
    icon_data = ("src/cellects/icons/cellects_icon.ico", "icons")
else:
    icon_file = "src/cellects/icons/cellects_icon.png"
    icon_data = ("src/cellects/icons/cellects_icon.png", "icons")

# Data files to include
datas = []
if icon_data:
    datas.append(icon_data)

a = Analysis(
    ['src/cellects/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cellects',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window - GUI mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file if icon_file else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cellects',
)

# Mac-specific: Create .app bundle
if IS_MAC:
    app = BUNDLE(
        coll,
        name='Cellects.app',
        icon='src/cellects/icons/cellects_icon.icns',
        bundle_identifier='org.github.cellects',  # Standard open-source format
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleName': 'Cellects',
            'CFBundleDisplayName': 'Cellects',
            'CFBundleShortVersionString': os.environ.get('APP_VERSION', '1.0.0'),
            'CFBundleVersion': os.environ.get('APP_VERSION', '1.0.0'),
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': 'CLCT',
            'LSMinimumSystemVersion': '10.13.0',  # macOS High Sierra or later
            'NSHumanReadableCopyright': 'Copyright © 2025 Cellects Project. Licensed under GPL-3.0.',
            'LSApplicationCategoryType': 'public.app-category.education',
            'NSRequiresAquaSystemAppearance': 'False',  # Supports dark mode
        },
    )
