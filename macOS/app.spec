# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden = []
hidden += collect_submodules('hydra')
hidden += collect_submodules('omegaconf')
hidden += collect_submodules('nemo')

datas = []
datas += collect_data_files('lightning_fabric')
datas += collect_data_files('lightning')
datas += collect_data_files('pytorch_lightning')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[('assets/ffmpeg', 'ffmpeg')],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    excludes=['nemo.export'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='app',
)

# app = BUNDLE(
#     coll,
#     name='Parakeet STT API.app',
#     icon='assets/icon.icns',
#     bundle_identifier='com.Raz1ner.Parakeet-STT-API',
#     info_plist={
#         'CFBundleName': 'Parakeet STT API',
#         'CFBundleDisplayName': 'Parakeet STT API',
#         'CFBundleVersion': '1.0.0',
#         'NSHighResolutionCapable': True,
#     },
#     version='1.0.0'
# )