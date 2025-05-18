# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['V:\\_project_\\ai_platform\\media\\models\\staging\\prediction_app.py'],
    pathex=[],
    binaries=[],
    datas=[('V:\\_project_\\ai_platform\\media\\models\\staging\\trained_model.keras', '.'), ('V:\\_project_\\ai_platform\\media\\models\\staging\\scalers.pkl', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='prediction_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
