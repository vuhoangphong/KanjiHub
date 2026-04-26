# -*- mode: python ; coding: utf-8 -*-
# Build spec dành riêng cho macOS (dùng trong GitHub Actions)
from PyInstaller.utils.hooks import collect_all
import os

# ── Data files ──────────────────────────────────────────────────────────────
datas = [
    ('icon.png', '.'),
    ('config.json', '.'),
    ('progress.json', '.'),
    ('history.json', '.'),
    # Font được tải về bởi workflow trước khi build
    ('NotoSansJP-VF.ttf', '.'),
]

binaries = []
hiddenimports = [
    'customtkinter',
    'edge_tts',
    'google.genai',
    'google.genai.types',
    'reportlab',
    'reportlab.pdfgen',
    'reportlab.pdfgen.canvas',
    'reportlab.lib',
    'reportlab.lib.pagesizes',
    'reportlab.lib.units',
    'reportlab.lib.colors',
    'reportlab.pdfbase',
    'reportlab.pdfbase.pdfmetrics',
    'reportlab.pdfbase.ttfonts',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'requests',
]

tmp_ret = collect_all('customtkinter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('edge_tts')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('reportlab')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['gen_database', 'ai_fill_missing', 'inject_n3', 'inject_n2', 'inject_n1'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='KanjiHub',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='KanjiHub',
)
app = BUNDLE(
    coll,
    name='KanjiHub.app',
    icon='icon.png',
    bundle_identifier='com.vietkanji.kanjihub',
    info_plist={
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
    },
)
