# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for "ReST-GI ECRO Live.app" (macOS, Apple Silicon). Driven by build_macos.sh:
#   ECRO_RUNTIME  staged runtime tree (tools/stage_runtime.py)        default build/runtime
#   ECRO_LIBLLVM  libLLVM.dylib to bundle (Dr.Jit LLVM backend)        default: not bundled (Homebrew llvm required at run time)
import json, os
from PyInstaller.utils.hooks import collect_all

HERE = os.path.abspath(SPECPATH)
RUNTIME = os.environ.get('ECRO_RUNTIME', os.path.join(HERE, 'build', 'runtime'))
LLVM = os.environ.get('ECRO_LIBLLVM')
man = json.load(open(os.path.join(RUNTIME, 'RUNTIME_MANIFEST.json')))

datas, binaries, hidden = [], [], []
for pkg in ('mitsuba', 'drjit'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h
datas += [(RUNTIME, 'runtime'),
          (os.path.join(HERE, 'rest_ecro_live', 'runtime_manifest.json'), 'rest_ecro_live'),
          (os.path.join(HERE, 'rest_ecro_live', 'reference'), 'rest_ecro_live/reference')]
if LLVM:
    binaries += [(LLVM, '.')]           # found by backend._find_libllvm as <_MEIPASS>/libLLVM.dylib
# the runtime modules are loaded from runtime/ at run time (not analysed by PyInstaller): declare what they import
hidden += [m for m in man['module_imports'] if m not in ('drjit', 'mitsuba')]
hidden += ['rest_ecro_live.app', 'rest_ecro_live.backend', 'rest_ecro_live.runtime', 'rest_ecro_live.selfcheck',
           'PIL.ImageTk', 'tkinter', 'importlib.util', 'dataclasses', 'glob']

a = Analysis([os.path.join(HERE, 'launcher.py')], pathex=[HERE], datas=datas, binaries=binaries, hiddenimports=hidden,
             excludes=['triton'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='rest-ecro-live', console=False, argv_emulation=False,
          target_arch=os.environ.get('ECRO_TARGET_ARCH', 'arm64'))
coll = COLLECT(exe, a.binaries, a.datas, name='rest-ecro-live')
app = BUNDLE(coll, name='ReST-GI ECRO Live.app', bundle_identifier='org.rest-gi.ecro-live',
             info_plist={'NSHighResolutionCapable': True, 'CFBundleShortVersionString': '0.1.0',
                         'LSMinimumSystemVersion': '12.0'})
