"""WL83 direct fits: the WL82 fit82.py command line, unchanged, run with the closure83 robust-refraction patch installed.

  python fit83.py <data> <set> <arm> <mode> <objective> <steps> <init> <out>
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
import closure83  # noqa: E402,F401
import fit82  # noqa: E402

if __name__ == '__main__':
    fit82.main()
    os._exit(0)
