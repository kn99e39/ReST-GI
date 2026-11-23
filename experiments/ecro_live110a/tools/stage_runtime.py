"""Stage the ECRO runtime tree (repository layout) for a standalone build.

  python tools/stage_runtime.py <out_dir> [--repo <ReST-GI root>] [--checkpoints <dir with configs.json, sss/, near/>]

Copies exactly the files listed in rest_ecro_live/runtime_manifest.json (11 modules, 5 data files) and the WL100 checkpoints /
configs (gitignored in the repository: taken from --checkpoints, default <repo>/results/ecro_material100), verifies the
checkpoint hashes, and writes <out_dir>/RUNTIME_MANIFEST.json (sha256 of every staged file, source commit, stdlib / third-party
imports of the staged modules for the PyInstaller hidden-import list).
"""
import argparse, ast, hashlib, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'rest_ecro_live'
M = json.loads((PKG / 'runtime_manifest.json').read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def imports_of(path):
    tree = ast.parse(Path(path).read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split('.')[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--repo', default=str(HERE.parents[2]))
    ap.add_argument('--checkpoints', default=None)
    a = ap.parse_args()
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    ck = Path(a.checkpoints).resolve() if a.checkpoints else repo / 'results' / 'ecro_material100'
    if out.exists():
        sys.exit(f'{out} exists; remove it first')
    files = {}
    for rel in M['modules'] + M['data']:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dst)
        files[rel] = sha(dst)
    for rel, want in M['checkpoints'].items():
        src = ck / Path(rel).relative_to('results/ecro_material100')
        got = sha(src)
        if got != want:
            sys.exit(f'checkpoint hash mismatch for {src}: {got} != {want}')
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[rel] = got
    local = {Path(r).stem for r in M['modules']}
    imps = sorted(set().union(*(imports_of(out / r) for r in M['modules'])) - local)
    meta = {'source_repo': str(repo),
            'source_commit': subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'files': files, 'module_imports': imps}
    (out / 'RUNTIME_MANIFEST.json').write_text(json.dumps(meta, indent=1))
    print('STAGED', len(files), 'files ->', out, '| imports:', ' '.join(imps))


if __name__ == '__main__':
    main()
