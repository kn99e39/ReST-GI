"""WL111 reviewer package: copy reviewer-facing artifacts and the decision JSON into evaluation/111/ with a provenance
manifest (sha256 of every copied file and of the evidence it came from). Never modifies evidence.

python package_r111.py EVIDENCE_ROOT FIGS_DIR EVAL_DIR COMMIT
"""
import shutil
import sys
from pathlib import Path

import common_r111 as C


def main(root, figs, dest, commit):
    root, figs, dest = Path(root), Path(figs), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    man = {"batch": "WL111", "code_commit": commit, "protocol_sha256": C.sha(C.PROTOCOL), "files": {}, "sources": {}}
    for f in sorted(figs.iterdir()):
        if f.suffix in (".png", ".gif", ".json"):
            shutil.copy2(f, dest / f.name)
            man["files"][f.name] = {"sha256": C.sha(dest / f.name), "generator": "figs_r111.py"}
    for name, src in (("analysis111.json", root / "analysis.json"), ("design111.json", root / "design.json"),
                      ("synth111.json", root / "synth" / "synth.json"), ("qualification111.json", root / "refs" / "qualification.json"),
                      ("sealed_views111.json", root / "data" / "sealed_views.json")):
        if src.exists():
            shutil.copy2(src, dest / name)
            man["files"][name] = {"sha256": C.sha(dest / name), "source": str(src.relative_to(root))}
    for sub in ("static", "seq", "timing", "refs", "stageA", "synth"):
        d = root / sub
        if d.exists():
            for f in sorted(d.glob("*.json")):
                man["sources"][f"{sub}/{f.name}"] = C.sha(f)
    C.write(dest / "manifest111.json", man)
    print("PACKAGED", len(man["files"]), "files;", len(man["sources"]), "evidence JSON hashed", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
