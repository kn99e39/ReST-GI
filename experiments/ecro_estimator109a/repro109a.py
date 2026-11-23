"""WL109A Stage A: reproduce the published WL109 metrics from the authenticated WL109 data, CPU only.

python repro109a.py WL109_EVIDENCE_DIR OUT_DIR

1. identities: every WL109 source and the protocol on disk equal the hashes recorded in the WL109 run manifest
   (amendment2.code_sha256, protocol_sha256); the published metrics109.json equals the committed reviewer copy hash.
2. rerun of the unchanged analyze109.main on the WL109 run (it re-verifies all 4,992 raw-file hashes and the WL107 reference
   hashes) with torch on the CPU: the only change is device placement ("cuda" -> "cpu"), applied by a shim around
   torch.as_tensor; the analysis file itself is imported unchanged.
3. every numeric leaf of the rerun equals the published metrics (rel <= 1e-8 or abs <= 1e-12); decision and verdict exact.
This process never initialises CUDA (CUDA_VISIBLE_DEVICES must be empty).
"""
import json
import math
import os
from pathlib import Path
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "Stage A runs CPU-only (CUDA_VISIBLE_DEVICES='')"
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("WL109A_THREADS", "6")))
_as_tensor = torch.as_tensor


def _cpu_as_tensor(x, *a, device=None, **k):
    if isinstance(device, str) and device.startswith("cuda"):
        device = "cpu"
    return _as_tensor(x, *a, device=device, **k)


torch.as_tensor = _cpu_as_tensor

from common109a import HERE, W109, read, sha, write  # noqa: E402


def compare(a, b, path="", out=None, rel=1e-8, ab=1e-12):
    out = [] if out is None else out
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            out.append((path, "keys", sorted(set(a) ^ set(b)) if isinstance(a, dict) else type(a).__name__))
            return out
        for k in b:
            compare(a[k], b[k], f"{path}/{k}", out, rel, ab)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            out.append((path, "len", None))
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out, rel, ab)
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        if a != b:
            out.append((path, a, b))
    elif isinstance(b, (int, float)):
        if not isinstance(a, (int, float)) or isinstance(a, bool):
            out.append((path, a, b))
        elif not (abs(a - b) <= ab or abs(a - b) <= rel * abs(b)):
            out.append((path, a, b))
    else:
        out.append((path, "type", type(b).__name__))
    return out


def main(ev, out):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = ev / "run"
    man = read(run / "manifest.json")
    ident = {"run_manifest_sha256": sha(run / "manifest.json")}
    code_now = {p.name: sha(p) for p in sorted(W109.glob("*.py"))}
    rec = man["amendment2"]["code_sha256"]
    ident["code_equal_to_manifest"] = code_now == rec
    ident["code_mismatch"] = sorted(k for k in set(code_now) | set(rec) if code_now.get(k) != rec.get(k))
    ident["protocol_equal"] = sha(W109 / "protocol109.json") == man["protocol_sha256"]
    ident["table_sha_after"] = man["table_sha_after"]
    ident["table_sha_admissions"] = sorted({a["table_bytes_sha256"] for a in man["admission"]})
    ident["raw_files"] = len(man["files"])
    pub = ev / "metrics" / "metrics109.json"
    ident["published_metrics_sha256"] = sha(pub)
    committed = HERE.parent.parent / "evaluation/109/metrics109.json"
    if committed.exists():
        ident["committed_reviewer_copy_equal"] = sha(committed) == ident["published_metrics_sha256"]
    assert ident["code_equal_to_manifest"] and ident["protocol_equal"], ident
    assert ident["table_sha_admissions"] == [ident["table_sha_after"]], ident
    import analyze109                                                        # unchanged frozen analysis
    analyze109.main(run, ev / "dev/dev.json", ev / "synth/synth.json", out / "metrics", None)
    a, b = read(out / "metrics/metrics109.json"), read(pub)
    diffs = compare(a, b)

    def leaves(x, y, acc):
        if isinstance(y, dict):
            for k in y:
                leaves(x[k], y[k], acc)
        elif isinstance(y, list):
            for u, v in zip(x, y):
                leaves(u, v, acc)
        elif isinstance(y, (int, float)) and not isinstance(y, bool) and isinstance(x, (int, float)):
            acc.append(abs(x - y) / max(abs(y), 1e-300) if y != 0 else abs(x))
    acc = []
    leaves(a, b, acc)
    worst = max(acc) if acc else 0.0
    res = {"identity": ident, "n_numeric_leaves": len(acc), "max_relative_difference": worst, "n_mismatches": len(diffs),
           "mismatches_head": [list(map(str, d)) for d in diffs[:20]], "decision_equal": a["decision"] == b["decision"],
           "verdict_equal": a["verdict"] == b["verdict"], "verdict": a["verdict"], "device": "cpu",
           "torch": torch.__version__, "threads": torch.get_num_threads(), "elapsed_s": time.time() - t0,
           "repro_code_sha256": sha(Path(__file__)), "analyze109_sha256": sha(W109 / "analyze109.py")}
    res["pass"] = bool(res["n_mismatches"] == 0 and res["decision_equal"] and res["verdict_equal"])
    write(out / "stageA.json", res)
    print("STAGE A", "PASS" if res["pass"] else "FAIL", res["n_numeric_leaves"], worst, flush=True)
    sys.exit(0 if res["pass"] else 3)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
