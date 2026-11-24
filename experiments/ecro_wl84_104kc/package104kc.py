"""WL104K-C reviewer package: evaluation/104K-C/manifest.json (sha256 of every raw result file), review copies of the small JSON
results, and index.html listing every figure with its caption (from figures_manifest.json). Reads results only.

  python package104kc.py <run_root_parent results/ecro_wl84_104kc> <out evaluation/104K-C>
"""
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

COPIES = {"primary/stageA/stageA.json": "stageA.json", "primary/contracts/prefit.json": "contracts_prefit.json",
          "primary/contracts/fitted.json": "contracts_fitted.json", "primary/contracts/leak.json": "contracts_leak.json",
          "primary/data/data.json": "data_test_c.json", "primary/fit_decision/fit_decision.json": "fit_decision.json",
          "primary/test_eval/test.json": "test_c_evaluation.json", "primary/decision.json": "decision.json",
          "primary/decision_first_pass_before_N4_recovery.json": "decision_first_pass_before_N4_recovery.json",
          "primary/scenes_new/manifest.json": "new_scene_references_manifest.json",
          "primary/posthoc/truth_constraint_attribution.json": "posthoc_truth_constraint_attribution.json",
          "primary/posthoc/scene_signed_cancellation.json": "posthoc_scene_signed_cancellation.json",
          "design/design_dev_CR_v1.json": "design_dev_CR_v1.json"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in raw.rglob("*") if p.is_file() and "gpu_duty" not in p.parts[len(raw.parts):][:1])
    raw_sha = {p.relative_to(raw).as_posix(): sha(p) for p in files}
    for p in sorted((raw / "gpu_duty").glob("*.csv")):
        raw_sha[p.relative_to(raw).as_posix()] = sha(p)
    for src in sorted((raw / "primary/scenes").glob("*/analysis104kc.json")):
        COPIES[src.relative_to(raw).as_posix()] = f"analysis_{src.parent.name}.json"
    copies = []
    for src, dst in COPIES.items():
        if (raw / src).exists():
            shutil.copyfile(raw / src, out / dst)
            copies.append({"file": dst, "from": src, "sha256": raw_sha[src]})
    figs = json.loads((out / "figures_manifest.json").read_text(encoding="utf-8"))
    man = {"batch": "WL104K-C", "stageA_freeze_commit": "ed69ee1", "main_freeze_commit": "02274e9",
           "eval_profile_storage_commit": "cab31e7", "figure_script_commit": "ae82533",
           "execution_recovery_commits": ["5f1d046", "b66d5f2", "8c7d0b2"], "raw_root": raw.as_posix(), "raw_sha256": raw_sha,
           "review_copies": copies, "figures": [{**f, "sha256": sha(out / f["file"])} for f in figs["figures"]],
           "note": ("Figures by experiments/ecro_wl84_104kc/figures104kc.py, package by package104kc.py, both from the raw files "
                    "hashed here. posthoc_truth_constraint_attribution.json is POST HOC and descriptive (after all labels). "
                    "Scene results are fixed-proposal replays (TOTAL = e1 + fixed e2 + fixed R0 HIGH), NOT end-to-end ECRO.")}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    rows = "\n".join(f'<figure><img src="{html.escape(f["file"])}" alt="{html.escape(f["file"])}">'
                     f'<figcaption><b>{html.escape(f["file"])}</b><br>{html.escape(f["caption"])}</figcaption></figure>'
                     for f in figs["figures"])
    links = " | ".join(f'<a href="{html.escape(c["file"])}">{html.escape(c["file"])}</a>' for c in copies)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WL104K-C figures</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#222;max-width:1400px}}
figure{{margin:0 0 32px 0}} img{{max-width:100%;height:auto;border:1px solid #ddd}} figcaption{{font-size:14px;margin-top:6px}}
</style></head><body><h1>WL104K-C: constrained fidelity fitting of ER-v1-SDM</h1>
<p>Worklog: docs/worklogs/104K-C_WORKLOG_CONSTRAINED_FIDELITY_FITTING_ER_V1_SDM.md. Scene panels are fixed-proposal replays, not end-to-end ECRO.</p>
<p>Result files: {links}</p>{rows}</body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print("package", len(raw_sha), "raw files,", len(copies), "copies,", len(figs["figures"]), "figures")


if __name__ == "__main__":
    main()
