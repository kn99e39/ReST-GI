#!/usr/bin/env python3
"""WL97 MAP: for the two run dirs whose deployed code is not byte-identical to
the repository (rest_ecro86, rest_ecro86a) test whether the difference is only
CRLF line endings. Input norm_<rundir>.tsv = git blob id of each server file
after `tr -d '\r'`, number of CR-containing lines, number of lines."""
import json, os, subprocess
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
B = os.path.join(REPO, "results", "audit97", "MAP", "server_blobs")
def tree(c):
    o = {}
    for l in subprocess.run(["git", "-C", REPO, "ls-tree", "-r", c], capture_output=True, text=True).stdout.splitlines():
        m, p = l.split("\t", 1); o[p] = m.split()[2]
    return o
def found(blob, path):
    return subprocess.run(["git", "-C", REPO, "log", "--all", "--format=%h", "--find-object=" + blob, "--", path],
                          capture_output=True, text=True).stdout.split()
out = {}
for rundir, commits in (("rest_ecro86", ["7786d09", "63972c8"]), ("rest_ecro86a", ["46593f0", "1bd7df1", "f6483ab", "bd81442"])):
    trees = {c: tree(c) for c in commits}
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(B, "norm_%s.tsv" % rundir))]
    res = {"n": 0, "crlf_files": 0, "per_commit_identical_after_cr_strip": {c: 0 for c in commits}, "mismatch_vs_last": []}
    for h, p, ncr, nl in rows:
        p = p[2:]
        if p == "COMMIT":
            continue
        res["n"] += 1
        if int(ncr) > 0:
            res["crlf_files"] += 1
        for c in commits:
            if trees[c].get(p) == h:
                res["per_commit_identical_after_cr_strip"][c] += 1
        last = commits[-1]
        if trees[last].get(p) != h:
            res["mismatch_vs_last"].append({"path": p, "server_norm_blob": h, "repo_blob": trees[last].get(p), "norm_blob_in_commits": found(h, p)})
    out[rundir] = res
    print(rundir, json.dumps({k: v for k, v in res.items() if k != "mismatch_vs_last"}))
    for m in res["mismatch_vs_last"]:
        print("   MISMATCH", m)
json.dump(out, open(os.path.join(REPO, "results", "audit97", "MAP", "crlf_check.json"), "w", newline="\n"), indent=1)
