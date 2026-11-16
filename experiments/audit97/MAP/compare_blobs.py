#!/usr/bin/env python3
"""WL97 audit, track MAP: compare deployed server code (git blob ids computed on
the server with `git hash-object`) against the repository tree at the stated
freeze / amendment commit of every evidence run.

Input : results/audit97/MAP/server_blobs/blobs_<rundir>.tsv
        (columns: blob sha1, ./path, mtime, size) -- produced on LabServer63.
Output: results/audit97/MAP/code_revision_check.json (+ stdout summary)

A git blob id is content-addressed over the raw bytes, so equality with the
repository blob (stored LF-normalised) proves byte identity with the LF
content of the committed file.
"""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BLOBS = os.path.join(REPO, "results", "audit97", "MAP", "server_blobs")
OUT = os.path.join(REPO, "results", "audit97", "MAP", "code_revision_check.json")

# run dir -> (worklog, own experiment dir, [candidate commits, last = stated executable revision])
RUNS = [
    ("rest_ecro86", "WL86", "ecro86", ["7786d09", "63972c8"]),
    ("rest_ecro86a", "WL86A", "ecro86a", ["46593f0", "1bd7df1", "f6483ab"]),
    ("rest_gm187", "WL87", "gm1_explicit87", ["60efecb"]),
    ("rest_rel88", "WL88", "gm1_relation88", ["26e13f2"]),
    ("rest_rel89", "WL89", "gm1_relation89", ["7515a29", "5d6bb5c"]),
    ("rest_gap90", "WL90", "gm1_gap90", ["51b93b8"]),
    ("rest_x191", "WL91", "gm1_c3urx1_91", ["7a8544e"]),
    ("rest_x92", "WL92", "ecro_attr92", ["5e672ff"]),
    ("rest_x92a", "WL92A", "ecro_own92a", ["7247d30", "c40226d"]),
    ("rest_x92b", "WL92B", "ecro_cf92b", ["b29d8ff"]),
    ("rest_x93", "WL93", "ecro_stage93", ["c30c797"]),
    ("rest_x93a", "WL93A", "ecro_stage93a", ["888ceb6"]),
    ("rest_x93b", "WL93B", "ecro_stage93b", ["03fbc84"]),
    ("rest_x93c", "WL93C", "ecro_stage93c", ["cab6e18"]),
    ("rest_x93d", "WL93D", "ecro_local93d", ["4f3fc56"]),
    ("rest_x93d_ck2", "WL93D-addendum", "ecro_local93d", ["d8d092f"]),
    ("rest_x94", "WL94", "ecro_mat94", ["e347904"]),
    ("rest_x94r", "WL94A", "ecro_mat94r", ["5b7c0c8"]),
    ("rest_x94b", "WL94B", "ecro_exp94b", ["5c0a6cd"]),
    ("rest_x94c", "WL94C", "ecro_q94c", ["6d8b254"]),
    ("rest_x94d", "WL94D", "ecro_s94d", ["a08c9bf"]),
    ("rest_x95a", "WL95A", "ecro_attr95a", ["c152120"]),
    ("rest_x96a", "WL96A", "ecro_util96a", ["a1718f7"]),
]


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def tree(commit):
    out = {}
    for line in git("ls-tree", "-r", commit).splitlines():
        meta, path = line.split("\t", 1)
        out[path] = meta.split()[2]
    return out


def commits_with_blob(blob, path):
    """commits (short) whose tree holds this exact blob at this path."""
    out = git("log", "--all", "--format=%h %cI", "--find-object=" + blob, "--", path)
    return [l for l in out.splitlines() if l.strip()]


def main():
    report = []
    for rundir, wl, own, commits in RUNS:
        fn = os.path.join(BLOBS, "blobs_%s.tsv" % rundir)
        server = {}
        for line in open(fn, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            server[p[1][2:]] = {"blob": p[0], "mtime": p[2], "size": int(p[3])}
        stated = commits[-1]
        t = tree(stated)
        same, diff, extra = [], [], []
        for path, s in sorted(server.items()):
            if path == "COMMIT":
                continue
            if path not in t:
                extra.append({"path": path, **s})
            elif t[path] == s["blob"]:
                same.append(path)
            else:
                d = {"path": path, **s, "repo_blob_at_stated": t[path],
                     "blob_found_in_commits": commits_with_blob(s["blob"], path)[:6]}
                diff.append(d)
        for e in extra:
            e["blob_found_in_commits"] = commits_with_blob(e["blob"], e["path"])[:6]
        # own-dir status
        own_prefix = "experiments/%s/" % own
        own_files = [p for p in server if p.startswith(own_prefix)]
        own_same = [p for p in same if p.startswith(own_prefix)]
        # files in the stated tree under experiments/ that the server lacks (own dir only)
        missing_own = [p for p in t if p.startswith(own_prefix) and p not in server]
        mt = sorted(v["mtime"] for k, v in server.items())
        rec = {
            "rundir": rundir, "worklog": wl, "own_dir": own,
            "candidate_commits": commits, "stated_executable_commit": stated,
            "n_server_files": len(server), "n_identical_to_stated": len(same),
            "n_different": len(diff), "n_not_in_stated_tree": len(extra),
            "own_dir_files_on_server": len(own_files), "own_dir_identical": len(own_same),
            "own_dir_missing_on_server": missing_own,
            "different": diff, "not_in_stated_tree": extra,
            "server_code_mtime_min": mt[0], "server_code_mtime_max": mt[-1],
        }
        report.append(rec)
        print("%-14s %-15s stated=%s files=%d same=%d diff=%d extra=%d own=%d/%d mtime[%s .. %s]" % (
            rundir, wl, stated, len(server), len(same), len(diff), len(extra),
            len(own_same), len(own_files), mt[0], mt[-1]))
        for d in diff:
            print("    DIFF  %s  server=%s (%s) repo=%s  server blob in: %s" % (
                d["path"], d["blob"][:10], d["mtime"], d["repo_blob_at_stated"][:10],
                "; ".join(d["blob_found_in_commits"]) or "NO COMMIT"))
        for e in extra:
            print("    EXTRA %s  server=%s (%s)  blob in: %s" % (
                e["path"], e["blob"][:10], e["mtime"],
                "; ".join(e["blob_found_in_commits"]) or "NO COMMIT"))
        if missing_own:
            print("    MISSING own-dir files on server:", missing_own)
    json.dump(report, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
