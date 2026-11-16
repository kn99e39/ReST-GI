#!/usr/bin/env python3
"""WL97 MAP: static extraction of cross-batch dependencies from the experiment code.

For every audited experiment directory:
  * module edges   : `import X` where X is a module file of ANOTHER audited directory
  * artifact edges : os.path.join(<UPSTREAM_VAR>, 'results', ...) / "$UPSTREAM"/results/... where the variable names an
                     upstream run directory (R86, R92A, W94, RCK, ...), i.e. numbers / tensors / checkpoints consumed
  * seeds          : literal uses of the shared state realization seed 86001 and of xport87.realization_us / build_state
  * metric helper  : uses of an94.W / an94.ex / an94.eh and local re-definitions of the cos dOmega weight

Output: results/audit97/MAP/static_deps.json and a condensed stdout table. Read-only on the repository.
"""
import json, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP = os.path.join(REPO, "experiments")
DIRS = {
    "ecro86": "WL86", "ecro86a": "WL86A", "gm1_explicit87": "WL87", "gm1_relation88": "WL88", "gm1_relation89": "WL89",
    "gm1_gap90": "WL90", "gm1_c3urx1_91": "WL91", "ecro_attr92": "WL92", "ecro_own92a": "WL92A", "ecro_cf92b": "WL92B",
    "ecro_stage93": "WL93", "ecro_stage93a": "WL93A", "ecro_stage93b": "WL93B", "ecro_stage93c": "WL93C",
    "ecro_local93d": "WL93D", "ecro_local93d/addendum_checker": "WL93D-addendum", "ecro_mat94": "WL94", "ecro_mat94r": "WL94A",
    "ecro_exp94b": "WL94B", "ecro_q94c": "WL94C", "ecro_s94d": "WL94D", "ecro_attr95a": "WL95A", "ecro_util96a": "WL96A",
    "fnto_compiled83": "WL83(ER)", "fnto_compiled84": "WL84(ER)", "fnto_compiled82": "WL82", "fnto_compiled81": "WL81",
}
UPVAR = {"R86": "WL86", "R86A": "WL86A", "R87": "WL87", "R88": "WL88", "R89": "WL89", "R90": "WL90", "R92": "WL92",
         "R92A": "WL92A", "R92B": "WL92B", "R93": "WL93", "R93A": "WL93A", "R93B": "WL93B", "R93C": "WL93C", "R93D": "WL93D",
         "RCK": "WL93D-addendum", "W": "WL94", "W94": "WL94", "W94A": "WL94A", "W94B": "WL94B", "W94C": "WL94C",
         "WD": "WL94D", "W94D": "WL94D", "WC": "WL95(calc)", "R95A": "WL95A", "R91": "WL91"}


def files_of(d):
    p = os.path.join(EXP, d)
    out = []
    for f in sorted(os.listdir(p)):
        if f.endswith((".py", ".sh")) and os.path.isfile(os.path.join(p, f)):
            out.append(f)
    return out


def main():
    mod_home = {}
    for d in DIRS:
        if not os.path.isdir(os.path.join(EXP, d)):
            continue
        for f in files_of(d):
            if f.endswith(".py"):
                mod_home.setdefault(f[:-3], []).append(d)
    res = {}
    for d, wl in DIRS.items():
        if wl.startswith("WL8") and wl[2:4] in ("81", "82", "83", "84"):
            continue
        if not os.path.isdir(os.path.join(EXP, d)):
            continue
        mods, arts, seeds, metric = {}, {}, [], []
        for f in files_of(d):
            src = open(os.path.join(EXP, d, f), encoding="utf-8", errors="replace").read()
            for i, line in enumerate(src.splitlines(), 1):
                s = line.strip()
                if f.endswith(".py"):
                    m = re.match(r"(?:import|from)\s+([A-Za-z_0-9]+)", s)
                    names = []
                    if m:
                        names.append(m.group(1))
                    names += re.findall(r"__import__\('([A-Za-z_0-9]+)'\)", s)
                    names += re.findall(r"\bimport ([a-z_0-9]+) as [A-Za-z_0-9]+", s)
                    for n in set(names):
                        for home in mod_home.get(n, []):
                            if home != d:
                                mods.setdefault("%s/%s.py" % (home, n), []).append("%s:%d" % (f, i))
                    for v, rest in re.findall(r"os\.path\.join\(\s*([A-Z][A-Z0-9_]*)\s*,\s*([^)]*)\)", s):
                        if v in UPVAR:
                            key = UPVAR[v] + ":" + "/".join(x.strip().strip("'\"") for x in rest.split(","))
                            arts.setdefault(key, []).append("%s:%d" % (f, i))
                    for v, rest in re.findall(r"jl\(\s*f?['\"]\{([A-Z][A-Z0-9_]*)\}/([^'\"]*)['\"]", s):
                        if v in UPVAR:
                            arts.setdefault(UPVAR[v] + ":" + rest, []).append("%s:%d" % (f, i))
                else:
                    for v, rest in re.findall(r"\"?\$\{?([A-Z][A-Z0-9_]*)\}?\"?/([A-Za-z0-9_./?*{}$-]+)", s):
                        if v in UPVAR:
                            arts.setdefault(UPVAR[v] + ":" + rest, []).append("%s:%d" % (f, i))
                if re.search(r"\b86001\b|realization_us|build_state\(|table_seed", s):
                    seeds.append("%s:%d: %s" % (f, i, s[:150]))
                if re.search(r"AN\.W\b|AN\.ex\(|AN\.eh\(|an94|W94\s*=|\bWT\s*=|flux_w\(", s) and f.endswith(".py"):
                    metric.append("%s:%d: %s" % (f, i, s[:150]))
        res[wl] = {"dir": d, "module_deps": mods, "artifact_deps": arts, "seed_86001_and_state": seeds, "metric_helper": metric}
    out = os.path.join(REPO, "results", "audit97", "MAP", "static_deps.json")
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    for wl, r in res.items():
        print("== %s (%s)" % (wl, r["dir"]))
        print("   modules:", ", ".join("%s[%d]" % (k, len(v)) for k, v in sorted(r["module_deps"].items())))
        for k, v in sorted(r["artifact_deps"].items()):
            print("   artifact %-70s <- %s" % (k[:70], ",".join(v[:3])))
        print("   seed/state refs: %d ; metric-helper refs: %d" % (len(r["seed_86001_and_state"]), len(r["metric_helper"])))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
