# Track B: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List recent commits, worklogs, experiment and result directories] git log --oneline | head -80 && ls docs/worklogs | sed -n '80,200p' && ls docs/audits 2>/dev/null; ls experiments | tr '\n' ' '; echo; ls results | tr '\n' ' '
RUN [Inspect the WL86 experiment and result layout] ls -la experiments/ecro86 experiments/ecro86/protocol && wc -l experiments/ecro86/*.py && ls results/ecro86 | head -50
  -> <persisted-output>
Output too large (46.3KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\b9bh0d41v.txt

Preview (first 2KB):
efcf86a WL96A complete: LEARNED_RESOLVENT_UTILITY NO_MEANINGFUL_HEADROOM (CK2)
a1718f7 WL96A f ...[cut]... reeze: sharp-panel transport exposure smoke (protocol ecro_exp94b)
b34bdf9 WL94A (brief label WL94-R) complete: INCONCLUSIVE by the frozen mapping (seed disagreement)
5b7c0c8 WL94A (brief label WL94-R) freeze: material fidelity retention smoke (protocol ecro_mat94r)
f3b431f WL94 complete: H-M INCONCLUSIVE, H-A INCONCLUSIVE (INCONCLUSIVE_SMOKE_FIT) by the frozen mapping
e347904 WL94 freeze: ECRO material generalization vs sharp angular representation (protocol ecro_mat94)
16a8aa6 WL93D checkerboard addendum rev. 2: manifest hashes for sg JSONs and scene-iteration renders
...
</persisted-output>
  -> experiments/ecro86:
total 200
drwxr-xr-x 1 dna10 197609     0 Nov 11 11:23 .
drwxr-xr-x 1 dna10 197609     0 Nov 16 01:48 ..
drwxr-xr-x 1 dna10 197609     0 Nov 11 09:34 __pycache__
-rw-r--r-- 1 dna10 197609 15149 Nov 11 11:23 analyze86.py
drwxr-xr-x 1 dna10 197609     0 Nov 11 09:34 data
-rw-r--r-- ...[cut]... 11 10:27 ecro86.json
   248 experiments/ecro86/analyze86.py
    37 experiments/ecro86/diag_v3.py
   179 experiments/ecro86/econ86.py
   206 experiments/ecro86/ecro86.py
    62 experiments/ecro86/gen86.py
   151 experiments/ecro86/masks86.py
    34 experiments/ecro86/mats86.py
   137 experiments/ecro86/model86.py
   498 experiments/ecro86/pt86.py
   237 experiments/ecro86/scene86.py
   233 experiments/ecro86/tests86.py
   113 experiments/ecro86/train86.py
   187 experiments/ecro86/viz86.py
  2322 total
chain.status
chain.status.run1
chain_frozen_run1.status
code
results
watch.out
watch_run1.out
RUN [List recent commits, worklogs, experiment and result directories] cd /c/Projects/ReST-GI && git log --oneline | head -80 && ls docs/worklogs | sed -n '80,200p' && ls docs/audits 2>/dev/null; ls experiments | tr '\n' ' '; echo; ls results | tr '\n' ' '
RUN [Inspect the WL86 experiment and result layout] cd /c/Projects/ReST-GI && ls -la experiments/ecro86 experiments/ecro86/protocol && wc -l experiments/ecro86/*.py && ls results/ecro86 | head -50
RUN [List the ECRO-lineage worklogs, experiment and result directories] ls docs/worklogs | grep -E '^(8[6-9]|9[0-9])' ; ls experiments | grep -E 'ecro|gm1|audit' ; ls results | grep -E 'ecro|gm1|audit|wl9|arch'; ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Read the WL86 dataset generator and training script] cat experiments/ecro86/gen86.py && cat experiments/ecro86/train86.py
  -> """WL86 response supervision: random geometries of the family x random point lights -> geometry state, source encoding, target field.

  python gen86.py <root> <split> <start> <count>
Per geometry (file <root>/data/<split>/g<idx>.npz): geom / lights (json), patch tables, state rays (hit, dbin), and  ...[cut]... , val), 'train_subset': evaluate(model, items[:len(val)]), 'trainhq': evaluate(model, load(root, 'trainhq')) if split == 'train' else None,
           'val_no_global': evaluate(model, val, use_global=False), 'val_no_single': evaluate(model, val, use_single=False)}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    json.dump(res, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print('FINAL', json.dumps({k: res[k] for k in ('val', 'trainhq', 'train_subset', 'val_no_global', 'val_no_single', 'params', 'train_s')}), flush=True)


if __name__ == '__main__':
    main()
  -> Exit code 2
86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md
86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
88_WORKLOG_GM1_DIRECTIONAL_RELATION_FEASIBILITY.md
89_WORKLOG_GM1_SENDER_RELATION_QUALIFICATION_ ...[cut]... cro_attr95a
ecro_cf92b
ecro_exp94b
ecro_local93d
ecro_mat94
ecro_mat94r
ecro_own92a
ecro_q94c
ecro_s94d
ecro_stage93
ecro_stage93a
ecro_stage93b
ecro_stage93c
ecro_util96a
gm1_c3urx1_91
gm1_char85
gm1_explicit87
gm1_gap90
gm1_relation88
gm1_relation89
arch_study86b
ecro86
ecro86a
ecro_attr92
ecro_attr95a
ecro_cf92b
ecro_exp94b
ecro_local93d
ecro_local93d_checker
ecro_mat94
ecro_mat94r
ecro_own92a
ecro_q94c
ecro_s94d
ecro_stage93
ecro_stage93a
ecro_stage93b
ecro_stage93c
ecro_util96a
fnto_arch77a
gm1_c3urx1_91
gm1_char85
gm1_explicit87
gm1_gap90
gm1_relation88
gm1_relation89
wl95_calc
wl96_calc
RUN [List the ECRO-lineage worklogs, experiment and result directories] cd /c/Projects/ReST-GI && ls docs/worklogs | grep -E '^(8[6-9]|9[0-9])' ; ls experiments | grep -E 'ecro|gm1|audit' ; ls results | grep -E 'ecro|gm1|audit|wl9|arch'; ls docs/audits 2>/dev/null; ls experiments/audit97 results/audit97 2>/dev/null
RUN [Read the WL86 dataset generator and training script] cd /c/Projects/ReST-GI && cat experiments/ecro86/gen86.py && cat experiments/ecro86/train86.py
RUN [Read the WL86 scene family definition] cat experiments/ecro86/scene86.py
RUN [Print the WL86 frozen protocol and amendment] python -c "
import json
p=json.load(open('experiments/ecro86/protocol/ecro86.json'))
def show(d,ind=0,maxlen=600):
    for k,v in d.items():
        if isinstance(v,dict) and ind<2:
            print(' '*ind+k+':'); show(v,ind+2)
        else:
            s=json.dumps(v); print(' '*ind+k+': '+s[:maxlen])
show(p)
"; cat experiments/ecro86/protocol/amendment1.json
  -> """WL86 scene family, patch / direction-bin discretisation and sampling rules (numpy only; no renderer import).

Room x,y in [-2,2], z in [0,3], open towards -y. Stationary: floor (ER B), ceiling + 3 walls (Lambert), panel (ER S1), sphere (ER S2).
Movable: box A and box B (Lambert), each given by ce ...[cut]... ntinue
        return [{'pos': [float(v) for v in pos], 'rgb': [FX['light_intensity']] * 3}]


def light_arrays(lights):
    """-> pos [2,3], rgb [2,3], axis [2,3], cos_cut [2] (point light: cos_cut = -2; unused slot: rgb = 0)."""
    pos = np.zeros((2, 3)); rgb = np.zeros((2, 3)); axis = np.tile(Z, (2, 1)); cc = np.full(2, -2.0)
    for i, l in enumerate(lights):
        pos[i] = l['pos']; rgb[i] = l['rgb']
        if 'aim' in l:
            a = np.array(l['aim'], np.float64) - pos[i]; axis[i] = a / np.linalg.norm(a); cc[i] = math.cos(math.radians(l['cone_deg']))
    return pos, rgb, axis, cc
  -> protocol: "ecro86_v1"
status: "FROZEN by the freeze commit recorded in docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md (before any evidence run)"
hypothesis: "A compact, source-independent, geometry-conditioned learned linear operator can predict the higher-order directional transport generated b ...[cut]... in 1e-5 m AND identical geometry state'. The null-edit and move-and-return clauses keep the per-pixel 2e-4 tolerance. The per-pixel maximum is still reported.",
 "not_changed": "workload criteria and thresholds, masks, references (run 1 renders are kept and reused), representation, operator, training, gates of Stages 1-4",
 "procedure": "run-1 test results are preserved as results/tests/tests86_run1.json; V3 is re-run alone with the amended clause and merged; the chain resumes with RESUME=1 (completed jobs are not repeated; the qualification analysis is recomputed from the run-1 references)"
}
RUN [Read the WL86 scene family definition] cd /c/Projects/ReST-GI && cat experiments/ecro86/scene86.py
RUN [Print the WL86 frozen protocol and amendment] cd /c/Projects/ReST-GI && python -c "
import json
p=json.load(open('experiments/ecro86/protocol/ecro86.json'))
def show(d,ind=0,maxlen=600):
    for k,v in d.items():
        if isinstance(v,dict) and ind<2:
            print(' '*ind+k+':'); show(v,ind+2)
        else:
            s=json.dumps(v); print(' '*ind+k+': '+s[:maxlen])
show(p)
"; cat experiments/ecro86/protocol/amendment1.json
RUN [Size worklogs and list ECRO-lineage experiment files] wc -l docs/worklogs/{86,86A,87,88,89,90,91,92,92A,92B,93,93A,93B,93C,93D,94,94A,94B,94C,94D,95,95A,96,96A}_WORKLOG_*.md && for d in ecro86a ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a; do echo "== $d"; ls experiments/$d experiments/$d/protocol 2>/dev/null | tr '\n' ' '; echo; done
  -> 291 docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
   262 docs/worklogs/86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md
   373 docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
   427 docs/worklogs/88_WORKLOG_GM1_DIRECTIONAL_RELATION_FEASIBILITY.md
   471 docs/worklogs/89_WORKLOG_GM1_SEN ...[cut]... .py figb4_posthoc.py posthoc94b.py protocol run94b.sh  experiments/ecro_exp94b/protocol: ecro_exp94b.json 
== ecro_q94c
experiments/ecro_q94c: protocol q94.py run94c.sh  experiments/ecro_q94c/protocol: ecro_q94c.json 
== ecro_s94d
experiments/ecro_s94d: protocol run94d.sh s94.py  experiments/ecro_s94d/protocol: ecro_s94d.json 
== ecro_attr95a
experiments/ecro_attr95a: a95.py protocol run95a.sh wl95calc  experiments/ecro_attr95a/protocol: ecro_attr95a.json 
== ecro_util96a
experiments/ecro_util96a: figw3_posthoc.py protocol run96a.sh w96a.py  experiments/ecro_util96a/protocol: ecro_util96a.json
RUN [Size worklogs and list ECRO-lineage experiment files] cd /c/Projects/ReST-GI && wc -l docs/worklogs/{86,86A,87,88,89,90,91,92,92A,92B,93,93A,93B,93C,93D,94,94A,94B,94C,94D,95,95A,96,96A}_WORKLOG_*.md && for d in ecro86a ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_attr95a ecro_util96a; do echo "== $d"; ls experiments/$d experiments/$d/protocol 2>/dev/null | tr '\n' ' '; echo; done
RUN [Find CK2 definitions and exclusion-rule usages in later batches] grep -rn "CK2" experiments --include=*.py --include=*.json --include=*.sh -l | head -40; echo ---; grep -rn "exclusion\|near_test\|exclude" experiments/ecro* experiments/gm1_* --include=*.py --include=*.json | grep -v "^experiments/ecro86/" | cut -c1-260 | head -60
  -> experiments/ecro_attr95a/a95.py
experiments/ecro_attr95a/protocol/ecro_attr95a.json
experiments/ecro_exp94b/b94.py
experiments/ecro_exp94b/figb4_posthoc.py
experiments/ecro_exp94b/posthoc94b.py
experiments/ecro_exp94b/protocol/ecro_exp94b.json
experiments/ecro_exp94b/run94b.sh
experiments/ecro_local ...[cut]...  signed error >= 0.5 (same sign) and the realization-mean 95 % interval excludes 0 (region C, per edit and T0 / T1)",
experiments/gm1_explicit87/xport87.py:8:receive 3..6 scattering events from it; LOW (e1 = q, e2 = T q at the camera vertices) is unchanged and B excludes q (no double count).
experiments/gm1_gap90/econ90.py:3:Warm, synchronised wall clock; every timed operation is executed once untimed first (compilation excluded); medians over repeats and sources.
experiments/gm1_relation88/econ88.py:7:first (kernel compilation excluded from warm numbers; cold numbers are reported separately).
RUN [Find CK2 definitions and exclusion-rule usages in later batches] cd /c/Projects/ReST-GI && grep -rn "CK2" experiments --include=*.py --include=*.json --include=*.sh -l | head -40; echo ---; grep -rn "exclusion\|near_test\|exclude" experiments/ecro* experiments/gm1_* --include=*.py --include=*.json | grep -v "^experiments/ecro86/" | cut -c1-260 | head -60
RUN [Read the CK2 scene definition and the WL94 selection code] cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -80; sed -n 1,110p experiments/ecro_mat94/c94.py
  -> {
 "tag": "CK2",
 "targets": [
  "A"
 ],
 "light_set": "LCK2",
 "camera": {
  "origin": [
   0.3,
   -1.6,
   1.3
  ],
  "target": [
   -1.3,
   0.45,
   1.0
  ],
  "fov_deg": 32.0,
  "res": 512
 },
 "geom": {
  "A": {
   "c": [
    -1.08,
    0.6,
    1.0
   ],
   "h": [
    0.2,
    0.45,
    0.45 ...[cut]... , start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = sel_files(R, split)[int(start):int(start) + int(count)]
    out = os.path.join(R, 'data', f'{split}_{assign}'); os.makedirs(out, exist_ok=True)
    ctx = None; t0 = time.time(); rows = []
    for rec in files:
        fn = os.path.join(out, rec['file'])
        if os.path.exists(fn):
            continue
        z = np.load(os.path.join(R86, 'data', split, rec['file']))
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights']))
        if ctx is None:
            ctx = PT.Ctx(geom)
RUN [Read the CK2 scene definition and the WL94 selection code] cd /c/Projects/ReST-GI && cat experiments/ecro_local93d/addendum_checker/scene_ck2.json | head -80; sed -n 1,110p experiments/ecro_mat94/c94.py
RUN [Check server reachability and the WL86 run directory] ssh -o ConnectTimeout=15 LabServer63 'hostname; ls -d ~/rest_* | tr "\n" " "; echo; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; ls ~/rest_ecro86; ls ~/rest_ecro86/data; ls ~/rest_ecro86/data/train | wc -l; ls ~/rest_ecro86/results/models 2>/dev/null; ls ~/rest_x97 2>/dev/null'
  -> 63
/home/nd/rest_8dna_baseline /home/nd/rest_arch77a /home/nd/rest_ato74 /home/nd/rest_ato74_design /home/nd/rest_ato74_dev /home/nd/rest_attrib71b /home/nd/rest_b4rep /home/nd/rest_closure80d /home/nd/rest_cmp72 /home/nd/rest_cmp72_dev /home/nd/rest_compiled80c /home/nd/rest_compiled81 /home/nd/res ...[cut]... ome/nd/rest_x94cdev /home/nd/rest_x94cfeas /home/nd/rest_x94d /home/nd/rest_x94ddev /home/nd/rest_x94dev /home/nd/rest_x94dfeas /home/nd/rest_x94r /home/nd/rest_x94rdev /home/nd/rest_x95a /home/nd/rest_x95adev /home/nd/rest_x96a /home/nd/rest_x96adev /home/nd/rest_x96aprobe /home/nd/rest_x97 
1 MiB, 12288 MiB, 0 %
chain.out
chain.status
chain.status.run1
chain2.out
chain_frozen_run1.status
chain_resume.out
code
data
diag_v3.py
env.sh
gpu_util_trace.txt
gpu_util_trace_run1.txt
results
watch.out
watch_run1.out
train
trainhq
val
1024
ecro_s8601.json
ecro_s8601.pt
ecro_s8602.json
ecro_s8602.pt
MAP
RUN [Read the WL92A exclusion code and locate E4-E6 definitions] sed -n 20,75p experiments/ecro_own92a/own92a.py; grep -n "E4\|E5\|E6" experiments/ecro_attr92/fx92.py | head -30; ls results/ecro_own92a results/ecro_own92a/results 2>/dev/null | head -60
  -> P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_own92a.json')))
FROZEN = ['ecro_s8601', 'ecro_s8602']


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def models(R):
    out = [('frz', n) for n in FROZEN]
    for arm in ('sum', 'own'): ...[cut]... MIX4 / LSPOT4).
12:import fx87  # noqa: E402
24:    """Make E4 addressable. name: candidate id; default: the recorded choice in <root>/results/e4_choice.json."""
33:    FX['E4'] = {'T0': c['T0'], 'T1': c['T1'], 'moved': c['moved'], 'light_set': 'L4', 'role': 'WL92 held-out condition (WL87 candidate ' + name + ')'}
results/ecro_own92a:
chain.status
chain_run1.status
data
gpu_trace.csv
gpu_trace_run1.csv
results

results/ecro_own92a/results:
e4_admission.json
e4_choice.json
exclusion.json
fieldeval.json
final92a.json
lists
local92a.json
logs
models
tests_f2.json
tests_pre.json
timing92a.json
viz
RUN [Read the WL92A exclusion code and locate E4-E6 definitions] cd /c/Projects/ReST-GI && sed -n 20,75p experiments/ecro_own92a/own92a.py; grep -n "E4\|E5\|E6" experiments/ecro_attr92/fx92.py | head -30; ls results/ecro_own92a results/ecro_own92a/results 2>/dev/null | head -60
RUN [Regenerate the WL86 population locally from the frozen sampler] mkdir -p experiments/audit97/B results/audit97/B docs/audits/wl97_evidence_reliability && cat > experiments/audit97/B/regen_population.py <<'EOF'
"""WL97 audit track B: independent regeneration of the WL86 training / validation population (geometries + lights) from the
frozen sampling rule (experiments/ecro86/scene86.py, numpy only) and distance checks against every later 'held-out' condition.

  python regen_population.py <out_json>
No renderer, no GPU. Imports scene86 read-only (PYTHONDONTWRIT
  -> train 1024 1d33827645edbe3a
val 64 c45e44babacc5eda
trainhq 32 e928f3e3e47b2cb8
dev 48 0f31c20da85f91f9
RUN [Regenerate the WL86 population locally from the frozen sampler] cd /c/Projects/ReST-GI && mkdir -p experiments/audit97/B results/audit97/B docs/audits/wl97_evidence_reliability && cat > experiments/audit97/B/regen_population.py <<'EOF'
"""WL97 audit track B: independent regeneration of the WL86 training / validation population (geometries + lights) from the
frozen sampling rule (experiments/ecro86/scene86.py, numpy only) and distance checks against every later 'held-out' condition.

  python regen_population.py <out_json>
No renderer, no GPU. Imports scene86
RUN [Digest the stored WL86 population on the server for comparison] cat > experiments/audit97/B/server_digest_population.py <<'EOF'
"""WL97 audit track B (server side, read-only): digest of the geom / lights JSON actually stored in the WL86 data files, plus file
sha256 prefixes, to compare with the local regeneration (regen_population.py).

  python server_digest_population.py <R86> <out_json>
"""
import glob, hashlib, json, os, sys
import numpy as np

R86, outfn = sys.argv[1], sys.argv[2]
out = {}
for sp in ('train', 'val', 'trainhq'):
    fns = sorted(glob.glo
  -> experiments/audit97/B/server_digest_population.py: Python script, ASCII text executable
total 28
drwxr-xr-x   5 nd nd  4096 Nov 15 17:46 .
drwx------ 172 nd nd 12288 Nov 15 17:45 ..
drwxr-xr-x   2 nd nd  4096 Nov 15 17:46 B
drwxr-xr-x   2 nd nd  4096 Nov 15 17:46 C1
drwxr-xr-x   2 nd nd  4096 Nov 15 17:46 MAP
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
server_digest_population.py: Python script, ASCII text executable
train 1024 1d33827645edbe3a
val 64 c45e44babacc5eda
trainhq 32 e928f3e3e47b2cb8
RUN [Digest the stored WL86 population on the server for comparison] cd /c/Projects/ReST-GI && cat > experiments/audit97/B/server_digest_population.py <<'EOF'
"""WL97 audit track B (server side, read-only): digest of the geom / lights JSON actually stored in the WL86 data files, plus file
sha256 prefixes, to compare with the local regeneration (regen_population.py).

  python server_digest_population.py <R86> <out_json>
"""
import glob, hashlib, json, os, sys
import numpy as np

R86, outfn = sys.argv[1], sys.argv[2]
out = {}
for sp in ('train', 'val', 'trainhq'):
RUN [Read the first part of the WL92A worklog] sed -n 1,120p docs/worklogs/92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md | cut -c1-1200
  -> # Worklog — ECRO Physical Component Supervision: Controlled Generator Learning and Held-out Sharp Interreflection (WL92A) (2026-11-13)

## Origin and scope

- **Brief.** "ReST-GI — WL93 ECRO Physical Component Supervision: Controlled Generator Learning and Held-out Sharp Interreflection" (user, 2026 ...[cut]... -0 criteria, unchanged thresholds)

| Criterion | Value | Result |
|---|---|---|
| Q1 direct change in A | 0.639 | pass |
| Q2 indirect change, stationary | 0.0086 | pass |
| Q3 region C change; sharp / moderate share | 0.363; 0.96 (5,937 px) | pass |
| **Q4 broad multi-bounce change in D** | change 0.0101, higher-order part 0.0047 against 4 × floor 0.0048 (SNR 3.9) | **fail** |
| Q6 higher-order share | 0.229 | pass |
| Q7 reference uncertainty | 0.0037 | pass |

- The sharp-reader criteria hold; the failing clause is the resolvability of the broad higher-order change (0.0047 against 0.0048).
RUN [Read the first part of the WL92A worklog] cd /c/Projects/ReST-GI && sed -n 1,120p docs/worklogs/92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md | cut -c1-1200
RUN [Inspect the WL92A exclusion record and later result folders] python - <<'EOF'
import json
e=json.load(open('results/ecro_own92a/results/exclusion.json'))
print(e['rule']); print(e['E4_light_positions']); print({k:len(v) for k,v in e['excluded'].items()}, e['kept'], e['min_distance_kept'])
c=json.load(open('results/ecro_own92a/results/e4_choice.json'))
print(json.dumps(c)[:1500])
EOF
ls results/ecro_cf92b/results | head -30; ls results/ecro_stage93/results | head -30; ls results/ecro_mat94/results | head -40
  -> a file is removed if a box centre is within 0.15 m of the same box of E4 T0 or T1, or any of its lights is within 0.3 m of any E4 light (L4, LMIX4, LSPOT4) - the WL86 exclusion rule applied to E4; geometry / light positions only
[[-0.2, -0.6, 2.7], [1.1, 0.9, 2.4], [-1.2, -1.0, 1.9], [0.7, -1.2, 2. ...[cut]... 075235], ["g00284.npz", 0.06058442714312393], ["g00328.npz", 0.14927276190324962], ["g00845.npz", 0.12414282047203515], ["g00880.npz", 0.08983199301571271]], "val": [["g00057.npz", 0.11234872058528668]], "trainhq": [["g00012.npz", 0.12283409744320616]]}, "note": "geometry / visibility only; no radiance of any E4 candidate exists"}
e5_admission_k995.json
e5_candidates.json
final92b.json
lists
local92b.json
logs
repro.json
tables
tests92b.json
viz
ls: cannot access 'results/ecro_stage93/results': No such file or directory
ls: cannot access 'results/ecro_mat94/results': No such file or directory
RUN [Inspect the WL92A exclusion record and later result folders] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
e=json.load(open('results/ecro_own92a/results/exclusion.json'))
print(e['rule']); print(e['E4_light_positions']); print({k:len(v) for k,v in e['excluded'].items()}, e['kept'], e['min_distance_kept'])
c=json.load(open('results/ecro_own92a/results/e4_choice.json'))
print(json.dumps(c)[:1500])
EOF
ls results/ecro_cf92b/results | head -30; ls results/ecro_stage93/results | head -30; ls results/ecro_mat94/results | head -40
RUN [Find E3, E5 definitions] ls results/ecro_stage93 results/ecro_mat94 results/ecro_cf92b/results/lists | head -60; python - <<'EOF'
import json
c=json.load(open('results/ecro_cf92b/results/e5_candidates.json'))
print(list(c.keys()))
for k,v in c.items():
    s=json.dumps(v); print(k, s[:700])
EOF
grep -n "E3\b\|E3b\|E3c\|\"E3" experiments/gm1_explicit87/fx87.py | head -20
  -> results/ecro_cf92b/results/lists:
cf_E1_T1_LSPOT.txt
cf_E2_T0.txt
cf_E2_T1.txt
cf_E4_T0.txt
cf_E4_T1.txt
e5_E5k995_T0.txt
e5_E5k995_T1.txt
e5_E5k995_T1_LMIX.txt
e5_E5k995_T1_LSPOT.txt

results/ecro_mat94:
chain.status
final.json
gpu.txt
models
pop
posthoc94.json
pre.json
reused_refs.txt
select94.jso ...[cut]... no radiance of any candidate exists at this point"
1:"""WL87 configuration: imports the frozen WL86 transport module unchanged and adds the prospectively frozen untouched condition E3.
3:E3 is chosen from protocol['e3_candidates'] by a geometry / visibility-only rule (mask pixel counts, no radiance) and
4:recorded in results/e3_choice.json (run87.py e3select) before any E3 radiance exists. inject() puts it into the shared WL86 fixture dictionary so that the
27:    """Make E3 addressable. name: candidate id; default: the recorded choice in <root>/results/e3_choice.json."""
36:    FX['E3'] = ex
RUN [Find E3, E5 definitions] cd /c/Projects/ReST-GI && ls results/ecro_stage93 results/ecro_mat94 results/ecro_cf92b/results/lists | head -60; python - <<'EOF'
import json
c=json.load(open('results/ecro_cf92b/results/e5_candidates.json'))
print(list(c.keys()))
for k,v in c.items():
    s=json.dumps(v); print(k, s[:700])
EOF
grep -n "E3\b\|E3b\|E3c\|\"E3" experiments/gm1_explicit87/fx87.py | head -20
RUN [Read E3 candidates and E6 rule] cat experiments/gm1_explicit87/fx87.py | sed -n 1,60p; python - <<'EOF'
import json
p=json.load(open('experiments/gm1_explicit87/protocol/gm1_explicit87.json'))
print(json.dumps(p.get('e3_candidates'))[:2500])
p=json.load(open('experiments/ecro_stage93/protocol/ecro_stage93.json'))
print(json.dumps(p['e6'])[:1500])
EOF
ls results/gm1_explicit87/ results/gm1_explicit87/results 2>/dev/null | head -30
  -> """WL87 configuration: imports the frozen WL86 transport module unchanged and adds the prospectively frozen untouched condition E3.

E3 is chosen from protocol['e3_candidates'] by a geometry / visibility-only rule (mask pixel counts, no radiance) and
recorded in results/e3_choice.json (run87.py e3se ...[cut]... p": 8192, "seed_base": 93730, "per_candidate_offset": 100}, "sources": {"replicas": 4, "spp": 8192, "seed_base": 93750}, "oracle_field": {"spp": 4096, "seed": 93460}, "split": "diag92 split (2 x 2,048 spp, seeds 92400 / 92405)"}, "order": "qualified BEFORE training; if no candidate is admitted: STOPPED_WORKLOAD_UNQUALIFIED, no training, no held-out claim"}
results/gm1_explicit87/:
chain.status
code
gpu_util_trace.txt
results

results/gm1_explicit87/results:
e3_admission.json
e3_choice.json
econ
final87.json
lists
logs
masks
renders
stage1_87.json
stage2_87.json
stage3_87.json
tables
tests
viz
RUN [Read E3 candidates and E6 rule] cd /c/Projects/ReST-GI && cat experiments/gm1_explicit87/fx87.py | sed -n 1,60p; python - <<'EOF'
import json
p=json.load(open('experiments/gm1_explicit87/protocol/gm1_explicit87.json'))
print(json.dumps(p.get('e3_candidates'))[:2500])
p=json.load(open('experiments/ecro_stage93/protocol/ecro_stage93.json'))
print(json.dumps(p['e6'])[:1500])
EOF
ls results/gm1_explicit87/ results/gm1_explicit87/results 2>/dev/null | head -30
RUN [Compute held-out disjointness of each condition against each training population] cat > experiments/audit97/B/heldout_disjointness.py <<'EOF'
"""WL97 audit track B: disjointness of every later 'held-out' condition from the populations that actually trained each checkpoint.

  python heldout_disjointness.py <population_regen.json> <out_json>
Populations: P86 = all 1,024 WL86 train files (frozen ecro_s8601/8602); P92A = P86 minus results/ecro_own92a/results/exclusion.json
(sum_s*/own_s* of WL92A; init of every WL93A-C / WL94* fine-tune); P94 = the 96 files of results/ecro_mat94
  -> E1 {'P86_all1024': (0, 0.154, 0.3, 0.409), 'P92A_928': (0, 0.154, 0.301, 0.409), 'P94_96': (0, 0.201, 0.303, 0.496)}
E2 {'P86_all1024': (0, 0.155, 0.303, 0.32), 'P92A_928': (0, 0.155, 0.309, 0.32), 'P94_96': (0, 0.155, 0.362, 0.434)}
E3a {'P86_all1024': (25, 0.129, 0.089, 0.363), 'P92A_928': (22,  ...[cut]... .2056
],
[
"g00549.npz",
0.1793
],
[
"g00580.npz",
0.2545
],
[
"g00756.npz",
0.2708
],
[
"g00797.npz",
0.191
],
[
"g00956.npz",
0.1852
],
[
"g01017.npz",
0.1454
]
],
"n_files_violating": 17,
"min_box_m": 0.09490092648078466,
"min_light_m": 0.10545299193893418,
"min_joint_AB_m (max over boxes, nearest geometry)": 0.9373641486381705
},
"P94_96": {
"box_violations": [],
"light_violations": [],
"n_files_violating": 0,
"min_box_m": 0.5338628751322028,
"min_light_m": 0.338240993692988,
"min_joint_AB_m (max over boxes, nearest geometry)": 1.4714452911363143
}
}
RUN [Compute held-out disjointness of each condition against each training population] cd /c/Projects/ReST-GI && cat > experiments/audit97/B/heldout_disjointness.py <<'EOF'
"""WL97 audit track B: disjointness of every later 'held-out' condition from the populations that actually trained each checkpoint.

  python heldout_disjointness.py <population_regen.json> <out_json>
Populations: P86 = all 1,024 WL86 train files (frozen ecro_s8601/8602); P92A = P86 minus results/ecro_own92a/results/exclusion.json
(sum_s*/own_s* of WL92A; init of every WL93A-C / WL94* fine-tune); P94 = the 96 f
RUN [Read the WL94B worklog] cut -c1-1500 docs/worklogs/94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
  -> # Worklog — WL94B: ECRO Sharp-Panel Transport Exposure Smoke (2026-11-14)

## Origin and scope

- **Brief.** "WL94B — Sharp-Panel Transport Preservation Smoke" (user, 2026-11-14).
- **Batch class.** This is a suffix of WL94. It is a dependent evidence-gap follow-up of WL94 / WL94A under the preserve ...[cut]... son`: the replayed 2,000-step sequence of D (never trained), plus C's realized counts.
- **Training.** None. No steps were run and no queries were spent on D.
- **Validation inputs.** Unchanged and unused.
- **Cost.** 2.7 min wall in total:
  - audit renders 8 jobs, ~2.2 min, pool 6;
  - T_REPRO tables 4 jobs;
  - GPU mean 50 % over 8 samples.
- **Outputs.**
  - `results/ecro_exp94b/`: audit.json, alloc.json, schedules, repro.json, final_b.json, posthoc94b.json, chain.status, gpu.txt.
  - `results/evaluation/94B/`: figB0_audit and figB4_posthoc_face_coverage (post hoc), with a sha256 manifest.
RUN [Read the WL94B worklog] cd /c/Projects/ReST-GI && cut -c1-1500 docs/worklogs/94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
RUN [Find the source of the ring-0 share claim] grep -rn "ring-0\|ring 0\|ring0" docs/worklogs/9*.md docs/plans 2>/dev/null | cut -c1-420 | head -30
  -> docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:118:- **Energy and peak at the sender.** At CK2's grazing (ring-0) incidence, every material-level rank-16 operator loses the S1 lobe almost completely: peak 0.018 (NMF16) and 0.002 (L1-NMF16), with energy 0.46 and 0.1 ...[cut]... at sends flux from all lit panel patches into steeper directions that do hit box A. This mechanism is consistent with over-delivery (> 1) and with misplacement (negative correlation).
docs/plans/wl95_sharp_transport_design/auditA.md:150:   [INF] It is neutral for Lambert outgoing radiance. It is non-neutral for the S1 panel sender metrics: the CK2 lobe lies in ring 0, where the weights are ×0.5 or ×1.5 depending on the azimuth half. So the WL94D CK2 energy_ratio of 1.71–1.75 carries an unquantified weighting bias. `peak_ratio` has no W and is unaffected. Note that `s94.py:179, 288` uses the co
RUN [Find the source of the ring-0 share claim] cd /c/Projects/ReST-GI && grep -rn "ring-0\|ring 0\|ring0" docs/worklogs/9*.md docs/plans 2>/dev/null | cut -c1-420 | head -30
RUN [Read the WL95A worklog] cut -c1-1800 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
  -> # Worklog — WL95A: Sharp Sender Representability & Flight Attribution Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL95A — Sharp Sender Representability & Flight Attribution Smoke" (user). It is a suffix of WL95 and a bounded, learning-free attribution batch.
- **Proposition tested.** "The ...[cut]... d ~/wl95_calc ~/rest_x92a`. The chain runs `a95.py stage0 | main | an` and `mat94.py render`. Extension to 64 draws re-runs `main` with `A95_EXTEND=1`, triggered by the predeclared rule after the 16-draw pilot.
- **Outputs.**
  - `results/ecro_attr95a/`: s0.json, main.json, an.json, deliv.npz (per-patch exitance per arm, deployed and per draw), ops95a.npz, chain.status, gpu.txt.
  - `results/evaluation/95A/`, with a sha256 manifest:
    - figA1 senders;
    - figA2 delivery;
    - figA3 box-A footprint maps (shared scale);
    - figA4 ROI renders and signed errors (shared scale, one exposure).
RUN [Read the WL95A worklog] cd /c/Projects/ReST-GI && cut -c1-1800 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
RUN [Print the WL95A protocol] python - <<'EOF'
import json
p=json.load(open('experiments/ecro_attr95a/protocol/ecro_attr95a.json'))
def show(d,ind=0,maxlen=1500):
    for k,v in d.items():
        if isinstance(v,dict) and ind<4:
            print(' '*ind+k+':'); show(v,ind+2)
        else:
            s=json.dumps(v); print(' '*ind+k+': '+s[:maxlen])
show(p)
EOF
  -> protocol: "ecro_attr95a"
version: 1
title: "WL95A - sharp sender representability and flight attribution smoke (learning-free; CK2 none)"
brief: "user brief 'WL95A - Sharp Sender Representability & Flight Attribution Smoke' (2026-11-15)"
batch_class: "suffix of WL95 (dependent evidence-gap follo ...[cut]... normalising operators", "rank / bin increases as repair", "learned global residuals", "e2 / e3 rebalancing", "material / light / geometry changes", "promoting G1 / G2 / EXP / C3UR", "successor batch"]
pre_freeze_exposure: "All WL92-WL95 results are known. Dev run ~/rest_x95adev (D = 2 / 4 draws, q Mq = 64, 32-spp renders): Stage 0 test flags were read (all PASS; first attempt crashed on a JSON print of numpy bools after writing s0.json, fixed); the chain exercised main, renders, analysis and the rule-based extension path ('AN EXTEND' printed). No dev metric value, table or figure was opened."
RUN [Print the WL95A protocol] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
p=json.load(open('experiments/ecro_attr95a/protocol/ecro_attr95a.json'))
def show(d,ind=0,maxlen=1500):
    for k,v in d.items():
        if isinstance(v,dict) and ind<4:
            print(' '*ind+k+':'); show(v,ind+2)
        else:
            s=json.dumps(v); print(' '*ind+k+': '+s[:maxlen])
show(p)
EOF
RUN [Read the WL96A worklog] cut -c1-1600 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | sed -n 1,200p
  -> # Worklog — WL96A: Learned Higher-Order Transport Utility Ceiling (2026-11-16)

## Origin and scope

- **Brief.** "WL96A — Learned Higher-Order Transport Utility Ceiling" (user). A suffix of WL96: the dependent, learning-free feasibility test of the one learned element WL96 left standing (a broad ≥  ...[cut]... pe.** One workload: one grazing point light, Lambert box A as the ROI reader, a fixed panel. Other lights were not referenced.
- **Prefix.** The prefix is the path-traced oracle, which isolates the tail question by design. A deployable sharp prefix carrier adds its own error upstream; that error belongs to the prefix (WL95A / WL96), not to the resolvent.
- **Untagged predicate.** "Panel at s ≥ 2" sits inside e3_o and cannot be separated with the existing tag. The resolvent domain therefore includes some sharp structure in the middle of the chain. The explicit tail handles it within noise here.
RUN [Read the WL96A worklog] cd /c/Projects/ReST-GI && cut -c1-1600 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | sed -n 1,200p
RUN [Read WL96A remainder and protocol threshold provenance] cut -c1-1600 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | sed -n 200,263p; python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in p:
    s=json.dumps(p[k])
    if any(w in s for w in ('M_img','exposure','dev','freeze','threshold','0.005')) or k in ('pre_freeze_exposure','decisions','stage_1','stops'):
        print(k, ':', s[:3000]); print()
EOF
  -> - **Untagged predicate.** "Panel at s ≥ 2" sits inside e3_o and cannot be separated with the existing tag. The resolvent domain therefore includes some sharp structure in the middle of the chain. The explicit tail handles it within noise here.
- **Sharp-reader tables.** The panel's own e3_o tables h ...[cut]... itatively that |arm - F| in the ROI is visibly smallest for XOD / XOP among the tail arms and larger for the learned G arms and for Z (no number was read). (2) full-scale timing probe ~/rest_x96aprobe (stage 0 at full spp: all PASS incl. T_READOUT; tails 26 s; one render ~10 s; one extension half 18 s; Stage 2 two lights 21 s, two fits 178 s at 98 % GPU; lifecycle 18 s, boundary_ok); only walls, GPU utilisation and pass flags were read, no metric or timing value of an arm. No criterion was changed after either run; the only post-dev changes are the JSON-bool / import fixes and the pool sizes."
RUN [Read WL96A remainder and protocol threshold provenance] cd /c/Projects/ReST-GI && cut -c1-1600 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | sed -n 200,263p; python - <<'EOF'
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in p:
    s=json.dumps(p[k])
    if any(w in s for w in ('M_img','exposure','dev','freeze','threshold','0.005')) or k in ('pre_freeze_exposure','decisions','stage_1','stops'):
        print(k, ':', s[:3000]); print()
EOF
RUN [Grep WL92, WL92B, WL93 worklogs for amendments and deviations] for f in 92_WORKLOG 92B_WORKLOG 93_WORKLOG; do echo "=================== $f"; grep -n -i "amend\|post hoc\|post-hoc\|deviat\|disclos\|pre-freeze\|revis\|exposure\|extension\|wording\|re-interpret\|not converge\|seen before\|dev " docs/worklogs/${f}_*.md | cut -c1-700; done
  -> =================== 92_WORKLOG
88:## Pre-freeze development (non-evidence)
90:- **Design refinement from the local-operator dev run.** The plain Frobenius norm of M64 is dominated by the grazing ring (f ∝ 1 / (4 cos cos)): a rank-16 fit in that norm spends every channel on 16 grazing incident bins.  ...[cut]... ot evidence and are not exported.
234:- **Code and protocol.** Frozen commit `c30c797`. Amendment 1 and the post hoc diagnostic / visual (`posthoc93.py`, `vizgate93.py`) are in the closing commit.
238:- **Seeds.** q seed 93000 + 4·idx + lighting; post hoc second realisation + 500,000.
239:- **Compute.** About 1 GPU-minute of evidence (gate) plus about 5 minutes post hoc. The dev chain and probes (`~/rest_x93dev`, `~/rest_x93probe`) are not evidence.
247:- 동결 전 dev 체인에서 결함 두 개를 발견해 고쳤다.
255:- 정지 경로에서 분석 스크립트가 크래시했다. Amendment 1(비과학적 실행 수정)로 고쳤고, 소유자 승인을 기다린다.
285:- Amendment 1 승인 여부는 소유자가 결정한다.
RUN [Grep WL92, WL92B, WL93 worklogs for amendments and deviations] cd /c/Projects/ReST-GI && for f in 92_WORKLOG 92B_WORKLOG 93_WORKLOG; do echo "=================== $f"; grep -n -i "amend\|post hoc\|post-hoc\|deviat\|disclos\|pre-freeze\|revis\|exposure\|extension\|wording\|re-interpret\|not converge\|seen before\|dev " docs/worklogs/${f}_*.md | cut -c1-700; done
RUN [Grep WL93A-D worklogs for amendments, deviations and checkpoint identity] for f in 93A_WORKLOG 93B_WORKLOG 93C_WORKLOG 93D_WORKLOG_ECRO 93D_WORKLOG_ADDENDUM; do echo "=================== $f"; grep -n -i "amend\|post hoc\|post-hoc\|deviat\|disclos\|pre-freeze\|revis\|exposure\|extension\|wording\|re-interpret\|seen before\|freeze\|own_s\|checkpoint\|sha256\|CK2\|held-out\|heldout" docs/worklogs/${f}*.md | cut -c1-520; done
  -> =================== 93A_WORKLOG
20:- WL93 Amendment 1 (stop-path analysis crash fix) keeps its provenance and its **pending-approval** status.
26:- Unchanged: ECRO generator; frozen ER materials and path semantics; the deployed 16-sample source encoding; local rank 16; widths; the geometry interface ...[cut]... us`.
173:- **Images and data** (`results/evaluation/93D/addendum_checker/`; sha256 in `manifest_checker93d.json`):
178:| `figCK1_{bx38,bx75}_own_s{8601,8602}_sheet.png` | comparison sheets, columns REF / C0 / C2_src / C2_rcv / C2_both. Rows: full frame (reference LOW + arm higher-order), higher-order only, reflection close-up, signed error vs the reference (full and close-up), texture response |
179:| `figCK2_cell_response.png` | per-patch texture response, arm vs reference |
211:**Reproducibility.** 코드 커밋 d8d092f, 서버 63 `~/rest_x93d_ck2`, 이미지는 `results/evaluation/93D/addendum_checker/`에 있습니다.
RUN [Grep WL93A-D worklogs for amendments, deviations and checkpoint identity] cd /c/Projects/ReST-GI && for f in 93A_WORKLOG 93B_WORKLOG 93C_WORKLOG 93D_WORKLOG_ECRO 93D_WORKLOG_ADDENDUM; do echo "=================== $f"; grep -n -i "amend\|post hoc\|post-hoc\|deviat\|disclos\|pre-freeze\|revis\|exposure\|extension\|wording\|re-interpret\|seen before\|freeze\|own_s\|checkpoint\|sha256\|CK2\|held-out\|heldout" docs/worklogs/${f}*.md | cut -c1-520; done
RUN [Read how the CK2 scene was designed] sed -n 10,60p docs/worklogs/93D_WORKLOG_ADDENDUM_CHECKERBOARD_QUALITATIVE.md | cut -c1-1500; python - <<'EOF'
import json
p=json.load(open('experiments/ecro_local93d/addendum_checker/protocol_checker93d.json'))
for k,v in p.items():
    print(k,':',json.dumps(v)[:2500]); print()
EOF
  -> - Results: `results/ecro_local93d_checker/`. Reviewer images: `results/evaluation/93D/addendum_checker/`

## Revision history (why this report replaces the first one)

**Revision 1 is withdrawn.** It used a back-wall checker on E2 T1.
- The S1 panel lobe at about 1.8 m did not resolve 25 cm cells, s ...[cut]... rence median", "numbers": "descriptive only: region E_H (WL86 definition) in ROI_T (panel pixels whose ideal-mirror ray hits a checker patch), panel, box seen directly, frame; texture response slope / correlation / relative error vs the reference; reflection correspondence; table-level S / G ratios (diag_sg93d.py)", "labels": "none. No verdict, gate or label is computed; the frozen WL93D labels are unchanged"}

not_done : "no training, no checkpoint / operator change, no new transport mechanism, no image-space pattern, no protocol / verdict change of WL93D, no full-scale evaluation, no WL94"
RUN [Read how the CK2 scene was designed] cd /c/Projects/ReST-GI && sed -n 10,60p docs/worklogs/93D_WORKLOG_ADDENDUM_CHECKERBOARD_QUALITATIVE.md | cut -c1-1500; python - <<'EOF'
import json
p=json.load(open('experiments/ecro_local93d/addendum_checker/protocol_checker93d.json'))
for k,v in p.items():
    print(k,':',json.dumps(v)[:2500]); print()
EOF
RUN [Locate the computation behind the 4.3 % ring-0 irradiance figure] ls results/wl95_calc results/wl96_calc results/wl96_calc/scratch 2>/dev/null | head -80; grep -rn "4\.3 %\|0\.043\|ring0_share\|ring_share\|ring-0 irr" results/wl95_calc/*.py results/wl95_calc/*.json results/wl96_calc/*.py results/wl96_calc/*.json docs/plans/wl95_sharp_transport_design/*.md docs/plans/wl96_spatio_angular_carrier/*.md 2>/dev/null | cut -c1-400 | head -20
  -> results/wl95_calc:
S1_ops.npz
wl95_calc.json
wl95_calc.py
wl95_calc2.json
wl95_calc2.py
wl95_calc3.json
wl95_calc3.py

results/wl96_calc:
ck2_support.py
ck2_support_E.npy
ring_alloc.json
ring_alloc.py
scratch

results/wl96_calc/scratch:
critic_cost
critic_phys
d1
d2
d3
d4
results/wl95_calc/wl95_calc ...[cut]... harp_transport_design/read_wl89_92.md:85:  | truth noise | 0.043 |
docs/plans/wl95_sharp_transport_design/read_wl89_92.md:197:| RS | 0.048 | 0.043 |
docs/plans/wl95_sharp_transport_design/read_wl89_92.md:224:| SUM fits | single + L₃₊ 0.074–0.111; whole 0.016–0.043 | — |
docs/plans/wl95_sharp_transport_design/read_wl94.md:282:| s8601 E_H (signed) | 0.0439 (−0.030) | 0.0619 (+0.040) | 0.0438 (+0.003) | 0.0443 (−0.022) |
docs/plans/wl96_spatio_angular_carrier/critic_physics.md:109:- [INF] WL95A's RANK16_DELIVERY = INADEQUATE holds for the bounds as fitted (training incidence had ring 0 at 4.3 %).
RUN [Locate the computation behind the 4.3 % ring-0 irradiance figure] cd /c/Projects/ReST-GI && ls results/wl95_calc results/wl96_calc results/wl96_calc/scratch 2>/dev/null | head -80; grep -rn "4\.3 %\|0\.043\|ring0_share\|ring_share\|ring-0 irr" results/wl95_calc/*.py results/wl95_calc/*.json results/wl96_calc/*.py results/wl96_calc/*.json docs/plans/wl95_sharp_transport_design/*.md docs/plans/wl96_spatio_angular_carrier/*.md 2>/dev/null | cut -c1-400 | head -20
RUN [Read the WL95 train ring share computation] sed -n 120,175p results/wl95_calc/wl95_calc.py | cut -c1-330; sed -n 40,60p results/wl95_calc/wl95_calc.json
  -> for k in (4, 8, 16, 24, 32):
                A, info = rrr(tr, M, W, k)
                arms[f'rrr{k}_{wn}'] = A
                o[f'rrr_info_{wn}'] = info
        A16, _ = rrr(tr, M, W94, 16)
        arms['rrr16_W94_clip0'] = np.maximum(A16, 0)
        # per-panel-patch-index field-adapted rank-16  ...[cut]... s.path.join(OUT, 'wl95_calc.json'), 'w'), indent=1)


if __name__ == '__main__':
    run()
     "rank999": 58
    }
   ],
   "rrr16_per_patch_W94_val": {
    "field_rel_l1": 0.582609680082697,
    "energy": 0.7110879092892842,
    "peak": 0.4338633785350783,
    "op_l1_mean": 0.775309038343374
   },
   "train_ring_share": [
    0.043244618360592346,
    0.14591156728664084,
    0.2887325876942821,
    0.5221112266584848
   ],
   "train_bins_lit_per_row_median": 1.0,
   "train_top1_bin_share_median": 0.9373142936578305,
   "arms": {
    "svd1": {
     "op_l1": 1.281322983981894,
     "fro_c": [
RUN [Read the WL95 train ring share computation] cd /c/Projects/ReST-GI && sed -n 120,175p results/wl95_calc/wl95_calc.py | cut -c1-330; sed -n 40,60p results/wl95_calc/wl95_calc.json
RUN [Read WL95 calc inputs and the source encoding code location] sed -n 1,60p results/wl95_calc/wl95_calc.py | cut -c1-300; grep -n "def source\|def bin_of\|def state_rays\|PSRC\|u_state\|table_seed" experiments/ecro86/pt86.py | head -20
  -> """WL95 Stage B offline CPU calculation (read-only inputs; writes only ~/wl95_calc/).
Matched-norm representability bounds for the M64 local operator (S1 panel mat 1; contrast B mat 0, S2 mat 2).
Metrics copied from experiments/ecro_s94d/s94.py:sender (operator, rel_l1_flux, energy_ratio, peak_ratio ...[cut]... 
114:        rs = np.random.default_rng(P['representation'].get('table_seed', 86001))
116:        self.u_state_np = us
117:        self.u_src_np = rs.random((2, 1, PSRC), np.float32)
126:        self.US = [fl(np.broadcast_to(self.u_state_np[k], (N, K))) for k in range(4)]
127:        self.UP = [fl(np.broadcast_to(self.u_src_np[k], (N, PSRC))) for k in range(2)]
153:    def bin_of(x, y, z):
353:    def state_rays(self, idx=None):
378:    def source(self, lights):
380:        Np = self.N * PSRC
382:        pid = lane // PSRC
402:        return (s / PSRC).reshape(self.N, NB, 3).astype(np.float32)
RUN [Read WL95 calc inputs and the source encoding code location] cd /c/Projects/ReST-GI && sed -n 1,60p results/wl95_calc/wl95_calc.py | cut -c1-300; grep -n "def source\|def bin_of\|def state_rays\|PSRC\|u_state\|table_seed" experiments/ecro86/pt86.py | head -20
RUN [Read the source-encoding implementation in pt86] sed -n 105,128p experiments/ecro86/pt86.py | cut -c1-260; sed -n 150,160p experiments/ecro86/pt86.py; sed -n 376,403p experiments/ecro86/pt86.py | cut -c1-260; grep -n "def sample_patch\|def patch_point\|def ppoint" experiments/ecro86/pt86.py
  -> for b, name in (('A', 'boxA'), ('B', 'boxB')):
            M = g.box_matrix(b)
            V = np.asarray(self.params[f'{name}.vertex_positions'], np.float64).reshape(-1, 3)
            self.V0[name] = (V - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
            Nn = np.asarray(self.params[f'{name}.verte ...[cut]... f.scene.ray_test(r, on)
            e = dr.select(on & ~occ, cs * dr.rcp(d2), 0.0)
            bn = self.bin_of(dr.dot(wl, t), dr.dot(wl, bv), dr.dot(wl, n))
            ch = [e * lights.rgb[l].x, e * lights.rgb[l].y, e * lights.rgb[l].z]
            slot = pid * NB + bn
            dr.eval(ch, slot); dr.sync_thread()
            sl = np.asarray(slot)
            for c in range(3):
                s[:, c] += np.bincount(sl, weights=np.asarray(ch[c], np.float64), minlength=self.N * NB)
        return (s / PSRC).reshape(self.N, NB, 3).astype(np.float32)

183:    def patch_point(self, pid, a, b):
RUN [Read the source-encoding implementation in pt86] cd /c/Projects/ReST-GI && sed -n 105,128p experiments/ecro86/pt86.py | cut -c1-260; sed -n 150,160p experiments/ecro86/pt86.py; sed -n 376,403p experiments/ecro86/pt86.py | cut -c1-260; grep -n "def sample_patch\|def patch_point\|def ppoint" experiments/ecro86/pt86.py
RUN [Compute geometry-only coverage of CK2-like panel transport in the training populations] sed -n 183,200p experiments/ecro86/pt86.py | cut -c1-240 && cat > experiments/audit97/B/panel_coverage.py <<'EOF'
"""WL97 audit track B: geometry-only coverage of CK2-like sharp-panel transport in the ECRO training populations.

  python panel_coverage.py <population_regen.json> <out_json>

Independent numpy re-implementation (no renderer, no stored field): for every training lighting, the direct irradiance of the S1 panel
(42 patches x the 16 deployed source points, rng 86001 as in pt86.Ctx; sh
  -> Exit code 2
/usr/bin/bash: -c: line 117: unexpected EOF while looking for matching `''
RUN [Compute geometry-only coverage of CK2-like panel transport in the training populations] cd /c/Projects/ReST-GI && sed -n 183,200p experiments/ecro86/pt86.py | cut -c1-240 && cat > experiments/audit97/B/panel_coverage.py <<'EOF'
"""WL97 audit track B: geometry-only coverage of CK2-like sharp-panel transport in the ECRO training populations.

  python panel_coverage.py <population_regen.json> <out_json>

Independent numpy re-implementation (no renderer, no stored field): for every training lighting, the direct irradiance of the S1 panel
(42 patches x the 16 deployed source points, rn
WRITE c:\Projects\ReST-GI\experiments\audit97\B\panel_coverage.py
RUN [Run the panel coverage analysis] sed -n 183,196p experiments/ecro86/pt86.py | cut -c1-240; PYTHONDONTWRITEBYTECODE=1 python experiments/audit97/B/panel_coverage.py results/audit97/B/population_regen.json results/audit97/B/panel_coverage.json
  -> def patch_point(self, pid, a, b):
        g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
        pr = mi.Point3f(g(0), g(1), g(2)) + a * mi.Vector3f(g(3), g(4), g(5)) + b * mi.Vector3f(g(6), g(7), g(8))
        z = (g(20) + b) * (2.0 / S.SPH_NZ) - 1.0
        phi = (g(19) + a) * (TWO_PI / S.SPH_ ...[cut]... gs_mirror_to_box_ge_half_CK2": 244, "lightings_ring0_mirror_to_box_ge_CK2": 0, "lightings_ring0_mirror_to_box_ge_half_CK2": 1, "lightings_ring0_mirror_to_box_ge_tenth_CK2": 1, "lightings_grazing_and_any_mirror_to_box": 0, "files_with_a_ring0_mirror_to_box_ge_tenth_CK2": 1, "top5_ring0_mirror_to_box": [[11.632200314894666, "g00091.npz", 1, [-1.084, 1.49, 2.01], 0.525], [0.474971593279012, "g00056.npz", 0, [-0.958, -1.552, 2.627], 0.32], [0.0, "g00104.npz", 3, [0.757, 1.593, 1.365], 0.955], [0.0, "g00104.npz", 2, [0.232, -0.853, 1.846], 0.945], [0.0, "g00104.npz", 1, [0.6, -0.096, 1.656], 1.0]]}
WRITE c:\Projects\ReST-GI\experiments\audit97\B\panel_coverage.py
RUN [Run the panel coverage analysis] cd /c/Projects/ReST-GI && sed -n 183,196p experiments/ecro86/pt86.py | cut -c1-240; PYTHONDONTWRITEBYTECODE=1 python experiments/audit97/B/panel_coverage.py results/audit97/B/population_regen.json results/audit97/B/panel_coverage.json
RUN [List local WL94-96A result files and inspect WL95A main.json structure] ls -la results/ecro_attr95a results/ecro_util96a results/ecro_s94d results/ecro_q94c results/ecro_exp94b results/ecro_mat94r results/ecro_mat94/models results/ecro_mat94/pop 2>/dev/null | head -120; python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
def keys(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<2:
            print(pre+k+'/'); keys(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v); print(pre+k, s[:160])
key
  -> results/ecro_attr95a:
total 4565
drwxr-xr-x 1 dna10 197609       0 Nov 15 06:08 .
drwxr-xr-x 1 dna10 197609       0 Nov 16 02:45 ..
-rw-r--r-- 1 dna10 197609   26886 Nov 15 06:08 an.json
-rw-r--r-- 1 dna10 197609    1557 Nov 15 06:08 chain.status
-rw-r--r-- 1 dna10 197609 4213168 Nov 15 06:08 deliv. ...[cut]... ssive true
    nonneg true
  A0_s8601/
    op_flux_l1_mean 1.0179047599119777
    albedo_max 2.263723684963198
    passive_all false
    nonneg true
    mass_mean 0.8689575677345066
  A0_s8602/
    op_flux_l1_mean 0.9870053594284064
    albedo_max 3.077449703949969
    passive_all false
    nonneg true
    mass_mean 0.8349744492310064
D_pilot 16
D_used 64
T_DEPLOYED_ID true
gpu_mem_peak_mib_torch 168.98388671875
tables ["G_s8601", "G_s8602", "dep_A0_s8601", "dep_A0_s8602", "dep_A1", "dep_A2", "dep_A3", "dep_A4", "floor", "pool_A0_s8601", "pool_A0_s8602", "pool_A1", "pool_A2",
RUN [List local WL94-96A result files and inspect WL95A main.json structure] cd /c/Projects/ReST-GI && ls -la results/ecro_attr95a results/ecro_util96a results/ecro_s94d results/ecro_q94c results/ecro_exp94b results/ecro_mat94r results/ecro_mat94/models results/ecro_mat94/pop 2>/dev/null | head -120; python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
def keys(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<2:
            print(pre+k+'/'); keys(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v);
RUN [Find the learned-operator albedo computation in a95.py] grep -n "albedo\|def learned\|factors\|FZ_own" experiments/ecro_attr95a/a95.py | cut -c1-260 | head -50
  -> 148:    """operator [NB_in, NB_out, 3] vs M64: flux-L1 (zb_in zb_out), mass, directional albedo max (W_T), min entry."""
151:            'mass': float((A * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum()), 'albedo_max': float(alb.max()),
152:            'albedo_mean': float(alb.mean()), 'min_entry' ...[cut]...  convention on a Lambert material (id 3, albedo 0.7)
196:    alb = float(np.asarray(ctx.mats.albedo)[3])
317:        om[f'A0_s{sd}'] = {'op_flux_l1_mean': float(np.mean([p['op_flux_l1'] for p in per])), 'albedo_max': float(np.max([p['albedo_max'] for p in per])),
383:        g = np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{sd}.npz'))['none|G'].astype(np.float64)
559:    ax[3].bar(ok_, [om[k]['albedo_max'] for k in ok_]); ax[3].axhline(1, color='r', ls='--', label='passivity limit')
560:    ax[3].set_title('operator max directional albedo (W_T)', fontsize=8); ax[3].legend(fontsize=7)
RUN [Find the learned-operator albedo computation in a95.py] cd /c/Projects/ReST-GI && grep -n "albedo\|def learned\|factors\|FZ_own" experiments/ecro_attr95a/a95.py | cut -c1-260 | head -50
RUN [Read learned-operator construction and the model definition] sed -n 140,182p experiments/ecro_attr95a/a95.py | cut -c1-280; sed -n 300,320p experiments/ecro_attr95a/a95.py | cut -c1-300; sed -n 60,137p experiments/ecro86/model86.py | cut -c1-240
  -> """S1 field [42, NB, 3]: flux rel L1 / energy / peak of X against reference Y."""
    Wd = W[None, :, None]
    pk = Y.argmax(1)[:, None, :]
    return {'rel_l1': float((np.abs(X - Y) * Wd).sum() / (Y * Wd).sum()), 'energy': float((X * Wd).sum() / (Y * Wd).sum()),
            'peak': float(np.take_a ...[cut]... ], u, gs.area)
            y = y + torch.einsum('ncrp,lcp->lncr', f['P'], wq)
        return torch.einsum('ndcr,lncr->lndc', f['a'], y)

    def forward(self, gs, s, **kw):
        return self.apply(self.factors(gs), gs, s, **kw)


def loss_fn(pred, tgt):
    """Relative L2 with a denominator from the target's patch mean over bins (64-bin average: low-noise) + a floor."""
    den = tgt.mean(2, keepdim=True) + 0.05 * tgt.mean(dim=(1, 2, 3), keepdim=True)
    return (((pred - tgt) / den) ** 2).mean()


def rel_l1(pred, tgt):
    return float((pred - tgt).abs().sum() / tgt.sum().clamp(min=1e-30))
RUN [Read learned-operator construction and the model definition] cd /c/Projects/ReST-GI && sed -n 140,182p experiments/ecro_attr95a/a95.py | cut -c1-280; sed -n 300,320p experiments/ecro_attr95a/a95.py | cut -c1-300; sed -n 60,137p experiments/ecro86/model86.py | cut -c1-240
RUN [Read the first part of the WL94 worklog] cut -c1-1300 docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md | sed -n 1,140p
  -> # Worklog — ECRO Material Generalization vs. Sharp Angular Representation (WL94) (2026-11-14)

## Origin and scope

- **Brief.** "WL94 — ECRO Material Generalization vs. Sharp Angular Representation: Controlled Attribution and Conditional Small-Scale Training" (user, 2026-11-14).
- **Batch class.**  ...[cut]... | 0.31 / 0.48 |
| S, exact receiver (C2_rcv) | 1.00 / 0.97 | 1.00 / 0.98 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| G, learned global | 0.60 / 0.58 | 0.16 / 0.42 | 0.63 / 0.62 | 0.32 / 0.48 | 0.36 / 0.48 |

- Reference S/N: the median |rho| / noise over box-A patches is 990–3,950, so references are not noise-limited.
- Sender-side substitution (C2_src, C2_srcS1) leaves the material amplitude unchanged (±0.02).

**Faces and bins (C0, B amplitude).**
- Panel face vs other faces: bx38 0.22 / 0.31 (s8601), 0.45 / 0.49 (s8602).
- Panel-facing bins vs other bins: bx38 0.33 / 0.30 and 0.55 / 0.48.
RUN [Read the first part of the WL94 worklog] cd /c/Projects/ReST-GI && cut -c1-1300 docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md | sed -n 1,140p
RUN [Read the remainder of the WL94 worklog] cut -c1-900 docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md | sed -n 140,300p | grep -v "^$" | head -150
  -> - Panel-facing bins vs other bins: bx38 0.33 / 0.30 and 0.55 / 0.48.
**Textured error decomposition at box-A patches (bx38, C0).**
- Response term 0.30 / 0.24 against baseline-carried term 0.065 / 0.068.
- The untextured baseline error is 0.06 / 0.07.
**Sender split, untextured.** The single flight  ...[cut]... pervising assistant.
## Reproducibility
- **Commit and protocol.** Freeze `e347904` (protocol sha256 prefix 99e199f3847939e8). The post hoc script `posthoc94.py` is committed with the results.
- **Server.** LabServer63 `~/rest_x94`; `chain.status`; `results/`: pre, fields, tables, renders, models, pop, an, eval94. The dev root is `~/rest_x94dev` (not evidence).
- **Checkpoints.** own_s8601 `739e7eb5…`, own_s8602 `cee9a09a…` (file sha256, = WL92B repro.json). Candidate init sha256 `a85ddb42…` for both A and B.
- **Scene.** `experiments/ecro_local93d/addendum_checker/scene_ck2.json` (unchanged).
RUN [Read the remainder of the WL94 worklog] cd /c/Projects/ReST-GI && cut -c1-900 docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md | sed -n 140,300p | grep -v "^$" | head -150
RUN [Grep WL94A/C/D worklogs for disclosures and revisions] for f in 94A 94C 94D; do echo "=================== $f"; grep -n -i "amend\|post hoc\|post-hoc\|deviat\|disclos\|pre-freeze\|revis\|exposure\|extension\|wording\|re-interpret\|seen\|feas\|G_PHYS\|match\|result in one line\|seed disagree\|one seed\|both seeds" docs/worklogs/${f}_WORKLOG*.md | cut -c1-900; done
  -> =================== 94A
36:  - CK2 tables recomputed from the checkpoints match the stored ones to 1.8e-7;
38:- T_MIX: 48 / 48 examples; per-item material digests match select94; the step split is as precomputed.
45:**Seed 8602.** It was run because C_s8601 passed F2 and MR (brief Stage D). It compr ...[cut]... n is not separated at this budget.
170:- SENDER_IMP is false on both seeds, and RENDER_IMP is false on both. KEEP holds on both.
180:| Does direct sender supervision improve both seeds? | **No.** The sender criterion fails on both seeds. |
182:| Are global transport, F2 and MR preserved? | **Yes**, on both seeds. On the other box-A faces the global error rises slightly on s8601 (0.073 vs 0.054). |
191:  - Feasibility: `~/rest_x94dfeas` (2 audit runs + 1 probe; no training).
210:    - figS4 images per variant (REF / own / D / T / R / P / D with exact sender, both seeds, shared absolute scales).
RUN [Grep WL94A/C/D worklogs for disclosures and revisions] cd /c/Projects/ReST-GI && for f in 94A 94C 94D; do echo "=================== $f"; grep -n -i "amend\|post hoc\|post-hoc\|deviat\|disclos\|pre-freeze\|revis\|exposure\|extension\|wording\|re-interpret\|seen\|feas\|G_PHYS\|match\|result in one line\|seed disagree\|one seed\|both seeds" docs/worklogs/${f}_WORKLOG*.md | cut -c1-900; done
RUN [Read WL94C and WL94D result and feasibility sections] sed -n 22,70p docs/worklogs/94C_WORKLOG_ECRO_PANEL_MEDIATED_QUERY_SUPERVISION_SMOKE.md | cut -c1-900; echo =========94D; sed -n 22,60p docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md | cut -c1-1100
  -> - Figures: `results/evaluation/94C/`.

## Result in one line

The outcome is **INCONCLUSIVE** by the frozen mapping, because the seeds disagree.

| seed | finding |
|---|---|
| s8601 | D is better than C and E. Untextured sharp E_H is 0.044 vs C 0.084 / E 0.086, which is the frozen-own level 0.047.  ...[cut]... showed that the sharp-reflector e2_p label at 256 spp is noise-limited: its two independent realizations correlate only 0.525. The exact flight agrees with the label better than the label agrees with itself.
- The rule was therefore replaced before the freeze by the noise-aware form above. The noisy e2_p label is not used as a sender target anywhere.

**CK2 flight-floor diagnostic** (precise 2 × 4,096-spp reference; not a gate).

| receivers | ratio | corr | rel L1 | reference half noise |
|---|---|---|---|---|
| all receivers | 1.38 | 0.71 | 0.94 | 0.13 |
| box A | 1.21 | 0.72 | 0.53 | 0.06 |
RUN [Read WL94C and WL94D result and feasibility sections] cd /c/Projects/ReST-GI && sed -n 22,70p docs/worklogs/94C_WORKLOG_ECRO_PANEL_MEDIATED_QUERY_SUPERVISION_SMOKE.md | cut -c1-900; echo =========94D; sed -n 22,60p docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md | cut -c1-1100
RUN [Hash every checkpoint across the batch run directories] ssh LabServer63 'cd ~; sha256sum rest_ecro86/results/models/*.pt rest_x92a/results/models/*.pt 2>/dev/null | cut -c1-16,65-; echo; for d in rest_x92b rest_x93 rest_x93a rest_x93b rest_x93c rest_x93d rest_x93d_ck2 rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d rest_x95a rest_x96a; do echo "== $d"; find $d -maxdepth 4 \( -name "*.pt" -o -name "*.pth" \) 2>/dev/null | head -40 | while read f; do if [ -L "$f" ]; then echo "L $(readlink $f) <- $f"; else echo "$(sha256sum $f | cut -c1-16) $f"; fi; d
  -> 9d332cd61a7118bd  rest_ecro86/results/models/ecro_s8601.pt
4d993121e41dd0bb  rest_ecro86/results/models/ecro_s8602.pt
739e7eb50b25d329  rest_x92a/results/models/own_s8601.pt
cee9a09a289fca8c  rest_x92a/results/models/own_s8602.pt
3ec1d5910a3e3a3e  rest_x92a/results/models/sum_s8601.pt
d911c6b22cfd90 ...[cut]... s/models/B_s8601.pt
L /home/nd/rest_x94c/results/models/D_s8601.pt <- rest_x94d/results/models/D_s8601.pt
L /home/nd/rest_x94c/results/models/C_s8601.pt <- rest_x94d/results/models/C_s8601.pt
L /home/nd/rest_x94c/results/models/E_s8602.pt <- rest_x94d/results/models/E_s8602.pt
668bf4502c3c2f3a rest_x94d/results/models/P_s8602.pt
L /home/nd/rest_x94c/results/models/C_s8602.pt <- rest_x94d/results/models/C_s8602.pt
c256bd3ef72a12bf rest_x94d/code/results/fnto_direction64/independent_lights.pt
== rest_x95a
c256bd3ef72a12bf rest_x95a/code/results/fnto_direction64/independent_lights.pt
== rest_x96a
RUN [Check the WL92A training population used by the code and logs] grep -n "def train\|kept_files\|limit\|rng\|permutation\|torch.manual_seed\|load_state\|init" experiments/ecro_own92a/own92a.py | sed -n 1,60p | cut -c1-240; ls results/ecro_own92a/results/models; python - <<'EOF'
import json
d=json.load(open('results/ecro_own92a/results/models/own_s8601.json'))
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in d.items()})
EOF
  -> 72:def kept_files(R, R86, sp, limit=None):
75:    return fns[:limit]
89:    keep = {os.path.basename(f) for f in kept_files(R, R86, split)}
128:    def __init__(self, R, sp, fn):
153:def load_items(R, R86, sp, limit=None):
155:    fns = [f for f in kept_files(R, R86, sp) if os.path.exists(os.path.jo ...[cut]... a256'] for a in ('sum', 'own') if os.path.exists(os.path.join(R, 'results', 'models', f'{a}_s{sd}.json'))]
291:        inits[sd] = h
292:    res['T_INIT'] = {'hashes': inits, 'PASS': bool(inits and all(len(v) == 2 and v[0] == v[1] for v in inits.values()))}
own_s8601.json
own_s8602.json
sum_s8601.json
sum_s8602.json
{'seed': 8601, 'arm': 'own', 'steps': 24000, 'params': 735664, 'init_sha256': '939ab4702706dc073711f6faa7cf1b7a3c8ad9071c0b4e04771a8499559979f7', 'train_s': 2287.977981567383, 'items': 928, 'val_items': 55, 'nonfinite_steps': 0, 'gpu_mem_peak_mib': 629.33642578125, 'log': 'list48'}
RUN [Check the WL92A training population used by the code and logs] cd /c/Projects/ReST-GI && grep -n "def train\|kept_files\|limit\|rng\|permutation\|torch.manual_seed\|load_state\|init" experiments/ecro_own92a/own92a.py | sed -n 1,60p | cut -c1-240; ls results/ecro_own92a/results/models; python - <<'EOF'
import json
d=json.load(open('results/ecro_own92a/results/models/own_s8601.json'))
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in d.items()})
EOF
RUN [Extract exposure and disclosure statements from every ECRO protocol] python - <<'EOF'
import json,glob,os
for d in ['ecro_attr92','ecro_own92a','ecro_cf92b','ecro_stage93','ecro_stage93a','ecro_stage93b','ecro_stage93c','ecro_local93d','ecro_mat94','ecro_mat94r','ecro_exp94b','ecro_q94c','ecro_s94d']:
    fn=glob.glob(f'experiments/{d}/protocol/*.json')[0]
    p=json.load(open(fn))
    print('=====',d, os.path.basename(fn))
    def walk(o,path=''):
        if isinstance(o,dict):
            for k,v in o.items(): walk(v,path+'/'+k)
        elif isinstance(o,list):
  -> ===== ecro_attr92 ecro_attr92.json
  /local_operator/flux_matrix : X[bi, bo] = M64[bi, bo] z_bi z_bo (z = mean cosine of the bin's ring): incident-cosine-weighted irradiance in a bin -> outgoing flux in a bin; rows sum to the directional albedo, so all incident bins are commensurable. The plain Fro ...[cut]... YS. Dev chain ~/rest_x94ddev (60 steps, 32-spp renders): run 1 failed with CUDA OOM (10 concurrent processes on the 12 GB GPU) -> the chain now runs at most 4 CUDA processes; run 2 completed end to end. Dev candidate outputs (tables, sender evaluations, renders, analysis, figures, final label) were not opened.
  /feasibility_result/run : ~/rest_x94dfeas (run 2, revised G_PHYS): REPRO PASS, AUDIT94D PASS
  /feasibility_result/G_PHYS : training items: pooled ratio 0.953; corr exact vs 256-spp label 0.641 >= label reliability 0.525 (256 vs 64 spp); run 1 with corr >= 0.9 failed and is disclosed
RUN [Extract exposure and disclosure statements from every ECRO protocol] cd /c/Projects/ReST-GI && python - <<'EOF'
import json,glob,os
for d in ['ecro_attr92','ecro_own92a','ecro_cf92b','ecro_stage93','ecro_stage93a','ecro_stage93b','ecro_stage93c','ecro_local93d','ecro_mat94','ecro_mat94r','ecro_exp94b','ecro_q94c','ecro_s94d']:
    fn=glob.glob(f'experiments/{d}/protocol/*.json')[0]
    p=json.load(open(fn))
    print('=====',d, os.path.basename(fn))
    def walk(o,path=''):
        if isinstance(o,dict):
            for k,v in o.items(): walk(v,path+'/'+k)
      
RUN [Read the setup portion of a95.py] sed -n 1,135p experiments/ecro_attr95a/a95.py | cut -c1-230
  -> """WL95A (protocol ecro_attr95a): sharp sender representability and flight attribution smoke on CK2 'none' (learning-free diagnostics).
Production code (model86 / ecro86 / pt86 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python a95.py stage0 <R> <R92A>     contract tests,  ...[cut]... thread()
    out[:, 0], out[:, 1], out[:, 2] = np.asarray(L.x), np.asarray(L.y), np.asarray(L.z)
    return out


def x1_rays(PT, ctx, lt, hit, seg, centre):
    """per-ray exact S1 radiance [N*K, 3] for the rays whose recorded hit is an S1 patch (0 elsewhere)."""
    idx = np.nonzero(np.isin(hit, S1))[0]
    o = seg[idx, :3].astype(np.float64); e = seg[idx, 3:].astype(np.float64)
    w = e - o; w /= np.linalg.norm(w, axis=1, keepdims=True)
    L = np.zeros((len(hit), 3))
    L[idx] = x1_radiance(PT, ctx, lt, o, w, ctx.G.PC[hit[idx].astype(np.int64)] if centre else None)
    return L, len(idx)
RUN [Read the setup portion of a95.py] cd /c/Projects/ReST-GI && sed -n 1,135p experiments/ecro_attr95a/a95.py | cut -c1-230
RUN [Inspect mat94 helper functions for CK2 context and checkpoint loading] grep -n "^def \|^DEV\|^SC \|^P = \|os.makedirs\|np.savez\|json.dump\|def jd" experiments/ecro_mat94/mat94.py | cut -c1-200 | head -80; grep -n "def ctx_ck2" -A 22 experiments/ecro_mat94/mat94.py | cut -c1-220; grep -n "def load_frozen" -A 14 experiments/ecro_mat94/mat94.py | cut -c1-220; grep -n "def install" -A 25 experiments/ecro_mat94/mat94.py | cut -c1-200
  -> 23:DEV = '--dev' in sys.argv
24:P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_mat94.json')))
26:SC = json.load(open(SCENE_FN))
42:def jd(o, fn):
43:    os.makedirs(os.path.dirname(fn), exist_ok=True)
44:    json.dump(o, open(fn, 'w'), indent=1)
47:def jl(fn):
51:def sha_file(fn):
55:def la ...[cut]... sc[3:4], (k, 1))]).astype(np.float32)
158-        self.lambert = np.concatenate([self.lambert, np.ones(k, self.lambert.dtype)])
159-        self.albedo = np.concatenate([self.albedo, np.array([ALB[i] for i in NEW_IDS], self.albedo.dtype)])
160-        self.desc_cols = [PT.fl(self.desc[:, j]) for j in range(10)]
161-        self.lam_f = PT.fl(self.lambert); self.alb_f = PT.fl(self.albedo)
162-        self.cls = PT.U(np.array(list(S.MAT_CLASS) + [2] * k, np.uint32))
163-    PT.Mats.__init__ = mats_init
164-    ost = PT.Ctx.set_tables
165-
166-    def set_tables(self, g):
167-        ost(self, g)
RUN [Inspect mat94 helper functions for CK2 context and checkpoint loading] cd /c/Projects/ReST-GI && grep -n "^def \|^DEV\|^SC \|^P = \|os.makedirs\|np.savez\|json.dump\|def jd" experiments/ecro_mat94/mat94.py | cut -c1-200 | head -80; grep -n "def ctx_ck2" -A 22 experiments/ecro_mat94/mat94.py | cut -c1-220; grep -n "def load_frozen" -A 14 experiments/ecro_mat94/mat94.py | cut -c1-220; grep -n "def install" -A 25 experiments/ecro_mat94/mat94.py | cut -c1-200
RUN [Check import paths and state-ray function for the server script] sed -n 14,45p experiments/ecro_attr92/fx92.py | cut -c1-200; grep -n "def load_own" -A 14 experiments/ecro_stage93/train93.py | cut -c1-200; grep -n "sys.path" experiments/ecro_mat94/mat94.py experiments/ecro_attr92/fx92.py experiments/ecro_stage93/train93.py | cut -c1-200; grep -n "def state_rays" -A 24 experiments/ecro86/pt86.py | cut -c1-200
  -> PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
P92 = json.load(open(os.path.join(HERE, 'protocol', 'ecro_attr92.json')))
P87 = fx87.P87
E87 = P87['explicit']
SEEDS = ['ecro_s8601', 'ecro_s8602']


def inject(root=None, name=None):
    """Make E4 addressable. name: candidate id; de ...[cut]... si.is_valid()
366-        sid = self.shape_id(si)
367-        front = si.to_local(-w).z > 1e-6
368-        ph, bh = self.patch_bin(si, sid, -w)
369-        hit = dr.select(valid & front, mi.Int32(ph), mi.Int32(-1))
370-        tt = dr.select(valid, si.t, 30.0)
371-        e = o + w * tt
372-        vals = [hit, bh, o.x, o.y, o.z, e.x, e.y, e.z]
373-        dr.eval(vals); dr.sync_thread()
374-        hit = np.asarray(hit).astype(np.int16); bh = np.asarray(bh).astype(np.uint8)
375-        seg = np.stack([np.asarray(v) for v in vals[2:]], 1).astype(np.float32)
376-        return hit, bh, seg
377-
RUN [Check import paths and state-ray function for the server script] cd /c/Projects/ReST-GI && sed -n 14,45p experiments/ecro_attr92/fx92.py | cut -c1-200; grep -n "def load_own" -A 14 experiments/ecro_stage93/train93.py | cut -c1-200; grep -n "sys.path" experiments/ecro_mat94/mat94.py experiments/ecro_attr92/fx92.py experiments/ecro_stage93/train93.py | cut -c1-200; grep -n "def state_rays" -A 24 experiments/ecro86/pt86.py | cut -c1-200
WRITE c:\Projects\ReST-GI\experiments\audit97\B\learned_s1_albedo.py
RUN [Launch the learned-S1 albedo and coverage check on the server (short GPU job)] file experiments/audit97/B/learned_s1_albedo.py && scp -q experiments/audit97/B/learned_s1_albedo.py LabServer63:rest_x97/B/ && ssh LabServer63 'cd ~/rest_x97/B && file learned_s1_albedo.py && ls ~/rest_x96a/code/experiments | tr "\n" " " && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; mkdir -p own_R && (setsid nohup bash -c ". ~/rest_x95a/env.sh; \$PY learned_s1_albedo.py ~/rest_x96a/code/exper
  -> experiments/audit97/B/learned_s1_albedo.py: Python script, ASCII text executable
learned_s1_albedo.py: Python script, ASCII text executable
8dna_deformation_replication baseline_sup67 dynamic_transport_failure dynamic_transport_validity ecro86 ecro86a ecro_attr92 ecro_attr95a ecro_cf92b ecro_exp94b  ...[cut]... o_sel56 fnto_smoke47 fnto_spatial49 fnto_src73 fnto_srcid78 fnto_sym79 fnto_visual_review fnto_walker71c full_recompute_benchmark gm1_c3urx1_91 gm1_char85 gm1_explicit87 gm1_gap90 gm1_relation88 gm1_relation89 neural_recompute_cost phys_ref53 radiometric_relation_state_prototype relational_residual_prototype rna_baseline_quality rna_reproduction rna_selective_reconstruction_lego rna_surface_ownership rna_surface_ownership_b2 rna_surface_ownership_lego selective_refit_oracle server_setup spt_lego sunreuse70 surface_owned_radiance xbench69 1167 MiB, 2 %
3455953, 592 MiB
3456573, 560 MiB
launched
WRITE c:\Projects\ReST-GI\experiments\audit97\B\learned_s1_albedo.py
RUN [Launch the learned-S1 albedo and coverage check on the server (short GPU job)] cd /c/Projects/ReST-GI && file experiments/audit97/B/learned_s1_albedo.py && scp -q experiments/audit97/B/learned_s1_albedo.py LabServer63:rest_x97/B/ && ssh LabServer63 'cd ~/rest_x97/B && file learned_s1_albedo.py && ls ~/rest_x96a/code/experiments | tr "\n" " " && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; mkdir -p own_R && (setsid nohup bash -c ". ~/rest_x95a/env.sh; \$PY learned_s1_albedo
RUN [Inspect the WL94B audit.json structure] python - <<'EOF'
import json, numpy as np
a=json.load(open('results/ecro_exp94b/audit.json'))
def keys(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<1:
            print(pre+k+'/'); keys(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v); print(pre+k, s[:300])
keys(a)
EOF
  -> status "STOP_INVALID_INTERVENTION"
tests/
  T_PATH_ID {"pass": true, "id_f": 4.7173623004721995e-08, "id_f2": 5.790993639701784e-08}
  T_PANEL_ID {"pass": true}
  T_QUERY {"pass": true, "L": [4], "N": [1270], "queries_per_step": 975360}
  T_MIX {"pass": true, "examples": {"A": 48, "B": 48}}
   ...[cut]... 0562, "PMb_A": 0.07320514825513386, "PM_A": 0.0031138672055096763, "share_parts_item": [0.04331111345260041, 0.4568647859459218, 0.011564217722442564, 0.48825988288780275], "panel_share_e2_box": 0.1055504
semantics "the objective supervises e2_p + e2_o and e3_p + e3_o jointly; the panel's learned once-scattered response is supervised only through receivers. Exposure changes cannot add a path-specific supervision term (loss changes are not authorised); this is reported as a semantics limit, separate from sampli
protocol_sha256 "6b5b10c78d23933789c20232e0e2b2d4c0bfd7c9c17fe8085aa841e1c4685033"
RUN [Inspect the WL94B audit.json structure] cd /c/Projects/ReST-GI && python - <<'EOF'
import json, numpy as np
a=json.load(open('results/ecro_exp94b/audit.json'))
def keys(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<1:
            print(pre+k+'/'); keys(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v); print(pre+k, s[:300])
keys(a)
EOF
RUN [Verify the WL94B 4/96 count and hash protocols] python - <<'EOF'
import json, numpy as np
a=json.load(open('results/ecro_exp94b/audit.json'))
rows=a['rows']; ck=a['underrepresented_detail']['ck2']['PMb_A']
v=np.array([r['tgt']['PMb_A'] for r in rows])
print('n',len(v),'>=CK2',int((v>=ck).sum()),'>=0.5CK2',int((v>=0.5*ck).sum()),'mean',v.mean(),'CK2',ck)
print([ (r['file'],round(r['tgt']['PMb_A'],3)) for r in rows if r['tgt']['PMb_A']>=ck])
print(a['protocol_sha256'][:16])
# cross-check with my geometric proxy
c=json.load(open('results/audit97
  -> n 96 >=CK2 4 >=0.5CK2 12 mean 0.03617268906852757 CK2 0.1376716742311644
[('g00032.npz', 0.252), ('g00058.npz', 0.142), ('g00081.npz', 0.173), ('g00084.npz', 0.14)]
6b5b10c78d239337
spearman(mirror-to-boxA irradiance proxy, PMb_A)= 0.87011665762344
032a624e61ef10fd 032a624e61ef10fd Projects/ReST ...[cut]... CRLF
1159baaa9449e7aa e0a389463e172bc7 ojects/ReST-GI/experiments\ecro_s94d\protocol\ecro_s94d.json CRLF
68dcf525943082fe 68dcf525943082fe /ReST-GI/experiments\ecro_stage93\protocol\ecro_stage93.json LF
278af654eb89171b 278af654eb89171b eST-GI/experiments\ecro_stage93a\protocol\ecro_stage93a.json LF
471179e3c4601b32 471179e3c4601b32 eST-GI/experiments\ecro_stage93b\protocol\ecro_stage93b.json LF
9b7256739890ebca 9b7256739890ebca eST-GI/experiments\ecro_stage93c\protocol\ecro_stage93c.json LF
0a5936504fb8c345 0a5936504fb8c345 /ReST-GI/experiments\ecro_util96a\protocol\ecro_util96a.json LF
RUN [Verify the WL94B 4/96 count and hash protocols] cd /c/Projects/ReST-GI && python - <<'EOF'
import json, numpy as np
a=json.load(open('results/ecro_exp94b/audit.json'))
rows=a['rows']; ck=a['underrepresented_detail']['ck2']['PMb_A']
v=np.array([r['tgt']['PMb_A'] for r in rows])
print('n',len(v),'>=CK2',int((v>=ck).sum()),'>=0.5CK2',int((v>=0.5*ck).sum()),'mean',v.mean(),'CK2',ck)
print([ (r['file'],round(r['tgt']['PMb_A'],3)) for r in rows if r['tgt']['PMb_A']>=ck])
print(a['protocol_sha256'][:16])
# cross-check with my geometric proxy
c=json.
RUN [List the commit history across the audited batches] git log --format='%h %ad %s' --date=format:'%m-%d %H:%M' -75 | sed -n 1,75p | cut -c1-150
RUN [Check the server albedo job] ssh LabServer63 'cd ~/rest_x97/B; cat albedo.done 2>/dev/null; tail -12 albedo.log'
  -> efcf86a 11-16 02:13 WL96A complete: LEARNED_RESOLVENT_UTILITY NO_MEANINGFUL_HEADROOM (CK2)
a1718f7 11-16 02:04 WL96A freeze: learned higher-order transport utility ceiling (protocol ecro_util96a)
dbdc0e7 11-15 22:39 WL96 design-only: NO_VIABLE_LEARNED_MECHANISM (scoped, provisional; dissent recorded ...[cut]... 01:33 WL85 complete: GM1 mechanism-neutral characterisation - ER lobe census, qualified REF-P / REF-ER references, path-class and dirty 
355a161 11-11 00:59 WL85 protocol v1 freeze + code: GM1 mechanism-neutral transport characterisation - ER lobe census, rigid-edit glossy fixture, inst
db37c72 11-10 23:44 WL84A: GM1 global transport architecture review (design-only, multi-agent): three sealed design tracks (Directional Delta Transpor
c7b0698 11-10 21:39 docs: link qualified ER baseline evidence explicitly
1e15c3b 11-10 21:38 docs: align ReST-GI contribution goals with global milestone program
  -> rc=0
TRAIN ring share [0.04676284259598036, 0.15699955220279346, 0.287580673502623, 0.5086569316986032]
CK2 lit bins [12, 13, 14, 15] train share of those bins 0.0200391324714897
8601 CK2 max 2.263723684963198 ring max [2.263723684963198, 2.120184277131823, 1.7669183129196937, 1.4397494852241617] lit-bin max 1.8566016132486207 | train-state pooled max 2.6718508704367308 ring max [2.6610682997906236, 2.6718508704367308, 2.265782567716474, 1.4403597596806288] frac states nonpassive 1.0
8602 CK2 max 3.077449703949969 ring max [2.980313858291079, 3.077449703949969, 2.152442103964069, 1.307535572921655] lit-bin max 1.860365378901081 | train-state pooled max 3.5872976147022713 ring max [2.678626040739625, 3.5872976147022713, 2.4818301507958593, 1.3473024913498945] frac states nonpassive 1.0
DONE
RUN [List the commit history across the audited batches] cd /c/Projects/ReST-GI && git log --format='%h %ad %s' --date=format:'%m-%d %H:%M' -75 | sed -n 1,75p | cut -c1-150
RUN [Check the server albedo job] ssh LabServer63 'cd ~/rest_x97/B; cat albedo.done 2>/dev/null; tail -12 albedo.log'
RUN [Retrieve and summarise the learned-S1 albedo coverage result] scp -q LabServer63:rest_x97/B/learned_s1_albedo.json results/audit97/B/ && scp -q LabServer63:rest_x97/B/albedo.log results/audit97/B/learned_s1_albedo.log && python - <<'EOF'
import json, numpy as np
d=json.load(open('results/audit97/B/learned_s1_albedo.json'))
sh=np.array(d['train_P92A']['panel_irradiance_share_per_incident_bin'])
print('train', {k:v for k,v in d['train_P92A'].items() if k!='panel_irradiance_share_per_incident_bin'})
print('bin share min/max', sh.min(), sh.max(), 'bins 12-15',
  -> train {'files': 928, 'lightings': 3712, 'ring_share': [0.04676284259598036, 0.15699955220279346, 0.287580673502623, 0.5086569316986032], 'lightings_panel_lit': 3587, 'lightings_ring0_share_ge_0.9': 190, 'lightings_ring0_share_eq_1': 163}
bin share min/max 0.0007640976839236132 0.05568653101998363 b ...[cut]... 8602 CK2 max 3.077 frac>1 0.1539 ring max [2.98 3.08 2.15 1.31] ring mean [0.577 0.584 0.552 0.496] ring frac>1 [0.252  0.1652 0.1453 0.0531] argmax [2, 19, 1] lit 1.860365378901081 0.9127981307301889
own_s8602 train_states_pooled max 3.587 frac>1 0.1496 ring max [2.68 3.59 2.48 1.35] ring mean [0.547 0.521 0.516 0.519] ring frac>1 [0.256  0.1408 0.1438 0.0576] argmax [589, 19, 1] lit None None
  bins>1 (train states): 29 share of train irradiance in those bins 0.52 spearman 0.159
  per-bin mean CK2 (ring0): [0.23 0.15 0.33 0.09 0.08 0.03 0.12 0.08 1.58 1.43 1.13 0.32 1.27 1.14
 0.82 0.41]
RUN [Retrieve and summarise the learned-S1 albedo coverage result] cd /c/Projects/ReST-GI && scp -q LabServer63:rest_x97/B/learned_s1_albedo.json results/audit97/B/ && scp -q LabServer63:rest_x97/B/albedo.log results/audit97/B/learned_s1_albedo.log && python - <<'EOF'
import json, numpy as np
d=json.load(open('results/audit97/B/learned_s1_albedo.json'))
sh=np.array(d['train_P92A']['panel_irradiance_share_per_incident_bin'])
print('train', {k:v for k,v in d['train_P92A'].items() if k!='panel_irradiance_share_per_incident_bin'})
print('bin share min/max', sh.min(
RUN [Inspect WL94D sender evaluations on train, val and CK2 fields] ls results/ecro_s94d/sender results/ecro_s94d/models results/ecro_s94d/pop | head -60; python - <<'EOF'
import json,glob
for f in sorted(glob.glob('results/ecro_s94d/sender/SN_own_*.json'))+sorted(glob.glob('results/ecro_s94d/sender/SN_P_*.json'))+sorted(glob.glob('results/ecro_s94d/sender/SN_T_*.json')):
    d=json.load(open(f))
    print(f.split('sender')[-1], json.dumps(d)[:1500]); print()
print(open('results/ecro_s94d/audit.json').read()[:2400])
EOF
  -> results/ecro_s94d/models:
P_s8601.json
P_s8602.json
R_s8601.json
R_s8602.json
T_s8601.json
T_s8602.json

results/ecro_s94d/pop:
P_P_s8601.json
P_P_s8602.json
P_R_s8601.json
P_R_s8602.json
P_T_s8601.json
P_T_s8602.json

results/ecro_s94d/sender:
SN_D_s8601.json
SN_D_s8602.json
SN_P_s8601.json
SN_P_s8 ...[cut]... 423719253391027
    },
    "8602": {
     "T": 0.012704798185574741,
     "R": 1.143011621142709,
     "grad_own": 0.17591364146210253,
     "grad_T": 13.84623658657074,
     "grad_R": 0.15390363335609436,
     "teacher_loss_T0": 7.433330625295639,
     "teacher_loss_R0": 0.0877349351067096
    }
   }
  }
 },
 "coverage": {
  "train_bins": 64,
  "val_bins": 62,
  "val_bins_not_in_train": 0,
  "ck2_bins": 4,
  "ck2_bins_not_in_train": 0,
  "ck2_irradiance_share_in_train_bins": 1.0
 },
 "protocol_sha256": "e0a389463e172bc75c4a1b298b6addfb1e6e18c6529280419880b056d3894059"
}
RUN [Inspect WL94D sender evaluations on train, val and CK2 fields] cd /c/Projects/ReST-GI && ls results/ecro_s94d/sender results/ecro_s94d/models results/ecro_s94d/pop | head -60; python - <<'EOF'
import json,glob
for f in sorted(glob.glob('results/ecro_s94d/sender/SN_own_*.json'))+sorted(glob.glob('results/ecro_s94d/sender/SN_P_*.json'))+sorted(glob.glob('results/ecro_s94d/sender/SN_T_*.json')):
    d=json.load(open(f))
    print(f.split('sender')[-1], json.dumps(d)[:1500]); print()
print(open('results/ecro_s94d/audit.json').read()[:2400])
EOF
RUN [Read the WL92 gate section] sed -n 160,200p docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md | cut -c1-1100
  -> ### Fixed-geometry free-factor fits (non-deployable; 192 fit / 32 validation lights; 8,000 steps)

Region-C E_H (rendered): single-flight term + true L₃₊ / whole table. Convergence = relative change of the objective over the last 500 steps (rule ≤ 2 %).

| Fit | E2 T1 | conv. | E2 T0 | conv. | E1 T1 ...[cut]... t R16: 1.14–1.52×, mixed around 1.25); D: component targets valid, SUM criterion not met; E and F NOT TESTED; completion code 1.
- **Protocol wording defect (disclosed, no re-mapping).** Completion code 1 is defined as "representation / interface insufficient (G_info or G_rep fails)". Here G_rep fails because the in-family point was **not established** (material rank-16 construction insufficient; the geometry-specific gate fit unconverged), while verdict A is INCONCLUSIVE by the same protocol. Code 1 must be read as "G_rep not established", not as a demonstrated representation limit.

## 최종 보고
RUN [Read the WL92 gate section] cd /c/Projects/ReST-GI && sed -n 160,200p docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md | cut -c1-1100
RUN [Read the WL93 preflight gate and stop sections] sed -n 60,90p docs/worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md | cut -c1-900; sed -n 171,226p docs/worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md | cut -c1-900
  -> **Arms (2 seeds each, 8601 / 8602, initialised from the same-seed WL92A OWN).**

| Arm | Factors | Objective |
|---|---|---|
| T0 | tied | OWN (WL92A objective) |
| TQ | tied | OWN + source-stage |
| U0 | untied | OWN |
| UQ | untied | OWN + source-stage |
| C0 | tiedcm (capacity control for U0) | O ...[cut]... - the deployed encoding's directional information loss (≈ 0.29–0.42 on typical geometries, which pass);
  - the Monte Carlo noise of the Mq 1,024 once-scattered target on a few panel-illumination configurations, which dominates the pooled value.
- **Not shown.** This does not show that the target is identifiable. It also does not license changing the gate, excluding geometries or rerunning. A noisy q would also be a noisy supervision target on those geometries.

## Visuals

`results/evaluation/93/fig1_identifiability_gate.png` (manifest with hashes):
- **(a)** The frozen gate per set and seed.
RUN [Read the WL93 preflight gate and stop sections] cd /c/Projects/ReST-GI && sed -n 60,90p docs/worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md | cut -c1-900; sed -n 171,226p docs/worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md | cut -c1-900
RUN [Check whether the WL93 heavy-tail validation geometries have near-panel lights] python - <<'EOF'
import json, numpy as np
p=json.load(open('results/audit97/B/population_regen.json'))
for nm in ('g00010.npz','g00039.npz'):
    it=[i for i in p['val']['items'] if i['file']==nm][0]
    print(nm, 'A', np.round(it['geom']['A']['c'],2), 'B', np.round(it['geom']['B']['c'],2))
    for l in it['lights']:
        pos=np.array(l[0]['pos']); 
        # distance to panel plane x=-1.5 and to panel rectangle
        dy=max(abs(pos[1]-0.3)-0.6,0); dz=max(abs(pos[2]-1.0)-0.7,0)
        prin
  -> g00010.npz A [-0.76  0.91  2.47] B [ 0.82 -1.46  0.59]
   light [-0.216 -0.997  2.054] x+1.5= 1.284 dist to panel rect 1.503
   light [-1.353  0.075  2.383] x+1.5= 0.147 dist to panel rect 0.699
   light [-1.087  0.71   1.76 ] x+1.5= 0.413 dist to panel rect 0.417
   light [-1.493  0.806  1.036] x+1.5= 0.007 dist to panel rect 0.007
g00039.npz A [0.11 0.83 1.82] B [-0.32  0.08  0.5 ]
   light [-1.453  0.543  1.398] x+1.5= 0.047 dist to panel rect 0.047
   light [-1.294  0.928  2.691] x+1.5= 0.206 dist to panel rect 1.013
   light [1.199 0.71  1.694] x+1.5= 2.699 dist to panel rect 2.699
   light [ 1.067 -0.113  2.713] x+1.5= 2.567 dist to panel rect 2.759
train lights in front of panel 3985 of 4096; within 0.10 m of the panel rect: 32  within 0.2: 63 behind panel plane: 111
RUN [Check whether the WL93 heavy-tail validation geometries have near-panel lights] cd /c/Projects/ReST-GI && python - <<'EOF'
import json, numpy as np
p=json.load(open('results/audit97/B/population_regen.json'))
for nm in ('g00010.npz','g00039.npz'):
    it=[i for i in p['val']['items'] if i['file']==nm][0]
    print(nm, 'A', np.round(it['geom']['A']['c'],2), 'B', np.round(it['geom']['B']['c'],2))
    for l in it['lights']:
        pos=np.array(l[0]['pos']); 
        # distance to panel plane x=-1.5 and to panel rectangle
        dy=max(abs(pos[1]-0.3)-0.6,0); dz=max(abs(pos[2
RUN [Inspect WL95A analysis output and the extension rule implementation] python - <<'EOF'
import json
a=json.load(open('results/ecro_attr95a/an.json'))
def keys(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<1:
            print(pre+k+'/'); keys(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v); print(pre+k, s[:420])
keys(a)
EOF
grep -n "EXTEND\|extend\|D_PILOT\|D_EXT\|within 2\|boundary" experiments/ecro_attr95a/a95.py experiments/ecro_attr95a/run95a.sh | cut -c1-260
  -> delivery/
  dep|A0_s8601|boxA {"ratio": 0.49540279362983214, "signed": -0.5045972063701678, "abs_rel_l1": 0.5878543503200824, "ref_half_noise": 0.05926934169469287, "peak_patch_arm": 2, "peak_patch_ref": 5, "peak_amp_ratio": 0.5235688860019254, "value_at_ref_peak_ratio": 0.23048492045142208, "corr" ...[cut]...    out['D_used'] = D_PILOT
experiments/ecro_attr95a/a95.py:481:              'E_q256_pool': E_pool('q256'), 'E_A4_pool': E_pool('A4'), 'verdict': fa, 'near_boundary_extend': bool(near and mj['D_used'] < D_EXT)}
experiments/ecro_attr95a/a95.py:535:    print('AN', 'EXTEND' if flight['near_boundary_extend'] else 'DONE', flush=True)
experiments/ecro_attr95a/run95a.sh:52:if grep -q '^AN EXTEND' $O/logs/N1.log; then
experiments/ecro_attr95a/run95a.sh:53:  log "EXTEND to 64 draws (rule: decision boundary within 2 SE)"
experiments/ecro_attr95a/run95a.sh:54:  run M2 "A95_EXTEND=1 $A main $R $R92A $DEV"
RUN [Inspect WL95A analysis output and the extension rule implementation] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
a=json.load(open('results/ecro_attr95a/an.json'))
def keys(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<1:
            print(pre+k+'/'); keys(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v); print(pre+k, s[:420])
keys(a)
EOF
grep -n "EXTEND\|extend\|D_PILOT\|D_EXT\|within 2\|boundary" experiments/ecro_attr95a/a95.py experiments/ecro_attr95a/run95a.sh | cut -c1-260
RUN [Check WL95A and WL96A chain status records] cat results/ecro_attr95a/chain.status; echo; cat results/ecro_util96a/chain.status; echo; cat results/ecro_util96a/final.json; python - <<'EOF'
import json
s=json.load(open('results/ecro_util96a/s1.json'))
print({k:(v if not isinstance(v,(dict,list)) else '...') for k,v in s.items()})
print(json.dumps(s.get('decision', s.get('stage1', {})))[:1200])
EOF
ssh LabServer63 'grep -n "near\|EXTEND\|Ra\|Rs" ~/rest_x95a/results/logs/N1.log 2>/dev/null | head; ls ~/rest_x95a/results/logs/ 2>/dev/null'
  -> 2026-11-14T21:02:21Z CHAIN_START dev= commit=c152120 protocol=fb136ad2da79fdca
2026-11-14T21:02:21Z START S0
2026-11-14T21:02:31Z END S0 rc=0
2026-11-14T21:02:31Z STAGE0 PASS
2026-11-14T21:02:31Z START M1
2026-11-14T21:02:59Z END M1 rc=0
2026-11-14T21:02:59Z START R_p_r95a_p_00
2026-11-14T21:02:59Z  ...[cut]... 14536676723824436, "tol_E": 0.016299129522265487, "M_img": 0.005, "outcome": "S1_NO_HEADROOM"}
1:AN EXTEND
M1.log
M1.rc
M2.log
M2.rc
N1.log
N1.rc
N2.log
N2.rc
R_e_r95a_e_00.log
R_e_r95a_e_00.rc
R_e_r95a_e_01.log
R_e_r95a_e_01.rc
R_e_r95a_e_02.log
R_e_r95a_e_02.rc
R_e_r95a_e_03.log
R_e_r95a_e_03.rc
R_e_r95a_e_04.log
R_e_r95a_e_04.rc
R_e_r95a_e_05.log
R_e_r95a_e_05.rc
R_p_r95a_p_00.log
R_p_r95a_p_00.rc
R_p_r95a_p_01.log
R_p_r95a_p_01.rc
R_p_r95a_p_02.log
R_p_r95a_p_02.rc
R_p_r95a_p_03.log
R_p_r95a_p_03.rc
R_p_r95a_p_04.log
R_p_r95a_p_04.rc
R_p_r95a_p_05.log
R_p_r95a_p_05.rc
S0.log
S0.rc
gpu.txt
RUN [Check WL95A and WL96A chain status records] cd /c/Projects/ReST-GI && cat results/ecro_attr95a/chain.status; echo; cat results/ecro_util96a/chain.status; echo; cat results/ecro_util96a/final.json; python - <<'EOF'
import json
s=json.load(open('results/ecro_util96a/s1.json'))
print({k:(v if not isinstance(v,(dict,list)) else '...') for k,v in s.items()})
print(json.dumps(s.get('decision', s.get('stage1', {})))[:1200])
EOF
ssh LabServer63 'grep -n "near\|EXTEND\|Ra\|Rs" ~/rest_x95a/results/logs/N1.log 2>/dev/null | head; ls ~/rest_x95a/resu
RUN [Read the flight-attribution analysis code and the deliv.npz layout] sed -n 440,482p experiments/ecro_attr95a/a95.py | cut -c1-260; python - <<'EOF'
import numpy as np
z=np.load('results/ecro_attr95a/deliv.npz')
for k in z.files: print(k, z[k].shape, z[k].dtype)
EOF
  -> dr_ = z[f'draws|{k}']
                D[f'pool|{k}|{mn}'] = dmetrics(dr_.mean(0), ref, ra, rb, mm)
                per = [dmetrics(x, ref, ra, rb, mm) for x in dr_]
                D[f'pool|{k}|{mn}']['per_draw'] = {q: {'mean': float(np.mean([p[q] for p in per])), 'median': float(np.median([p[q] for ...[cut]... at64
dep|A2 (1270,) float64
dep|A3 (1270,) float64
dep|A4 (1270,) float64
dep|A0_s8601 (1270,) float64
dep|A0_s8602 (1270,) float64
dep|M64_spool (1270,) float64
dep|q256 (1270,) float64
dep|X1c (1270,) float64
dep|X1 (1270,) float64
draws|A1 (64, 1270) float64
draws|A2 (64, 1270) float64
draws|A3 (64, 1270) float64
draws|A4 (64, 1270) float64
draws|A0_s8601 (64, 1270) float64
draws|A0_s8602 (64, 1270) float64
draws|M64_spool (64, 1270) float64
draws|q256 (64, 1270) float64
draws|X1c (64, 1270) float64
draws|X1 (64, 1270) float64
ref_a (1270,) float64
ref_b (1270,) float64
RUN [Read the flight-attribution analysis code and the deliv.npz layout] cd /c/Projects/ReST-GI && sed -n 440,482p experiments/ecro_attr95a/a95.py | cut -c1-260; python - <<'EOF'
import numpy as np
z=np.load('results/ecro_attr95a/deliv.npz')
for k in z.files: print(k, z[k].shape, z[k].dtype)
EOF
WRITE c:\Projects\ReST-GI\experiments\audit97\B\wl95a_flight_sensitivity.py
RUN [Recompute the WL95A flight attribution at 16 and 64 draws and under alternative E0] python experiments/audit97/B/wl95a_flight_sensitivity.py results/ecro_attr95a/deliv.npz results/audit97/B/wl95a_flight_sensitivity.json
  -> n0 0.05926934169469287
A4_single_draw_abs_rel_l1 {"deployed_86001": 0.5237329397555786, "draws_mean": 0.7614863909731039, "median": 0.7261669512279727, "min": 0.28821329346766844, "p10": 0.4990639065522027, "p90": 1.1483439683082886, "draws_below_deployed": 11, "n_draws": 64}
A4_flux_ratio {"deplo ...[cut]... 'E_X1_pool': 0.0513, 'threshold_0.25E0': 0.1756, 'Rs_counts': True, 'Ra_counts': True, 'verdict': 'MIXED'}
alt_E0_median_single_draw 
    D16 {'D': 16, 'E0': 0.6541, 'Rs': 0.1887, 'Rs_se': 0.0767, 'Ra': 0.2005, 'Ra_se': 0.0951, 'Rp': 0.0478, 'Rp_se': 0.1189, 'E_X1_pool': 0.2171, 'threshold_0.25E0': 0.1635, 'Rs_counts': True, 'Ra_counts': True, 'verdict': 'MIXED'}
    D64 {'D': 64, 'E0': 0.6669, 'Rs': 0.2266, 'Rs_se': 0.0341, 'Ra': 0.2319, 'Ra_se': 0.0466, 'Rp': 0.1571, 'Rp_se': 0.0601, 'E_X1_pool': 0.0513, 'threshold_0.25E0': 0.1667, 'Rs_counts': True, 'Ra_counts': True, 'verdict': 'MIXED'}
WRITE c:\Projects\ReST-GI\experiments\audit97\B\wl95a_flight_sensitivity.py
RUN [Recompute the WL95A flight attribution at 16 and 64 draws and under alternative E0] cd /c/Projects/ReST-GI && python experiments/audit97/B/wl95a_flight_sensitivity.py results/ecro_attr95a/deliv.npz results/audit97/B/wl95a_flight_sensitivity.json
RUN [Check the WL87 E3 choice and its training-disjointness disclosure] python -c "
import json
c=json.load(open('results/gm1_explicit87/results/e3_choice.json')); print({k:(v if k!='candidates' else list(v.keys())) for k,v in c.items()})
" | cut -c1-600; grep -n -i "E3.*exclu\|exclu.*E3\|not excluded\|training geometr" docs/worklogs/87_WORKLOG_*.md | cut -c1-400 | head -8; grep -n "e3\b\|E3" experiments/gm1_explicit87/protocol/gm1_explicit87.json | grep -i "train\|exclu\|disjoint\|margin" | cut -c1-500 | head
  -> {'chosen': 'E3a', 'candidates': ['E3a'], 'rule': {'regions': ['A', 'C', 'D', 'E'], 'min_pixels': 1000, 'also': 'pt86 geometry admissibility (scene86.geom_ok for T0 and T1) and light_ok; the first admissible candidate in e3_order is E3; geometry / visibility only (masks86.compute), no radiance of any candidate before the choice is recorded'}, 'note': 'geometry / visibility only; no radiance of any E3 candidate exists'}
389:  "admission": "WL86 stage0 criteria Q1-Q4, Q6, Q7 (analyze86 qualify logic) on E3; a non-admitted E3 is reported and excluded from the fidelity verdicts (no stop)"
RUN [Check the WL87 E3 choice and its training-disjointness disclosure] cd /c/Projects/ReST-GI && python -c "
import json
c=json.load(open('results/gm1_explicit87/results/e3_choice.json')); print({k:(v if k!='candidates' else list(v.keys())) for k,v in c.items()})
" | cut -c1-600; grep -n -i "E3.*exclu\|exclu.*E3\|not excluded\|training geometr" docs/worklogs/87_WORKLOG_*.md | cut -c1-400 | head -8; grep -n "e3\b\|E3" experiments/gm1_explicit87/protocol/gm1_explicit87.json | grep -i "train\|exclu\|disjoint\|margin" | cut -c1-500 | head
RUN [Check WL87 disclosure of E3 training proximity] grep -n -i "untouched\|margin\|held-out\|disjoint\|0\.15 m\|0\.3 m" docs/worklogs/87_WORKLOG_*.md | cut -c1-360 | head -14; python - <<'EOF'
import json
d=json.load(open('results/audit97/B/heldout_disjointness.json'))
for c in ('E3a','E4(=E3b)','CK2'):
    for p in ('P86_all1024','P92A_928'):
        r=d['conditions'][c][p]; print(c,p,'box viol',len(r['box_violations']),'light viol',len(r['light_violations']),'files',r['n_files_violating'],'min box',round(r['min_box_m'],3),'min light',round(r['m
  -> 77:  - realisation reproducibility; RR expectation; seed disjointness.
96:- **Untouched condition E3.**
338:### G. ARCHITECTURE VERDICT (scope: one room family, rigid box edits, three frozen ER materials, point / cone sources, 512², E1 / E2 historical, E3 untouched)
E3a P86_all1024 box viol 1 light viol 24 files 25 min box 0.129 min light 0.089
E3a P92A_928 box viol 1 light viol 21 files 22 min box 0.129 min light 0.089
E4(=E3b) P86_all1024 box viol 12 light viol 85 files 96 min box 0.061 min light 0.051
E4(=E3b) P92A_928 box viol 0 light viol 0 files 0 min box 0.153 min light 0.302
CK2 P86_all1024 box viol 4 light viol 16 files 20 min box 0.095 min light 0.105
CK2 P92A_928 box viol 4 light viol 13 files 17 min box 0.095 min light 0.105
RUN [Check WL87 disclosure of E3 training proximity] cd /c/Projects/ReST-GI && grep -n -i "untouched\|margin\|held-out\|disjoint\|0\.15 m\|0\.3 m" docs/worklogs/87_WORKLOG_*.md | cut -c1-360 | head -14; python - <<'EOF'
import json
d=json.load(open('results/audit97/B/heldout_disjointness.json'))
for c in ('E3a','E4(=E3b)','CK2'):
    for p in ('P86_all1024','P92A_928'):
        r=d['conditions'][c][p]; print(c,p,'box viol',len(r['box_violations']),'light viol',len(r['light_violations']),'files',r['n_files_violating'],'min box',round(r['min_box_m']
RUN [Diff experiment code between each freeze and completion commit] chk(){ echo "== $1 freeze $2 -> complete $3"; git diff --stat $2 $3 -- experiments/$1 | tail -8; }; chk ecro86 7786d09 b97027d; chk ecro86a 46593f0 bd81442; chk ecro_attr92 5e672ff 9212a55; chk ecro_own92a 7247d30 ce87188; chk ecro_cf92b b29d8ff 95ee276; chk ecro_stage93 c30c797 56be753; chk ecro_stage93a 888ceb6 ab68553; chk ecro_stage93b 03fbc84 0da2d11; chk ecro_stage93c cab6e18 943decc; chk ecro_local93d 4f3fc56 94e13a0; chk ecro_mat94 e347904 f3b431f; chk ecro_mat94r 5b7c0c8 b34bdf9; chk ec
  -> == ecro86 freeze 7786d09 -> complete b97027d
 experiments/ecro86/analyze86.py             |  3 ++
 experiments/ecro86/diag_v3.py               | 37 +++++++++++++++++
 experiments/ecro86/protocol/amendment1.json | 11 +++++
 experiments/ecro86/run86.sh                 |  9 +++++
 experiments/ecro86/te ...[cut]... 36 insertions(+)
== ecro_mat94r freeze 5b7c0c8 -> complete b34bdf9
== ecro_exp94b freeze 5c0a6cd -> complete 2990a2d
 experiments/ecro_exp94b/figb4_posthoc.py | 22 +++++++++++
 experiments/ecro_exp94b/posthoc94b.py    | 64 ++++++++++++++++++++++++++++++++
 2 files changed, 86 insertions(+)
== ecro_q94c freeze 6d8b254 -> complete 487f83e
== ecro_s94d freeze a08c9bf -> complete 6c721fc
== ecro_attr95a freeze c152120 -> complete 30e9c37
== ecro_util96a freeze a1718f7 -> complete efcf86a
 experiments/ecro_util96a/figw3_posthoc.py | 29 +++++++++++++++++++++++++++++
 1 file changed, 29 insertions(+)
RUN [Diff experiment code between each freeze and completion commit] cd /c/Projects/ReST-GI && chk(){ echo "== $1 freeze $2 -> complete $3"; git diff --stat $2 $3 -- experiments/$1 | tail -8; }; chk ecro86 7786d09 b97027d; chk ecro86a 46593f0 bd81442; chk ecro_attr92 5e672ff 9212a55; chk ecro_own92a 7247d30 ce87188; chk ecro_cf92b b29d8ff 95ee276; chk ecro_stage93 c30c797 56be753; chk ecro_stage93a 888ceb6 ab68553; chk ecro_stage93b 03fbc84 0da2d11; chk ecro_stage93c cab6e18 943decc; chk ecro_local93d 4f3fc56 94e13a0; chk ecro_mat94 e347904 f3b431f; chk ecro_mat9
RUN [Compare deployed server code hashes with the frozen commits] h(){ printf "%s %s@%s\n" "$(git show $1:$2 | sha256sum | cut -c1-16)" "$2" "$1"; }; h c40226d experiments/ecro_own92a/own92a.py; h c40226d experiments/ecro_own92a/run92a.sh; h c40226d experiments/ecro86/train86.py; h c40226d experiments/ecro86/model86.py; h c40226d experiments/ecro86/scene86.py;  h e347904 experiments/ecro_mat94/c94.py; h e347904 experiments/ecro_mat94/mat94.py; h e347904 experiments/ecro_mat94/an94.py; h 6d8b254 experiments/ecro_q94c/q94.py; h a08c9bf experiments/ecro_s94d/s94.
  -> 6c372e0360371853 experiments/ecro_own92a/own92a.py@c40226d
27741e9e727dc7ac experiments/ecro_own92a/run92a.sh@c40226d
340180b3a06ef20a experiments/ecro86/train86.py@c40226d
72a728b81e490fb7 experiments/ecro86/model86.py@c40226d
b75a2d9cbae1f738 experiments/ecro86/scene86.py@c40226d
35a98c29a03d4180  ...[cut]... Z INPUTS 2fb8d5e0058cb85d b2f642a92276afee 995c0e070521c734 64d20a1eb5cb78a5 a28e192d4d5dbbbb 85dbadfbe316a1f0 fc7aa53bbee1eab3 751388c99ef66e99 b6d5312eea95f13a ee7fea53af6eedea 
2026-11-13T00:57:06Z START S0_excl
2026-11-14T10:44:00Z CHAIN_START dev= commit=e347904 protocol=99e199f3847939e8 R86=/home/nd/rest_ecro86 R92A=/home/nd/rest_x92a R92B=/home/nd/rest_x92b R93D=/home/nd/rest_x93d RCK=/home/nd/rest_x93d_ck2
2026-11-14T10:44:00Z START A_pre
2026-11-11T02:23:51Z CHAIN_START dev= commit=7786d09 + amendment1 63972c8
2026-11-11T02:23:51Z RESUME amendment=1 commit=7786d09 + amendment1 63972c8
RUN [Compare deployed server code hashes with the frozen commits] cd /c/Projects/ReST-GI && h(){ printf "%s %s@%s\n" "$(git show $1:$2 | sha256sum | cut -c1-16)" "$2" "$1"; }; h c40226d experiments/ecro_own92a/own92a.py; h c40226d experiments/ecro_own92a/run92a.sh; h c40226d experiments/ecro86/train86.py; h c40226d experiments/ecro86/model86.py; h c40226d experiments/ecro86/scene86.py;  h e347904 experiments/ecro_mat94/c94.py; h e347904 experiments/ecro_mat94/mat94.py; h e347904 experiments/ecro_mat94/an94.py; h 6d8b254 experiments/ecro_q94c/q94.py; h a08c9bf 
RUN [Check whether the WL86 server code differs from the freeze only by line endings] ssh LabServer63 'cd ~/rest_ecro86/code; cat COMMIT; for f in gen86 scene86 train86 model86 pt86; do printf "%s %s %s\n" "$(tr -d "\r" < experiments/ecro86/$f.py | sha256sum | cut -c1-16)" "$f" "$(file -b experiments/ecro86/$f.py | cut -c1-60)"; done; sha256sum experiments/ecro86/protocol/ecro86.json | cut -c1-16; tr -d "\r" < experiments/ecro86/protocol/ecro86.json | sha256sum | cut -c1-16'
  -> 7786d09 + amendment1 63972c8
c093d1c5f14aa5a7 gen86 Python script, ASCII text executable, with CRLF line termina
b75a2d9cbae1f738 scene86 Python script, ASCII text executable, with CRLF line termina
340180b3a06ef20a train86 Python script, ASCII text executable, with CRLF line termina
72a728b81e490fb7 model86 Python script, ASCII text executable, with CRLF line termina
912e68acd314d79e pt86 Python script, ASCII text executable, with CRLF line termina
5ec57d62e9e82cca
ecca67246ef013d7
RUN [Detail the training files that violate the CK2 exclusion rule] python - <<'EOF'
import json, numpy as np
p=json.load(open('results/audit97/B/population_regen.json'))
d=json.load(open('results/audit97/B/heldout_disjointness.json'))['conditions']['CK2']['P92A_928']
c=json.load(open('results/audit97/B/panel_coverage.json'))
rows={(r['file'],r['light']):r for r in c['rows']}
items={i['file']:i for i in p['train']['items']}
ckA=np.array([-1.08,0.6,1.0]); ckB=np.array([1.4,1.5,0.5]); ckL=np.array([-1.45,-0.6,1.7])
print('CK2 ring0_mirror_to_box',c['CK2']['E_ring0
  -> CK2 ring0_mirror_to_box 17.54289788368394 E_panel 92.55935119513936
BOX violations
g00361.npz dA 3.061 dB 0.118 A.c [ 1.11 -1.15  2.22] A.h [0.45 0.22 0.26]
g00603.npz dA 2.577 dB 0.095 A.c [ 0.88 -0.99  1.51] A.h [0.39 0.43 0.45]
g00883.npz dA 1.924 dB 0.114 A.c [ 0.31 -0.65  1.47] A.h [0.38 0. ...[cut]... 5 pos [-1.253 -0.699  1.828] ring0 share 0.452 cos_max 0.479 E_panel 215.2 mirror->box(ring0) 0.0 dA 2.71
g00756.npz 2 d 0.271 pos [-1.52  -0.761  1.494] ring0 share None cos_max None E_panel 0.0 mirror->box(ring0) 0.0 dA 2.01
g00797.npz 0 d 0.191 pos [-1.467 -0.417  1.648] ring0 share 1.0 cos_max 0.248 E_panel 251.3 mirror->box(ring0) 0.0 dA 0.64
g00956.npz 1 d 0.185 pos [-1.533 -0.714  1.58 ] ring0 share None cos_max None E_panel 0.0 mirror->box(ring0) 0.0 dA 2.09
g01017.npz 3 d 0.145 pos [-1.443 -0.462  1.656] ring0 share 0.768 cos_max 0.312 E_panel 261.2 mirror->box(ring0) 8.44 dA 0.48
RUN [Detail the training files that violate the CK2 exclusion rule] cd /c/Projects/ReST-GI && python - <<'EOF'
import json, numpy as np
p=json.load(open('results/audit97/B/population_regen.json'))
d=json.load(open('results/audit97/B/heldout_disjointness.json'))['conditions']['CK2']['P92A_928']
c=json.load(open('results/audit97/B/panel_coverage.json'))
rows={(r['file'],r['light']):r for r in c['rows']}
items={i['file']:i for i in p['train']['items']}
ckA=np.array([-1.08,0.6,1.0]); ckB=np.array([1.4,1.5,0.5]); ckL=np.array([-1.45,-0.6,1.7])
print('CK2 ring0_mirror
RUN [Quantify how close training boxes come to the CK2 box-near-panel configuration] python - <<'EOF'
import json, numpy as np
p=json.load(open('results/audit97/B/population_regen.json'))
ex={e[0] for e in json.load(open('results/ecro_own92a/results/exclusion.json'))['excluded']['train']}
ckA=np.array([-1.08,0.6,1.0]); ckh=np.array([0.2,0.45,0.45])
it=[i for i in p['train']['items'] if i['file'] not in ex]
dA=np.array([np.linalg.norm(np.array(i['geom']['A']['c'])-ckA) for i in it])
print('nearest box-A centre to CK2 A:',np.sort(dA)[:5], [it[j]['file'] for j in np.argsort(dA)[:5]
  -> nearest box-A centre to CK2 A: [0.26940742 0.31973296 0.34116591 0.36393803 0.38408279] ['g00272.npz', 'g00166.npz', 'g00392.npz', 'g00194.npz', 'g00650.npz']
box A min corner x <= -1.30 (within 0.2 m of panel plane): 21 of 928 ; <= -1.2: 69
... and centre within the panel y/z extent: 6 11
box B same: 12 30
A_half range 0.20004190054065135 0.4498869480828429 CK2 h [0.2  0.45 0.45] ; frac items with max(h)>=0.44: 0.0980603448275862  min(h)<=0.21: 0.13469827586206898
RUN [Quantify how close training boxes come to the CK2 box-near-panel configuration] cd /c/Projects/ReST-GI && python - <<'EOF'
import json, numpy as np
p=json.load(open('results/audit97/B/population_regen.json'))
ex={e[0] for e in json.load(open('results/ecro_own92a/results/exclusion.json'))['excluded']['train']}
ckA=np.array([-1.08,0.6,1.0]); ckh=np.array([0.2,0.45,0.45])
it=[i for i in p['train']['items'] if i['file'] not in ex]
dA=np.array([np.linalg.norm(np.array(i['geom']['A']['c'])-ckA) for i in it])
print('nearest box-A centre to CK2 A:',np.sort(dA)[:5], [it[j]['file'] f
RUN [Read the WL92B label reconciliation recorded in WL93] sed -n 14,32p docs/worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md | cut -c1-1300; grep -n -i "fall-through\|written rule\|H_interact\|outcome C\|code path\|mapping" docs/worklogs/92B_WORKLOG_*.md | cut -c1-500 | head -20
  -> - Source encoding, ranks / widths, ray budget, ER materials and WL86 thresholds are unchanged.
  - No explicit GI, no source-class handling, no adoption, no follow-up batch.
- **Where.** Branch `Selective_Recompute`, LabServer63 (main). Code `experiments/ecro_stage93/`; results `results/ecro_stage93 ...[cut]... he dev chain re-derived: official H_interact SUPPORTED / outcome C; written rule INCONCLUSIVE / D (fall-through).

**Materiality for WL93.** Not material:
- No measurement, counterfactual value, E5 admission or E5 criterion changes.
40:- H_interact: no dominant counterfactual, but at least two with f ≥ 0.25 for both OWN seeds in a case.
55:**Outcome mapping.**
74:- **Outcome by the frozen mapping: C (multiple interacting limitations).** H_interact and H_geometry are SUPPORTED, and E5 was admitted.
141:| H_interact | SUPPORTED | no dominant local / global counterfactual, several with f ≥ 0.25 |
RUN [Read the WL92B label reconciliation recorded in WL93] cd /c/Projects/ReST-GI && sed -n 14,32p docs/worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md | cut -c1-1300; grep -n -i "fall-through\|written rule\|H_interact\|outcome C\|code path\|mapping" docs/worklogs/92B_WORKLOG_*.md | cut -c1-500 | head -20
RUN [Verify the WL92B rule implementation mismatch] sed -n 100,118p experiments/ecro_cf92b/analyze92b.py | cut -c1-330; python -c "
import json
r=json.load(open('results/ecro_stage93/recon92b.json')); print(json.dumps(r)[:1200])"
  -> if L:
        gv = [L['geometry_variation'][n]['E2_T0->E2_T1']['panel'] for n in OWN]
        H['H_geometry'] = {'label': 'SUPPORTED' if min(gv) >= 0.10 else ('NOT SUPPORTED' if max(gv) < 0.03 else 'INCONCLUSIVE'), 'G_panel_E2': gv,
                           'seed_variation_panel_E2_T1': L['seed_va ...[cut]... dom = not (H_local or H_global SUPPORTED)  (only X_M12 and X_L3 checked)", "dominant_counterfactuals_both_E2_cases": {"X_L2": true, "X_L3": false, "X_M1": false, "X_M2": false, "X_M12": false, "X_Q": true, "X_SHQ": false}, "interaction_clause": true, "H_interact_official": "SUPPORTED", "H_interact_written_rule": "INCONCLUSIVE", "outcome_official": "C", "outcome_written_rule": "D (code fall-through; the written mapping has no clause for exactly one non-dominance hypothesis SUPPORTED)", "measurements_affected": false, "note": "WL92B files are not modified; this record is the WL93 clarification"}
RUN [Verify the WL92B rule implementation mismatch] cd /c/Projects/ReST-GI && sed -n 100,118p experiments/ecro_cf92b/analyze92b.py | cut -c1-330; python -c "
import json
r=json.load(open('results/ecro_stage93/recon92b.json')); print(json.dumps(r)[:1200])"
RUN [Verify reference and operator file hashes and reference provenance] sha256sum results/wl95_calc/S1_ops.npz results/ecro_attr95a/ops95a.npz results/ecro_util96a/T96A.npz results/ecro_util96a/s1.json results/ecro_util96a/tails.json results/ecro_util96a/final.json | cut -c1-16,65-; ssh LabServer63 'cd ~; sha256sum rest_x94/results/fields/CKF_none.npz rest_x94/results/pre/masks94.npz rest_x92/results/nmf.npz wl95_calc/S1_ops.npz 2>&1 | cut -c1-16,65-; ls rest_x94/results/renders | grep REF_ER_CK_none | head; ls rest_x94/results/fields; python3 - <<EOF
import numpy a
  -> 8654ae794a9a61f0 *results/wl95_calc/S1_ops.npz
d4d02e5a2f423ce9 *results/ecro_attr95a/ops95a.npz
8019cb39d03421dd *results/ecro_util96a/T96A.npz
92f5e763126d83b6 *results/ecro_util96a/s1.json
395f0166deedeb5b *results/ecro_util96a/tails.json
1c625581262250e4 *results/ecro_util96a/final.json
3a6238a8 ...[cut]... 5927785639283fe19cb09fbed0e1c7072192db9ff0431cc1dcb1bcbd0b11
REF_ER_CK_bx38_CK2_T1_r0 2f4496a79cb67bb4bb067a4e7f60a5ec17018e64d970381e5e68388c50a2e413
REF_ER_CK_bx38_CK2_T1_r1 077128bf5a27e3d1a1041cc45605c403baa29192ec05f6f2cbbb01a01fb4c0e2
REF_ER_CK_bx38_CK2_T1_r2 aa2d0e1a244c5708acf74e0aab541117ccfec3d46f75b20cd7ba073396e10a53
REF_ER_CK_bx38_CK2_T1_r3 2fba528d9595b217d4a2c584527609fa0ba1dd4c8ae8a04b905224b2606ee290
REF_ER_CK_bx75_CK2_T1_r0 27eb519825ebd997b90e1ffca63ce61e5adadea461d3e620b69d23bf93a9d0d7
REF_ER_CK_bx75_CK2_T1_r1 4199b79749c501895266c85b8f54bc68d5429aed8fdbf8d1f5be0f51c316c1b7
RUN [Verify reference and operator file hashes and reference provenance] cd /c/Projects/ReST-GI && sha256sum results/wl95_calc/S1_ops.npz results/ecro_attr95a/ops95a.npz results/ecro_util96a/T96A.npz results/ecro_util96a/s1.json results/ecro_util96a/tails.json results/ecro_util96a/final.json | cut -c1-16,65-; ssh LabServer63 'cd ~; sha256sum rest_x94/results/fields/CKF_none.npz rest_x94/results/pre/masks94.npz rest_x92/results/nmf.npz wl95_calc/S1_ops.npz 2>&1 | cut -c1-16,65-; ls rest_x94/results/renders | grep REF_ER_CK_none | head; ls rest_x94/results/fields; pyth
RUN [Read the WL86 F2 gate numbers] grep -n "F2\|0\.063\|0\.072\|noise\|floor" docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md | cut -c1-420 | head -30
  -> 31:- 1,270 surface patches (25 cm grid on floor / ceiling / walls, 20 cm on the panel, 3 × 3 per box face, 8 × 4 on the sphere) × 64 equal-solid-angle bins of the patch-local hemisphere, piecewise constant.
47:- **Fixture.** A semi-enclosed 4 × 4 × 3 m room (open towards the camera): ER floor (B), s ...[cut]... C (0.063 / 0.072 against 0.06). Chain status `STOPPED_GENERALIZATION_NOT_SUPPORTED`. The field-level evidence does not show memorisation; the failing class is sharp receivers |
269:| PHYSICAL FIDELITY | **NOT SUPPORTED** as declared (F1 and F2 on both edits) | F1 passes everywhere (0.004–0.006); F2 passes on E1 and on E2's A and D; fails on E2's C. ER's own gap g is reported separately |
271:| END-TO-END ECONOMICS | **NOT TESTED** (gated by Stage 3) | Descriptive: 9× per-frame GPU and 3–6× edit-to-first-frame wall at 0.05 / 0.03 with region-matched errors; g not reached; bias floor in E2's C |
RUN [Read the WL86 F2 gate numbers] cd /c/Projects/ReST-GI && grep -n "F2\|0\.063\|0\.072\|noise\|floor" docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md | cut -c1-420 | head -30
RUN [Read the WL96 ring-allocation calculation and related worklog lines] python -c "
import json
r=json.load(open('results/wl96_calc/ring_alloc.json')); print(json.dumps(r)[:1800])"; sed -n 50,62p docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-900
  -> {"rank_MP_ring0_c0": 16, "rank_MP_ring0_c1": 16, "rank_MP_ring0_c2": 16, "ring0_nonneg_rank16_exact_max_rel": 0.0, "ck2_support_bins": [12, 13, 14, 15], "ck2_support_rank_needed": 4, "A1": {"ring0_op_l1": 0.9729860855120802, "ring1_op_l1": 0.8609341607205565, "ring2_op_l1": 0.681923190234801, "ring3 ...[cut]... .97 vs ring 3 at 0.40 (`results/wl96_calc/ring_alloc.json`).
  - [MEAS-S, physics critic] A CK2-incidence-weighted (light-conditioned) rank-16 operator reaches 0.062 on CK2 but 0.85 / 0.69 on two other lights.
  - [INF] So the verdict is about fit-distribution allocation and light-independence, not rank capacity. It is also **moot for design**: an exact rank-16 copy of M64 inherits M64's +42–54 % binned-outgoing bias.
- **ANGULAR_PROJECTION_EVIDENCED.**
  - [MEAS-S, d4] Holding the exact position but snapping the outgoing direction to the bin centre reproduces the over-delivery: 1.423 / 0.527.
RUN [Read the WL96 ring-allocation calculation and related worklog lines] cd /c/Projects/ReST-GI && python -c "
import json
r=json.load(open('results/wl96_calc/ring_alloc.json')); print(json.dumps(r)[:1800])"; sed -n 50,62p docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-900
RUN [Read the WL93 identifiability gate raw values] python -c "
import json
r=json.load(open('results/ecro_stage93/ident.json')); print(json.dumps(r)[:1500])
p=json.load(open('results/ecro_stage93/posthoc_ident.json')); print(list(p.keys())); print(json.dumps(p)[:800])"
  -> {"val": {"8601": {"irreducible_panel_rel_l1": 0.5079254258260072, "own_learned_panel_rel_l1": 0.9703785788791728}, "8602": {"irreducible_panel_rel_l1": 0.5079254258260072, "own_learned_panel_rel_l1": 0.9658061080344419}}, "E2": {"E2_T1": {"irreducible_panel_rel_l1": 0.36236614845336473, "own_learned ...[cut]... _vs_independent_q": 0.2902603094313158, "implied_single_q_noise": 0.1451301547156579, "enc_vs_mean_of_two_q": 0.4958965630379395, "enc_over_q_panel_flux": 1.0047673451087384}, "per_geometry": [{"idx": 0, "enc": 0.3488959410122642, "own8601": 0.95233455672646, "own8602": 0.9265507238257646, "q_vs_q2": 0.04085843227188047, "panel_q_sum": 40200.26829004288}, {"idx": 1, "enc": 0.350863794637135, "own8601": 1.0104599154987572, "own8602": 0.9798633947224233, "q_vs_q2": 0.038465702547940454, "panel_q_sum": 31040.992020606995}, {"idx": 2, "enc": 0.35831989797281527, "own8601": 1.073069025368685, "own8
RUN [Read the WL93 identifiability gate raw values] cd /c/Projects/ReST-GI && python -c "
import json
r=json.load(open('results/ecro_stage93/ident.json')); print(json.dumps(r)[:1500])
p=json.load(open('results/ecro_stage93/posthoc_ident.json')); print(list(p.keys())); print(json.dumps(p)[:800])"
RUN [Inspect the WL86A post-freeze change to the event-split tracer] git diff 46593f0 bd81442 -- experiments/ecro86a/pt86a.py | cut -c1-240; python -c "
import json
for a in ('amendment1','amendment2'):
    d=json.load(open(f'experiments/ecro86a/protocol/{a}.json')); print(a, json.dumps(d)[:1500]); print()"
  -> diff --git a/experiments/ecro86a/pt86a.py b/experiments/ecro86a/pt86a.py
index a5fd288..19eb1fa 100644
--- a/experiments/ecro86a/pt86a.py
+++ b/experiments/ecro86a/pt86a.py
@@ -70,8 +70,9 @@ class Src:
             self.tb = U(np.ascontiguousarray(idx.ravel())); self.te = PT.fl(val); self.tc = PT.fl ...[cut]... asurement": "for Lambert receiver patches and panel senders, the irradiance of the 2-event family computed (BIN) through the deployed state rays from the path-traced once-scattered radiance of the panel averaged per (patch, outgoing bin), and (RAY) through the same rays from the exact radiance at each hit point; compared with the path-traced truth (SPLIT e2_p) by the importance-weighted reading of the region and at patch level (patches carrying > 0.2 % of the region importance)", "rules": {"ADEQUATE": "BIN and RAY both reproduce the truth's importance-weighted reading within 20 % and with patc
RUN [Inspect the WL86A post-freeze change to the event-split tracer] cd /c/Projects/ReST-GI && git diff 46593f0 bd81442 -- experiments/ecro86a/pt86a.py | cut -c1-240; python -c "
import json
for a in ('amendment1','amendment2'):
    d=json.load(open(f'experiments/ecro86a/protocol/{a}.json')); print(a, json.dumps(d)[:1500]); print()"
RUN [Verify the WL94 selection against the exclusion rules] python - <<'EOF'
import json
s=json.load(open('results/ecro_mat94/select94.json'))
for sp in ('train','val'):
    f=[r['file'] for r in s['splits'][sp]['files']]
    print(sp,len(f),f[0],f[-1],'rejected_by_ck2',s['splits'][sp]['rejected_by_ck2_rule_before_quota'])
print(s['T_LEAK'])
ex={e[0] for e in json.load(open('results/ecro_own92a/results/exclusion.json'))['excluded']['train']}
tr=[r['file'] for r in s['splits']['train']['files']]
allf=[f'g{i:05d}.npz' for i in range(int(tr[-1][1:6])+1)]
mi
  -> train 96 g00000.npz g00104.npz rejected_by_ck2 2
val 24 g00000.npz g00025.npz rejected_by_ck2 0
{'pass': True, 'train_ids_6_7': 0, 'train_nonuniform_rects': 0, 'ck2_rule_violations': 0, 'train_val_overlap_names': 22, 'note': 'train and val are different WL86 splits (seeds 86100 / 86200); every selected file passed the WL92A exclusion and the CK2 rule'}
skipped among first 105 : [('g00012.npz', 'E4excl'), ('g00024.npz', 'E4excl'), ('g00026.npz', 'CK2rule'), ('g00035.npz', 'E4excl'), ('g00042.npz', 'CK2rule'), ('g00051.npz', 'E4excl'), ('g00069.npz', 'E4excl'), ('g00096.npz', 'E4excl'), ('g00098.npz', 'E4excl')]
RUN [Verify the WL94 selection against the exclusion rules] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
s=json.load(open('results/ecro_mat94/select94.json'))
for sp in ('train','val'):
    f=[r['file'] for r in s['splits'][sp]['files']]
    print(sp,len(f),f[0],f[-1],'rejected_by_ck2',s['splits'][sp]['rejected_by_ck2_rule_before_quota'])
print(s['T_LEAK'])
ex={e[0] for e in json.load(open('results/ecro_own92a/results/exclusion.json'))['excluded']['train']}
tr=[r['file'] for r in s['splits']['train']['files']]
allf=[f'g{i:05d}.npz' for i in ran
RUN [Hash audit scripts and outputs and check the working tree] sha256sum experiments/audit97/B/*.py results/audit97/B/* | cut -c1-16,65- && file experiments/audit97/B/*.py | cut -c1-120 && git status --short | head
  -> f19496efd89aba17 *experiments/audit97/B/heldout_disjointness.py
423417a642852a9e *experiments/audit97/B/learned_s1_albedo.py
f84b9f3f2d46b4bd *experiments/audit97/B/panel_coverage.py
4612d046b45f45b8 *experiments/audit97/B/regen_population.py
c66e9ea671fa2645 *experiments/audit97/B/server_digest_pop ...[cut]... /B/wl95a_flight_sensitivity.json
experiments/audit97/B/heldout_disjointness.py:     Python script, ASCII text executable
experiments/audit97/B/learned_s1_albedo.py:        Python script, ASCII text executable
experiments/audit97/B/panel_coverage.py:           Python script, ASCII text executable
experiments/audit97/B/regen_population.py:         Python script, ASCII text executable
experiments/audit97/B/server_digest_population.py: Python script, ASCII text executable
experiments/audit97/B/wl95a_flight_sensitivity.py: Python script, ASCII text executable
?? docs/audits/
?? experiments/audit97/
RUN [Hash audit scripts and outputs and check the working tree] cd /c/Projects/ReST-GI && sha256sum experiments/audit97/B/*.py results/audit97/B/* | cut -c1-16,65- && file experiments/audit97/B/*.py | cut -c1-120 && git status --short | head
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
