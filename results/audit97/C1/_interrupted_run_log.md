# Track C1: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List relevant experiment, result and worklog directories] ls docs/audits 2>/dev/null; ls experiments | grep -E "ecro|gm1|audit" ; ls results | grep -E "ecro|gm1|audit|wl9"; ls docs/worklogs | grep -E "^(8[6-9]|9[0-9])"
RUN [Size up metric modules and list an94 functions] wc -l experiments/ecro_mat94/*.py experiments/ecro86/*.py experiments/ecro_attr92/*.py 2>/dev/null | tail -60; grep -n "def W\|def eh\|def \|^W\b\|W =" experiments/ecro_mat94/an94.py | head -80
  -> ecro86
ecro86a
ecro_attr92
ecro_attr95a
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
ecro86
ecro86a
ecro_ ...[cut]... TION.md
93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md
94A_WORKLOG_ECRO_MATERIAL_FIDELITY_RETENTION_SMOKE.md
94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
94C_WORKLOG_ECRO_PANEL_MEDIATED_QUERY_SUPERVISION_SMOKE.md
94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md
94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md
96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
  -> 379 experiments/ecro_mat94/an94.py
   310 experiments/ecro_mat94/c94.py
   578 experiments/ecro_mat94/mat94.py
    36 experiments/ecro_mat94/posthoc94.py
   168 experiments/ecro_mat94/viz94.py
   248 experiments/ecro86/analyze86.py
    37 experiments/ecro86/diag_v3.py
   179 experiments/ecro86/econ8 ...[cut]... esponse(tab, F, v, comp_m, comp_r, P_):
94:def bin_err(X, Fv, P_, binmask):
101:def bin_group_amp(tab, F, v, arm, P_, binmask):
103:    def eg(X):
112:def table_metrics(tab, F, M, arms):
155:def strip(o):
162:def lum(x):
166:def load_ref(R, v):
177:def load_hl(R, v, model, arm):
182:def eh(h_reps, x_reps, tot, m):
192:def img_response(Rv, R0, Xv, X0, m):
201:def image_metrics(R, M, model, arms):
221:def gate_noise(tabm):
226:def mr_ok(tm, arm='C0'):
236:def ssr(im, eps=0.005):
243:def stageB(R):
273:def pop_metrics(R, otag):
290:def stageC(R, sd):
333:def labels(B, Cs):
358:def final(R, outd):
RUN [List relevant experiment, result and worklog directories] cd /c/Projects/ReST-GI && ls docs/audits 2>/dev/null; ls experiments | grep -E "ecro|gm1|audit" ; ls results | grep -E "ecro|gm1|audit|wl9"; ls docs/worklogs | grep -E "^(8[6-9]|9[0-9])"
RUN [Size up metric modules and list an94 functions] cd /c/Projects/ReST-GI && wc -l experiments/ecro_mat94/*.py experiments/ecro86/*.py experiments/ecro_attr92/*.py 2>/dev/null | tail -60; grep -n "def W\|def eh\|def \|^W\b\|W =" experiments/ecro_mat94/an94.py | head -80
RUN [Read an94 metric module] cat experiments/ecro_mat94/an94.py
  -> """WL94 analysis: frozen Stage B metrics and gate, Stage C fit / outcome mapping per seed, final hypothesis labels, figures.

  python an94.py stageB <R>              -> results/an/stageB.json (prints the gate only)
  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints F ...[cut]...  'final.json'))
    import viz94
    files = viz94.figures(R, outd, B, Cs, L)
    man = []
    for f in files:
        man.append({'file': os.path.basename(f), 'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest()})
    jd({'worklog': 94, 'files': man, 'source_root': R}, os.path.join(outd, 'manifest.json'))
    print('FINAL', json.dumps(L), flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if a[0] == 'stageB':
        stageB(a[1])
    elif a[0] == 'stageC':
        stageC(a[1], a[2])
    elif a[0] == 'final':
        final(a[1], a[2])
RUN [Read an94 metric module] cd /c/Projects/ReST-GI && cat experiments/ecro_mat94/an94.py
RUN [Find angular bin layout definitions in production modules] grep -rn "NPHI\|NZ\b\|n_phi\|n_z\|NMU\|nphi\|nring\|N_RING\|NSEC" experiments/ecro86/*.py experiments/ecro86a/*.py experiments/gm1_explicit87/*.py | head -60
RUN [Find all angular weight constructions across batches] grep -rn "// NPHI\|//8\|// 8\|//16\|// 16\|np.arange(NB)\|arange(64)\|2 \* np.pi / NB\|2\*np.pi/NB\|WBIN\|cos_w\|bin_w\|W\[None" experiments/ecro*/ experiments/gm1*/ --include=*.py | grep -v "^experiments/audit97" | head -120
  -> experiments/ecro86/ecro86.py:157:    w = (ctx.G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None] * S.DOMEGA)[..., None]      # area x cos x solid angle
experiments/ecro86/model86.py:45:    z = ((k // KAPPA) // S.NPHI + us[2, 0]) / S.NZ
experiments/ecro86/pt86.py:154:        iz = dr.cl ...[cut]... g
experiments/gm1_explicit87/quad87.py:198:        self.W = np.zeros((6 + S.SPH_NZ, K, NB, 3), np.float64)
experiments/gm1_explicit87/quad87.py:206:        for iz in range(S.SPH_NZ):
experiments/gm1_explicit87/tests87.py:133:    tot = [float(q.W[6 + iz].sum(0).mean(-1).max()) for iz in range(S.SPH_NZ)]
experiments/gm1_explicit87/tests87.py:327:        w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
experiments/gm1_explicit87/xport87.py:28:NB, K, KAPPA, NZ, NPHI = S.NB, S.K, S.KAPPA, S.NZ, S.NPHI
experiments/gm1_explicit87/xport87.py:120:    iz = F(bn // NPHI)
  -> experiments/ecro86/ecro86.py:157:    w = (ctx.G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None] * S.DOMEGA)[..., None]      # area x cos x solid angle
experiments/ecro86a/analyze86a.py:46:        fw = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
expe ...[cut]... 5) / S.NZ
experiments/gm1_relation88/rel88.py:54:        zc = (np.arange(NB) // NPHI + 0.5) / NZ
experiments/gm1_relation88/rel88.py:72:    z = (bn // NPHI + (sz + 0.5) / sub) / NZ
experiments/gm1_relation88/rel88.py:461:        zc = (np.arange(NB) // NPHI + 0.5) / NZ
experiments/gm1_relation88/tests88.py:128:        zc = (np.arange(NB) // S.NPHI + 0.5) / S.NZ
experiments/gm1_relation89/rel89.py:103:            self.zc_t = torch.as_tensor((np.arange(NB) // NPHI + 0.5) / NZ, dtype=torch.float64, device=dev)
experiments/gm1_relation89/tests89.py:52:    zc = (np.arange(NB) // S.NPHI + 0.5) / S.NZ
RUN [Find angular bin layout definitions in production modules] cd /c/Projects/ReST-GI && grep -rn "NPHI\|NZ\b\|n_phi\|n_z\|NMU\|nphi\|nring\|N_RING\|NSEC" experiments/ecro86/*.py experiments/ecro86a/*.py experiments/gm1_explicit87/*.py | head -60
RUN [Find all angular weight constructions across batches] cd /c/Projects/ReST-GI && grep -rn "// NPHI\|//8\|// 8\|//16\|// 16\|np.arange(NB)\|arange(64)\|2 \* np.pi / NB\|2\*np.pi/NB\|WBIN\|cos_w\|bin_w\|W\[None" experiments/ecro*/ experiments/gm1*/ --include=*.py | grep -v "^experiments/audit97" | head -120
RUN [Trace weight and an94 usage across WL92-WL94D code] grep -n "bins_z\|bins_phi" experiments/ecro86/protocol/*.json | head; for d in ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d; do echo "== $d"; ls experiments/$d; grep -n "NPHI\|NZ\b\|flux_w\|an94\|AN\.\|\.W\b\|import an9\|as AN" experiments/$d/*.py | head -30; done
  -> experiments/ecro86/protocol/ecro86.json:47:  "bins_z": 4, "bins_phi": 16, "bins_note": "64 equal-solid-angle bins of the patch-local hemisphere (4 rings in cos(theta) x 16 azimuth sectors), piecewise constant",
== ecro_attr92
analyze92.py
diag92.py
fx92.py
interv92.py
protocol
run92.sh
tests92.py
vi ...[cut]... .load_hl(R, v, mt, a)
496:    E_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(x - href[v]))[cy, cx].ravel() for (mt, a, v), x in hl.items()]), 0.999))
497:    D_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(href[v] - href['none']))[cy, cx].ravel() for v in TEX]), 0.999))
503:            VZ.sq(ax[2, 0], AN.lum(href[v] - href['none'])[cy, cx], D_MAX, plt, 'REF texture response')
509:            VZ.sq(ax[1, j], AN.lum(x - href[v])[cy, cx], E_MAX, plt, 'signed error hl - h')
511:                VZ.sq(ax[2, j], AN.lum(x - hl[(mt, a, 'none')])[cy, cx], D_MAX, plt, 'texture response')
RUN [Trace weight and an94 usage across WL92-WL94D code] cd /c/Projects/ReST-GI && grep -n "bins_z\|bins_phi" experiments/ecro86/protocol/*.json | head; for d in ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_local93d ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d; do echo "== $d"; ls experiments/$d; grep -n "NPHI\|NZ\b\|flux_w\|an94\|AN\.\|\.W\b\|import an9\|as AN" experiments/$d/*.py | head -30; done
RUN [Trace every use of an94.W in WL94-94D code] sed -n 105,125p experiments/ecro_attr92/diag92.py; grep -n "W\b" experiments/ecro_mat94/mat94.py experiments/ecro_mat94/c94.py | grep -v "^.*#" | head -40; grep -n "AN\.W\|AN\.ex\|\bW\b" experiments/ecro_exp94b/b94.py experiments/ecro_q94c/q94.py experiments/ecro_mat94r/r94.py experiments/ecro_s94d/s94.py experiments/ecro_exp94b/posthoc94b.py | head -60
  -> for _ in range(iters):
            H *= (W.T @ V) / (W.T @ W @ H + eps)
            W *= (V @ H.T) / (W @ H @ H.T + eps)
        err = float(np.linalg.norm(V - W @ H))
        if best is None or err < best[0]:
            best = (err, W.copy(), H.copy())
    return best


def flux_w():
    """Mean c ...[cut]... ew(1, 1, -1, 1)).sum((2, 3)).cpu().numpy()
experiments/ecro_s94d/s94.py:157:    exm = (Sx * AN.W[None, :, None]).sum((1, 2)); ea = (zf['e2_p_a'] * AN.W[None, :, None]).sum((1, 2)); eb = (zf['e2_p_b'] * AN.W[None, :, None]).sum((1, 2))
experiments/ecro_s94d/s94.py:287:    W = torch.as_tensor(AN.W, device=dev, dtype=torch.float32).view(1, 1, -1, 1)
experiments/ecro_s94d/s94.py:300:        acc['l1'] += float(((Bl - Bm).abs() * W).sum()); acc['m'] += float((Bm * W).sum()); acc['l'] += float((Bl * W).sum())
experiments/ecro_s94d/s94.py:310:        el, em = (Dl * W).sum((2, 3)), (Dm * W).sum((2, 3))
RUN [Trace every use of an94.W in WL94-94D code] cd /c/Projects/ReST-GI && sed -n 105,125p experiments/ecro_attr92/diag92.py; grep -n "W\b" experiments/ecro_mat94/mat94.py experiments/ecro_mat94/c94.py | grep -v "^.*#" | head -40; grep -n "AN\.W\|AN\.ex\|\bW\b" experiments/ecro_exp94b/b94.py experiments/ecro_q94c/q94.py experiments/ecro_mat94r/r94.py experiments/ecro_s94d/s94.py experiments/ecro_exp94b/posthoc94b.py | head -60
RUN [Inventory local results for WL92-94D] for d in ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_local93d ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c; do echo "== $d $(du -sh results/$d | cut -f1)"; find results/$d -type f | sed 's/[0-9]\{4,\}/N/g' | awk -F/ '{print $3"/"$4"/"$NF}' | sed 's/_r[0-9]*\./_rK./' | sort | uniq -c | head -40; done 2>&1 | head -230
  -> == ecro_mat94 556K
      1 chain.status//chain.status
      1 final.json//final.json
      1 gpu.txt//gpu.txt
      1 models/A_sN.json/A_sN.json
      1 models/B_sN.json/B_sN.json
      1 pop/P_A_sN.json/P_A_sN.json
      1 pop/P_B_sN.json/P_B_sN.json
      1 pop/P_own_sN.json/P_own_sN.json
      1  ...[cut]... ogs/S1_ofield_LSPOT.rc
      1 results/logs/S1_split_LSPOT.log
      1 results/logs/S1_split_LSPOT.rc
      1 results/logs/S2_cf_E1_T1_LSPOT.log
      1 results/logs/S2_cf_E1_T1_LSPOT.rc
      1 results/logs/S2_cf_E2_T0.log
      1 results/logs/S2_cf_E2_T0.rc
      1 results/logs/S2_cf_E2_T1.log
      1 results/logs/S2_cf_E2_T1.rc
      1 results/logs/S2_cf_E4_T0.log
      1 results/logs/S2_cf_E4_T0.rc
      1 results/logs/S2_cf_E4_T1.log
      1 results/logs/S2_cf_E4_T1.rc
      1 results/logs/S2_local.log
      1 results/logs/S2_local.rc
== ecro_stage93 67K
      1 chain.status//chain.status
RUN [Inventory local results for WL92-94D] cd /c/Projects/ReST-GI && for d in ecro_mat94 ecro_mat94r ecro_exp94b ecro_q94c ecro_s94d ecro_local93d ecro_attr92 ecro_own92a ecro_cf92b ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c; do echo "== $d $(du -sh results/$d | cut -f1)"; find results/$d -type f | sed 's/[0-9]\{4,\}/N/g' | awk -F/ '{print $3"/"$4"/"$NF}' | sed 's/_r[0-9]*\./_rK./' | sort | uniq -c | head -40; done 2>&1 | head -230
RUN [Check server run directories for WL94 family] ssh -o ConnectTimeout=15 LabServer63 'ls -d ~/rest_x9* ~/rest_x97/* 2>/dev/null; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; for d in rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d; do echo "== $d"; ls ~/$d; ls ~/$d/results | head -30; done'
  -> /home/nd/rest_x92
/home/nd/rest_x92a
/home/nd/rest_x92adev
/home/nd/rest_x92b
/home/nd/rest_x92bdev
/home/nd/rest_x92dev
/home/nd/rest_x93
/home/nd/rest_x93a
/home/nd/rest_x93adev
/home/nd/rest_x93aprobe
/home/nd/rest_x93b
/home/nd/rest_x93bdev
/home/nd/rest_x93c
/home/nd/rest_x93cdev
/home/nd/rest_ ...[cut]... .json
tables
== rest_x94r
chain.out
chain.status
code
data
env.sh
results
an
eval94a
fields
lists
logs
models
pop
pre
renders
select94.json
tables
== rest_x94b
chain.out
chain.status
code
data
env.sh
results
an
audit
eval94b
fields
lists
logs
models
pop
pre
renders
select94.json
tables
== rest_x94c
chain.out
chain.status
code
data
env.sh
results
an
audit
eval94c
fields
lists
logs
manifest
models
pop
pre
renders
select94.json
tables
== rest_x94d
chain.out
chain.status
code
data
env.sh
results
an
audit
audit94d
eval94d
fields
lists
logs
manifest
models
pop
pre
renders
select94.json
sender
tables
RUN [Inspect WL94 raw artifact sizes on server] ssh LabServer63 'cd ~/rest_x94/results; du -sh tables fields renders pre pop an; ls -la tables fields pre pop an | head -60; ls renders | wc -l; ls renders | head -5; ls -la ~/rest_x94r/results/tables ~/rest_x94r/results/fields | head -30'
  -> 309M	tables
33M	fields
702M	renders
128K	pre
8.6M	pop
476K	an
an:
total 480
drwxr-xr-x  2 nd nd   4096 Nov 14 11:12 .
drwxr-xr-x 12 nd nd   4096 Nov 14 11:10 ..
-rw-r--r--  1 nd nd    273 Nov 14 11:10 final.json
-rw-r--r--  1 nd nd  52941 Nov 14 11:12 posthoc94.json
-rw-r--r--  1 nd nd 241991 Nov 14 ...[cut]... nd 68598643 Nov 14 12:21 CD_Ar_s8601.npz
lrwxrwxrwx  1 nd nd       47 Nov 14 12:18 CD_B_s8601.npz -> /home/nd/rest_x94/results/tables/CD_B_s8601.npz
-rw-r--r--  1 nd nd 68958260 Nov 14 12:21 CD_C_s8601.npz
-rw-r--r--  1 nd nd 68973804 Nov 14 12:26 CD_C_s8602.npz
lrwxrwxrwx  1 nd nd       49 Nov 14 12:18 FZ_own_s8601.npz -> /home/nd/rest_x94/results/tables/FZ_own_s8601.npz
lrwxrwxrwx  1 nd nd       49 Nov 14 12:18 FZ_own_s8602.npz -> /home/nd/rest_x94/results/tables/FZ_own_s8602.npz
-rw-r--r--  1 nd nd 68598369 Nov 14 12:19 RP_A_s8601.npz
-rw-r--r--  1 nd nd 69047592 Nov 14 12:19 RP_B_s8601.npz
RUN [Read WL94 worklog head] wc -l docs/worklogs/9[2-4]*.md && sed -n 1,80p docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
  -> 321 docs/worklogs/92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md
   259 docs/worklogs/92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBUTION.md
   296 docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md
   303 docs/worklogs/93A_WORKLOG_ECRO_SOURCE_STAGE_TARGET_RELIABIL ...[cut]... zen)

**Stage B: frozen counterfactuals.** CK2 geometry, light and camera are unchanged. Six arrangements on box A:

| variant | arrangement | what it tests |
|---|---|---|
| none | default (0.5) | baseline |
| U07 | all 0.7 | the walls' training-vocabulary state on new patches |
| U03 | all 0.3 | new value |
| bx75 | checker 0.7 / 0.5 | vocabulary values |
| bx38 | checker 0.3 / 0.8 | new values |
| bx38i | bx38 inverted | identical value histogram, permuted layout |

- **References.**
  - Patch × bin field: 2 × 4,096 spp, with common random numbers across variants.
  - Images: 4 × 8,192 spp.
RUN [Read WL94 worklog head] cd /c/Projects/ReST-GI && wc -l docs/worklogs/9[2-4]*.md && sed -n 1,80p docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
RUN [Read WL94 design and results] sed -n 80,260p docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
  -> - Images: 4 × 8,192 spp.
- **Arms.**
  - C0, C2_src, C2_rcv, C2_both (WL93D);
  - new: C2_srcS1, which uses M64 at the S1 panel sender only (isolates the sharp sender).
- **Metrics.** Log-response amplitude (slope) and location (corr) of S, G and B, measured over:
  - box-A patches;
  - the panel-fa ...[cut]... ox is 59–91 % wrong, independent of the receiver.
   - Substituting the exact panel sender removes 34 % (s8601; at the noise limit) and 54 % (s8602) of the untextured ROI error.
   - The error sits in what the sharp panel delivers to the box (sender lobe and panel-mediated global term). It does **not** sit in the box's outgoing directional bins.

## Interpretation

**Physical correctness.** Material routing, reference proportionality, reproduction of WL93D and the addendum, and reference S/N all hold. The Stage B reference differences are therefore valid targets.

**Model behaviour (frozen).**
RUN [Read WL94 design and results] cd /c/Projects/ReST-GI && sed -n 80,260p docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
RUN [Inspect raw table/mask/field keys on server] ssh LabServer63 'mkdir -p ~/rest_x97/C1 && cd ~/rest_x97/C1 && . ~/rest_x95a/env.sh && $PY - <<EOF
import numpy as np
z=np.load("/home/nd/rest_x94/results/tables/FZ_own_s8601.npz")
for k in list(z.keys())[:40]: print(k, z[k].shape, z[k].dtype)
print(len(z.keys()))
m=np.load("/home/nd/rest_x94/results/pre/masks94.npz")
for k in m.keys(): print(k, m[k].shape, m[k].dtype, m[k].sum())
f=np.load("/home/nd/rest_x94/results/fields/CKF_none.npz")
for k in f.keys(): print(k, f[k].shape, f[k].dtype)
p=np.
  -> none|G (1270, 64, 3) float32
none|S_C0 (1270, 64, 3) float32
none|S_C2_src (1270, 64, 3) float32
none|S_C2_rcv (1270, 64, 3) float32
none|S_C2_both (1270, 64, 3) float32
none|S_C2_srcS1 (1270, 64, 3) float32
none|S_C2_srcNonS1 (1270, 64, 3) float32
none|S_C0_fromS1 (1270, 64, 3) float32
none|S_C0_fr ...[cut]...  float64
ck|mS (24, 1270) float64
ck|mG (24, 1270) float64
ck|rB (24, 1270) float64
ck|rS (24, 1270) float64
ck|rG (24, 1270) float64
ck|rBa (24, 1270) float64
ck|rBb (24, 1270) float64
ck|rSa (24, 1270) float64
ck|rSb (24, 1270) float64
ck|rGa (24, 1270) float64
ck|rGb (24, 1270) float64
ck|cls (24, 1270) int64
ck|pmat (24, 1270) int64
hl (512, 512, 3) float32
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
C1
MAP
RUN [Read masks86 and masks94 construction] cat experiments/ecro86/masks86.py; grep -n "masks94\|T_panel_face\|ROI_T\|DIFF\|panel_facing_bins\|T_other" experiments/ecro_mat94/mat94.py | head -40
  -> """WL86 region masks from geometry and visibility only (no radiance, no candidate output, no reference error).

  python masks86.py <root> <E1|E2> [--ls=name] [--res=N]
Per pixel (centre ray), with T0 and T1 scenes of the edit and the lights of the workload:
  stationary  same surface point in T0 an ...[cut]... , 'T': tgt_p, 'T_panel_face': face == PANEL_FACE,
409:            'T_other_faces': tgt_p & (face != PANEL_FACE), 'panel_facing_bins': pfb, 'S1': G.PSID == 5, 'PSID': G.PSID.copy(),
554:    np.savez_compressed(os.path.join(R, 'results', 'pre', 'masks94.npz'), **{k: v for k, v in M.items() if k != 'DIFF_rule'})
555:    V = {'roi_pixels': int(M['ROI_T'].sum()), 'diff_pixels': int(M['DIFF'].sum()), 'diff_rule': M['DIFF_rule'],
556:         'panel_facing_bins_per_target_patch_mean': float(M['panel_facing_bins'][M['T']].sum(1).mean()),
557:         'panel_face_patches': int(M['T_panel_face'].sum())}
RUN [Read masks86 and masks94 construction] cd /c/Projects/ReST-GI && cat experiments/ecro86/masks86.py; grep -n "masks94\|T_panel_face\|ROI_T\|DIFF\|panel_facing_bins\|T_other" experiments/ecro_mat94/mat94.py | head -40
RUN [Read masks94 region construction] sed -n 335,412p experiments/ecro_mat94/mat94.py
  -> np.savez_compressed(os.path.join(out, f'{otag}.npz'), hl=PT.pack(acc, cam['res'], int(spp))['hl'])
        jd({'variant': v, 'src': src, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, os.path.join(out, f'{otag}.json'))
        print(otag, round(T_['wall_s'], 2), flush=True)
    print('RENDER_DON ...[cut]...  = hs1.reshape(G.N, S.NB, S.KAPPA).mean(2)
    pfb = (frac >= 0.5) & tgt_p[:, None]
    face = np.full(G.N, -1)
    for r in TARGET_RECTS:
        face[rect_patches(G, r)] = r
    return {'ROI_T': roi, 'DIFF': diff, 'DIFF_rule': rule, 'vf_boxA': vf.astype(np.float32), 'panel': a(w5), 'frame': front_, 'sid': sid_,
            'pid': pid_, 'mirror_patch': np.where(roi, mpid, -1), 'T': tgt_p, 'T_panel_face': face == PANEL_FACE,
            'T_other_faces': tgt_p & (face != PANEL_FACE), 'panel_facing_bins': pfb, 'S1': G.PSID == 5, 'PSID': G.PSID.copy(),
            'PMAT_default': default_pmat(G)}
RUN [Read masks94 region construction] cd /c/Projects/ReST-GI && sed -n 335,412p experiments/ecro_mat94/mat94.py
RUN [Read b94 audit code] sed -n 40,130p experiments/ecro_exp94b/b94.py
  -> # ------------------------------------------------------------------------------------------------ Stage A: split re-render
def agen(R, R86, which, start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = C94.sel_files(R, 'train'); asg = R94.assignments(len(files), ' ...[cut]... 'stable')
    H = np.zeros(n, bool); H[rank[:nh]] = True
    cnt = np.zeros(n, np.int64)
    for mask, budget in ((H, int(round(SIGMA * steps_half))), (~H, steps_half - int(round(SIGMA * steps_half)))):
        ids = np.where(mask)[0]
        ids = ids[np.argsort(-pm_half[ids], kind='stable')]
        q, r = divmod(budget, len(ids))
        cnt[ids] = q; cnt[ids[:r]] += 1
    return H, cnt


def d_items_expanded(names, counts):
    """the per-step copies in select order (what train() passes to c94.train)."""
    out = []
    for nm in names:
        out += [nm] * int(counts[nm])
    return out
RUN [Read b94 audit code] cd /c/Projects/ReST-GI && sed -n 40,130p experiments/ecro_exp94b/b94.py
RUN [Read analyze92 metric helpers] sed -n 1,140p experiments/ecro_attr92/analyze92.py
  -> """WL92 analysis by the prospective rules (protocol ecro_attr92: ladder, fits, identifiability, gate, verdict mapping). numpy only.

  python analyze92.py gate  <R> <R86>     stages S0-S5 -> results/core92.json, results/gate.json ; prints GATE ADMITTED | GATE STOP_INTERVENTION_<gate>
  python analyz ...[cut]... z'))
        q = D['q'].astype(np.float64)
        o['Bq_vs_q'] = {}
        for sd in SEEDS:
            Bq = T[f'Bq_{sd}'].astype(np.float64)
            o['Bq_vs_q'][sd] = {str(m): {'rel_l1': float(np.abs(Bq[G.PMAT == m] - q[G.PMAT == m]).sum() / max(q[G.PMAT == m].sum(), 1e-30)),
                                         'flux_ratio': float(Bq[G.PMAT == m].sum() / max(q[G.PMAT == m].sum(), 1e-30))} for m in range(6) if (G.PMAT == m).any() and q[G.PMAT == m].sum() > 0}
        o['field'] = {'split_sum_flux': float((L2T + L3T).sum()), 'L2_share_of_field': float(L2T.sum() / (L2T + L3T).sum()),
RUN [Read analyze92 metric helpers] cd /c/Projects/ReST-GI && sed -n 1,140p experiments/ecro_attr92/analyze92.py
RUN [Read analyze86 helpers] sed -n 1,110p experiments/ecro86/analyze86.py
  -> """WL86 analysis and prospective gates (protocol sections stage0 / stage1 / stage3).

  python analyze86.py qualify <root>     Stage 0: workload admission on REF-P (class shares on REF-ER), E1 and E2
  python analyze86.py stage1 <root>      Stage 1: higher-order identifiability and oracle-projection ...[cut]... ll', 'H_share_T1') >= G0['H_share_min'])}
        un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1'))
        q['Q7_uncertainty'] = {'value': un, 'PASS': bool(un <= G0['uncertainty_max'])}
        q['QUALIFIED'] = bool(all(v['PASS'] for k, v in q.items() if not k.startswith('Q5')))      # Q5 is a reported control class
        ok_all &= q['QUALIFIED']
        out[e] = {'regions': R, 'qualification': q}
    tj = os.path.join(root, 'results', 'tests', 'tests86.json')
    out['tests_pass'] = bool(json.load(open(tj))['PASS']) if os.path.exists(tj) else False
RUN [Read analyze86 helpers] cd /c/Projects/ReST-GI && sed -n 1,110p experiments/ecro86/analyze86.py
RUN [Read WL94 interpretation and limits] sed -n 260,374p docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
  -> **Model behaviour (frozen).**
- The under-response of the reflected checker decomposes into two parts.
  - Material-response compression, which is generic. It appears in Lambert receiver and global terms, and in sharp and diffuse readers alike.
  - A separate sharp-coupled baseline error at the pane ...[cut]... 차를 줄일 수 있는지는 확인하지 못했습니다(규칙상 중단, seed 8602 미실행).
- 패널 면 전역 항의 악화가 예산 부족 때문인지 용량 trade-off 때문인지 분리하지 못했습니다.
- SSR 판정이 시드 간에 불일치합니다.
- DIFF bx38 / bx38i는 노이즈 한계입니다.

**6. Architecture Verdict (아키텍처 판정)**
- **H-M: INCONCLUSIVE.** 원인은 INCONCLUSIVE_SMOKE_FIT이고, 재질 응답 회복은 서술적으로 관찰됐습니다. 이는 짧은 학습의 적합 실패이지 표현 클래스의 실패가 아닙니다.
- **H-A: INCONCLUSIVE.** SSR은 s8602만 통과했습니다. 표현 클래스 한계는 시험하지 않았습니다.
- 채택, 아키텍처 변경, 후속 배치는 없습니다. 결정은 owner에게 돌려드립니다.

**7. Reproducibility (재현성)**
- freeze `e347904`, LabServer63 `~/rest_x94`, 27분 소요, 그림은 `results/evaluation/94/`.
- 체크포인트 해시, 시드, 재질 맵 다이제스트는 위 Reproducibility 절에 있습니다.
RUN [Read WL94 interpretation and limits] cd /c/Projects/ReST-GI && sed -n 260,374p docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
RUN [Read s94 sender metric code] sed -n 95,200p experiments/ecro_s94d/s94.py
  -> dev = dev_(); Mq = mq_t()
    T, G = {}, {}
    files = C94.sel_files(R, 'train')
    z0 = S.Geom(json.loads(str(np.load(os.path.join(R86, 'data', 'train', files[0]['file']))['geom'])))
    pan = MT.rect_patches(z0, 5)
    T['T_INDEX'] = {'pass': bool(np.array_equal(pan, np.array(PANEL)) and (z0.PSI ...[cut]... ch.as_tensor(it.it.s, device=dev)
            for key in ('own', 'T', 'R'):
                m.zero_grad(set_to_none=True)
                F = m.factors(gs_)
                if key == 'own':
                    t = it.tgt(dev, 'f'); t2 = it.tgt(dev, 'f2')
                    Q.CUR['w'] = it.qw
                    loss = Q.wobjective(m.apply(F, gs_, s_, use_global=False), m.apply(F, gs_, s_, use_single=False), t, t2)[0]
                else:
                    Bl, Bm = sender_pair(F, s_, gs_.mat, Mq, masks_of(it)[key])
                    loss = teacher_loss(Bl, Bm); l0[key].append(float(loss))
RUN [Read s94 sender metric code] cd /c/Projects/ReST-GI && sed -n 95,200p experiments/ecro_s94d/s94.py
RUN [Read s94 teacher loss and sender evaluation code] sed -n 40,94p experiments/ecro_s94d/s94.py; sed -n 270,345p experiments/ecro_s94d/s94.py
  -> def mq_t():
    import torch
    return torch.as_tensor(MT.mq_ext().astype(np.float32), device=dev_())          # [11, NB_in, NB_out, 3]


def sender_pair(F, s, mat, Mq, mask):
    """Bl (learned) and Bm (exact) on the patches in mask: [L, n, NB, 3] each."""
    import torch
    c, a = F['c'][mask], ...[cut]... n op.items()}}
        PT, S_, A, FX, ctx, lt = MT.ctx_ck2(R)
        st = EC.build_state(ctx)
        s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
        MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
        gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
        F = model.factors(gs)
        acc = {k: 0.0 for k in ('l1', 'm', 'l', 'pl', 'pm', 'dl1', 'dm', 'dl', 'bl1', 'bm')}
        accumulate(acc, F, gs, s)
        op = operator(F)
        out['ck2'] = {**field_metrics(acc), 'operator_rel_l1': {k: (v[0] / v[1] if v[1] > 0 else None) for k, v in op.items()}}
RUN [Read s94 teacher loss and sender evaluation code] cd /c/Projects/ReST-GI && sed -n 40,94p experiments/ecro_s94d/s94.py; sed -n 270,345p experiments/ecro_s94d/s94.py
RUN [Check patch index ranges of box A and box B and hard-coded ranges in code] ssh LabServer63 'cd ~/rest_x97/C1 && . ~/rest_x95a/env.sh && $PY - <<EOF
import numpy as np
m=np.load("/home/nd/rest_x94/results/pre/masks94.npz")
T=np.nonzero(m["T"])[0]; print("T", T.min(), T.max(), len(T))
tp=np.nonzero(m["T_panel_face"])[0]; print("Tpf", tp)
ps=m["PSID"]
for s in np.unique(ps): 
    i=np.nonzero(ps==s)[0]; print("sid",s,i.min(),i.max(),len(i), "contig", (i.max()-i.min()+1)==len(i))
print("PMAT_default", {int(k):int((m["PMAT_default"]==k).sum()) for k in np.unique(m["PMAT_def
  -> T 1130 1183 54
Tpf [1130 1131 1132 1133 1134 1135 1136 1137 1138]
sid 0 0 255 256 contig True
sid 1 256 511 256 contig True
sid 2 512 703 192 contig True
sid 3 704 895 192 contig True
sid 4 896 1087 192 contig True
sid 5 1088 1129 42 contig True
sid 6 1238 1269 32 contig True
sid 7 1130 1183 54 cont ...[cut]... 9)
experiments/ecro_attr95a/a95.py:590:            ax[i, f_].imshow(v[1130 + 9 * f_:1139 + 9 * f_].reshape(3, 3), vmin=0, vmax=vmax, cmap='magma')
experiments/ecro_attr95a/wl95calc/wl95_calc.py:15:PANEL = np.arange(1088, 1130)
experiments/ecro_s94d/s94.py:30:PANEL = list(range(1088, 1130))
experiments/ecro_s94d/s94.py:309:        bx = torch.zeros(gs.N, dtype=torch.bool, device=dev); bx[1130:1238] = True
experiments/ecro_util96a/w96a.py:26:S1 = np.arange(1088, 1130)
experiments/ecro_util96a/w96a.py:27:BOXA = np.arange(1130, 1184)
experiments/ecro_util96a/w96a.py:28:PFACE = np.arange(1130, 1139)
RUN [Read WL94D worklog] sed -n 1,150p docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md
  -> # Worklog — WL94D: ECRO Sharp Sender Oracle-Supervised Smoke (2026-11-15)

## Origin and scope

- **Brief.** "WL94D — Sharp Sender Oracle-Supervised Smoke" (user). It is a bounded feasibility-to-smoke continuation of WL94C. The batch is a suffix of WL94 under the preserved ECRO contract and adds one ...[cut]... the rank-16 Eckart–Young floor (0.46–0.52, a different norm) and above WL92's per-geometry free-factor fits. The 1,000-step fit is therefore not near a representation limit; it is optimization- or generation-limited at this budget. The rank-16 bound itself is not tested here.
- R (the same oracle rule on non-S1 sources) behaves like P, so the T changes are specific to S1 supervision.
- The exact-sender control with these continuations stays at 0.017–0.035, close to the frozen-own exact-sender values (0.031 / 0.019). The gap of T to its own exact-sender value is 0.028 (s8601) and 0.079 (s8602).
RUN [Read WL94D worklog] cd /c/Projects/ReST-GI && sed -n 1,150p docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md
RUN [Show structure of final JSONs for WL94A-D] python - <<'EOF'
import json
def show(o, d=0, maxd=3, pre=''):
    if isinstance(o, dict):
        ks=list(o.keys())
        for k in ks[:14]:
            v=o[k]
            if isinstance(v,(dict,list)) and d<maxd:
                print(pre+str(k)+':'); show(v,d+1,maxd,pre+'  ')
            else:
                s=str(v); print(pre+str(k)+' = '+s[:90])
        if len(ks)>14: print(pre+'... +%d keys'%(len(ks)-14))
    elif isinstance(o,list):
        print(pre+'[list %d] %s'%(len(o), str(o[:3])[:
  -> ===== results/ecro_mat94r/final_r.json
outcome = INCONCLUSIVE
seeds:
  [list 2] [8601, 8602]
criteria:
  8601:
    A = {'F2': True, 'MR_table': False, 'MR_pop': False, 'MR': False, 'RET': True, 'RET_detail': {
    B = {'F2': False, 'MR_table': True, 'MR_pop': True, 'MR': True, 'RET': False, ' ...[cut]... 344404
  panel_specific_vs_E = MIXED
  sender_improved_vs_C = NO
  global_improved_vs_C = MIXED
  sharp_improved_vs_C = MIXED
  sharp_at_own_level = MIXED
  F2_D:
    8601 = True
    8602 = True
  MR_D:
    8601 = True
    8602 = True
  F2_E:
    8601 = True
    8602 = True
  MR_E:
    8601 = True
    8602 = True
seeds:
  [list 2] [8601, 8602]
===== results/ecro_s94d/final_s.json
outcome = NO_RELIABLE_IMPROVEMENT
per_seed:
  8601:
    SENDER_IMP = False
    RENDER_IMP = False
    KEEP_T = True
  8602:
    SENDER_IMP = False
    RENDER_IMP = False
    KEEP_T = True
RUN [Show structure of final JSONs for WL94A-D] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
def show(o, d=0, maxd=3, pre=''):
    if isinstance(o, dict):
        ks=list(o.keys())
        for k in ks[:14]:
            v=o[k]
            if isinstance(v,(dict,list)) and d<maxd:
                print(pre+str(k)+':'); show(v,d+1,maxd,pre+'  ')
            else:
                s=str(v); print(pre+str(k)+' = '+s[:90])
        if len(ks)>14: print(pre+'... +%d keys'%(len(ks)-14))
    elif isinstance(o,list):
        print(pre+'[list %d]
RUN [Read WL94A evaluation/gate code] sed -n 22,62p experiments/ecro_mat94r/r94.py; sed -n 108,186p experiments/ecro_mat94r/r94.py
  -> def assignments(n, cand):
    return ['A'] * n if cand in ('A', 'Ar') else ['A' if k % 2 == 0 else 'B' for k in range(n)]


def mix_items(R, R86, split, assign, limit=None):
    if split == 'train' and assign in ('C', 'Ar'):
        from concurrent.futures import ThreadPoolExecutor
        files = C ...[cut]...  'panel_face_G_signed': {'A': gA, 'B': gB, 'C': gC}}
    MT.jd(res, os.path.join(R, 'results', 'an', 'final_r.json'))
    files = figures(R, outd, rs)
    MT.jd({'worklog': '94A', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R},
          os.path.join(outd, 'manifest.json'))
    print('FINAL', oc, 'forgetting', fg, flush=True)


# ------------------------------------------------------------------------------------------------ figures
def figures(R, outd, rs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
RUN [Read WL94A evaluation/gate code] cd /c/Projects/ReST-GI && sed -n 22,62p experiments/ecro_mat94r/r94.py; sed -n 108,186p experiments/ecro_mat94r/r94.py
RUN [Read face metrics and population metric generation code] sed -n 76,107p experiments/ecro_mat94r/r94.py; grep -n "def face_ext" -A 28 experiments/ecro_exp94b/b94.py | head -60; grep -n "W\b\|ex(" experiments/ecro_mat94/c94.py | head -30
  -> import scene86 as S
    ok_dig = True; leak67 = 0; nonuni = 0; cnt = {'A': 0, 'B': 0}
    for rec, a in zip(files, asg):
        z = np.load(os.path.join(R, 'data', f'train_{a}', rec['file']))
        pm = z['pmat'].astype(np.int64)
        ok_dig &= str(z['digest']) == rec[f'digest_{a}'] == MT.pm_d ...[cut]... ise(F, v, kr, fm)}
344-    return o
345-
346-
347-def ref_split(R, M):
348-    zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
349-    r = {p: 0.5 * (zf[f'{p}_a'].astype(np.float64) + zf[f'{p}_b'].astype(np.float64)).sum(2) for p in PARTS}
350-    o = {}
351-    for fn, fm in (('panel_face', M['T_panel_face']), ('other_faces', M['T'] & ~M['T_panel_face'])):
352-        tot = sum(r[p][fm].sum() for p in PARTS)
353-        o[fn] = {p: float(r[p][fm].sum() / tot) for p in PARTS}
354-    return o
355-
356-
357-def model_list(seed):
358-    return ['own', 'A', 'C', 'D']
359-
360-
RUN [Read face metrics and population metric generation code] cd /c/Projects/ReST-GI && sed -n 76,107p experiments/ecro_mat94r/r94.py; grep -n "def face_ext" -A 28 experiments/ecro_exp94b/b94.py | head -60; grep -n "W\b\|ex(" experiments/ecro_mat94/c94.py | head -30
RUN [Find population metric weight usage in c94] grep -n "mB\|rBa\|AN\.\|cosw\|zb" experiments/ecro_mat94/c94.py | head -30
  -> 60:                pmB = MT.train_pmat_B(G, rec['idx'])
61:                rec['digest_A'] = MT.pm_digest(MT.default_pmat(G)); rec['digest_B'] = MT.pm_digest(pmB)
62:                leak['train_ids_6_7'] += int(np.isin(pmB, [6, 7]).sum())
63:                leak['train_nonuniform_rects'] += sum(int(len(np.unique(pmB[MT.rect_patches(G, r)])) > 1) for r in range(len(G.RMAT)))
64:                rec['B_albedos'] = {str(r): MT.ALB.get(int(pmB[MT.rect_patches(G, r)][0])) for r in MT.LAMBERT_RECTS}
269:        ex = {k: [] for k in ('mB', 'mS', 'mG', 'rB', 'rS', 'rG', 'rBa', 'rBb', 'rSa', 'rSb', 'rGa', 'rGb')}
282:                for k, x in (('mB', Bs + Bg), ('mS', Bs), ('mG', Bg), ('rB', t), ('rS', t2), ('rG', t - t2), ('rBa', fa), ('rBb', fb),
RUN [Find population metric weight usage in c94] cd /c/Projects/ReST-GI && grep -n "mB\|rBa\|AN\.\|cosw\|zb" experiments/ecro_mat94/c94.py | head -30
RUN [Read population metric generation] sed -n 255,300p experiments/ecro_mat94/c94.py
  -> z = (np.arange(S.NB) // S.NPHI + 0.5) / S.NZ
    return (z * S.DOMEGA).astype(np.float32)


def popeval(R, R86, R92A, spec, otag):
    import torch
    import scene86 as S
    dev = torch.device('cuda')
    model = MT.load_spec(R, R92A, spec)
    w = torch.as_tensor(exitance_w(S), device=dev).view(1 ...[cut]...   summ[assign] = {k: num[k] / den[k] for k in num}
        for k, v in ex.items():
            out[f'{assign}|{k}'] = np.stack(v).astype(np.float64)
        out[f'{assign}|cls'] = np.stack(cls)
        out[f'{assign}|pmat'] = np.stack([it.pm for it in items])
    fn = os.path.join(R, 'results', 'pop', f'{otag}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **out)
    MT.jd({'spec': spec, 'rel_l1': summ}, fn.replace('.npz', '.json'))
    print('POPEVAL', otag, flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
RUN [Read population metric generation] cd /c/Projects/ReST-GI && sed -n 255,300p experiments/ecro_mat94/c94.py
RUN [Write independent table-metric recompute script and copy to server] mkdir -p /c/Projects/ReST-GI/experiments/audit97/C1 /c/Projects/ReST-GI/results/audit97/C1 /c/Projects/ReST-GI/docs/audits/wl97_evidence_reliability && cd /c/Projects/ReST-GI/experiments/audit97/C1 && cat > c1_tables.py <<'EOF'
"""WL97 audit, track C1: independent recomputation of the WL94..WL94D CK2 table metrics from the stored raw tables / field references.

Own numpy implementation of the protocol definitions (exitance = sum_bins,channels X * cos * dOmega; log-response amplitude = least-squa
  -> Exit code 2
/usr/bin/bash: -c: line 128: unexpected EOF while looking for matching `''
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_tables.py
RUN [Copy script to server, list WL94B-D tables] cd /c/Projects/ReST-GI/experiments/audit97/C1 && file c1_tables.py && scp -q c1_tables.py LabServer63:rest_x97/C1/ && ssh LabServer63 'cd ~/rest_x97/C1 && file c1_tables.py; for d in rest_x94b rest_x94c rest_x94d; do echo == $d; ls -la ~/$d/results/tables ~/$d/results/fields ~/$d/results/pre | grep -v "^total" ; done'
  -> c1_tables.py: Python script, ASCII text executable
c1_tables.py: Python script, ASCII text executable
== rest_x94b
lrwxrwxrwx 1 nd nd   32 Nov 14 14:15 /home/nd/rest_x94b/results/fields -> /home/nd/rest_x94/results/fields

/home/nd/rest_x94b/results/pre:
drwxr-xr-x  2 nd nd   4096 Nov 14 14:17 .
drw ...[cut]... 38 Nov 14 16:35 CD_R_s8602.npz
-rw-r--r--  1 nd nd 68899128 Nov 14 16:35 CD_T_s8601.npz
-rw-r--r--  1 nd nd 68952517 Nov 14 16:35 CD_T_s8602.npz
lrwxrwxrwx  1 nd nd       49 Nov 14 16:30 FZ_own_s8601.npz -> /home/nd/rest_x94/results/tables/FZ_own_s8601.npz
lrwxrwxrwx  1 nd nd       49 Nov 14 16:30 FZ_own_s8602.npz -> /home/nd/rest_x94/results/tables/FZ_own_s8602.npz
-rw-r--r--  1 nd nd 68958460 Nov 14 16:31 RP_C_s8601.npz
-rw-r--r--  1 nd nd 68974013 Nov 14 16:31 RP_C_s8602.npz
-rw-r--r--  1 nd nd 68699747 Nov 14 16:31 RP_own_s8601.npz
-rw-r--r--  1 nd nd 68919295 Nov 14 16:31 RP_own_s8602.npz
RUN [Run table recompute on server CPU] ssh LabServer63 'cd ~/rest_x97/C1 && . ~/rest_x95a/env.sh && (time $PY c1_tables.py tables_recompute.json) 2>&1 | tail -30; ls -la tables_recompute.json'
  -> WL94|own_s8601 MR coded/true False False
WL94|own_s8602 MR coded/true False False
WL94|A_s8601 MR coded/true False False
WL94|B_s8601 MR coded/true True True
WL94A|C_s8601 MR coded/true True True
WL94A|C_s8602 MR coded/true True True
WL94A|A_s8602 MR coded/true False False
WL94A|Ar_s8601 MR coded/true False False
WL94C|CD_D_s8601 MR coded/true True True
WL94C|CD_D_s8602 MR coded/true True True
WL94C|CD_E_s8601 MR coded/true True True
WL94C|CD_E_s8602 MR coded/true True True
WL94D|CD_P_s8601 MR coded/true True True
WL94D|CD_P_s8602 MR coded/true True True
WL94D|CD_R_s8601 MR coded/true True True
WL94D|CD_R_s8602 MR coded/true True True
WL94D|CD_T_s8601 MR coded/true True True
WL94D|CD_T_s8602 MR coded/true True True

real	1m11.336s
user	1m10.936s
sys	0m4.124s
-rw-r--r-- 1 nd nd 2840141 Nov 15 17:50 tables_recompute.json
RUN [Verify coded-weight reimplementation reproduces stored WL94 Stage B JSON] scp -q LabServer63:rest_x97/C1/tables_recompute.json results/audit97/C1/ && python - <<'PYEOF'
import json, math
R=json.load(open('results/audit97/C1/tables_recompute.json'))['jobs']
# compare coded with stored stageB.json (WL94)
B=json.load(open('results/ecro_mat94/stageB.json'))
mx=0; n=0
for sd in ('s8601','s8602'):
    st=B['table'][sd]; mine=R[f'WL94|own_{sd}']['coded']
    for arm in st:
        if arm=='sender_split':
            for nm in st[arm]:
                for k,v in st[arm][nm].i
  -> Exit code 1
scp: open local "results/audit97/C1/": Not a directory
RUN [Verify coded-weight reimplementation reproduces stored WL94 Stage B JSON] cd /c/Projects/ReST-GI && scp -q LabServer63:rest_x97/C1/tables_recompute.json results/audit97/C1/ && python - <<'PYEOF'
import json, math
R=json.load(open('results/audit97/C1/tables_recompute.json'))['jobs']
# compare coded with stored stageB.json (WL94)
B=json.load(open('results/ecro_mat94/stageB.json'))
mx=0; n=0
for sd in ('s8601','s8602'):
    st=B['table'][sd]; mine=R[f'WL94|own_{sd}']['coded']
    for arm in st:
        if arm=='sender_split':
            for nm in st[arm]:
              
RUN [Check audit output directories] ls -la results/audit97/ 2>&1 | head; ls docs/audits 2>&1
  -> total 36
drwxr-xr-x 1 dna10 197609 0 Nov 16 02:46 .
drwxr-xr-x 1 dna10 197609 0 Nov 16 02:45 ..
drwxr-xr-x 1 dna10 197609 0 Nov 16 02:47 B
drwxr-xr-x 1 dna10 197609 0 Nov 16 02:50 C2
drwxr-xr-x 1 dna10 197609 0 Nov 16 02:47 MAP
wl97_evidence_reliability
RUN [Check audit output directories] cd /c/Projects/ReST-GI && ls -la results/audit97/ 2>&1 | head; ls docs/audits 2>&1
RUN [Create output dir and fetch recompute JSON] mkdir -p results/audit97/C1 && scp -q LabServer63:rest_x97/C1/tables_recompute.json results/audit97/C1/tables_recompute.json && ls -la results/audit97/C1
  -> total 2776
drwxr-xr-x 1 dna10 197609       0 Nov 16 02:50 .
drwxr-xr-x 1 dna10 197609       0 Nov 16 02:50 ..
-rw-r--r-- 1 dna10 197609 2840141 Nov 16 02:50 tables_recompute.json
RUN [Create output dir and fetch recompute JSON] cd /c/Projects/ReST-GI && mkdir -p results/audit97/C1 && scp -q LabServer63:rest_x97/C1/tables_recompute.json results/audit97/C1/tables_recompute.json && ls -la results/audit97/C1
NOTE: Now a comparison script (written as a file to avoid heredoc quoting problems).
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_compare_tables.py
RUN [Run comparison of coded vs stored and coded vs true weights] python experiments/audit97/C1/c1_compare_tables.py > results/audit97/C1/tables_compare.txt 2>&1; head -150 results/audit97/C1/tables_compare.txt
  -> IDENTITY WL94 stageB s8601: 1718 values compared, max abs diff 2.147e-07, mismatches > 1e-6: 0
IDENTITY WL94 stageB s8602: 1718 values compared, max abs diff 2.147e-07, mismatches > 1e-6: 0
IDENTITY WL94 stageC A: 1058 values compared, max abs diff 2.147e-07, mismatches > 1e-6: 0
IDENTITY WL94 st ...[cut]...                 coded -0.4506 true -0.5144 diff -0.0637
    C2_rcv|U07|S|T|corr                                          coded +0.0092 true +0.0711 diff +0.0619
    sender_split|C2_rcv_fromS1|U07|corr                          coded +0.0500 true +0.1039 diff +0.0539
    sender_split|C0_fromS1|U07|corr                              coded -0.0276 true +0.0239 diff +0.0514
    C2_rcv|U03|B|other_bins|corr                                 coded +0.0263 true -0.0248 diff -0.0512
WL94A|C_s8602
    C2_rcv|U07|S|T|corr                                          coded +0.0187 true -0.0447 diff -0.0635
NOTE: Now a comparison script (written as a file to avoid heredoc quoting problems).
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_compare_tables.py
RUN [Run comparison of coded vs stored and coded vs true weights] cd /c/Projects/ReST-GI && python experiments/audit97/C1/c1_compare_tables.py > results/audit97/C1/tables_compare.txt 2>&1; head -150 results/audit97/C1/tables_compare.txt
RUN [Show sender split coded vs true] sed -n '/^Sender split/,$p' results/audit97/C1/tables_compare.txt | cut -c1-330
  -> Sender split, untextured (C0_fromS1 = learned panel sender, learned receiver; C2_rcv_fromS1 = learned panel sender, exact receiver) vs e2_p
WL94|own_s8601         C0_fromS1        err_T coded 0.594 true 0.595 (signed true -0.463, half-noise true 0.059) | panel-face err true 0.584 (noise 0.070) | sha ...[cut]...       C0_fromS1        err_T coded 1.558 true 1.550 (signed true +1.483, half-noise true 0.059) | panel-face err true 1.325 (noise 0.070) | share coded 0.100 true 0.100 pf share true 0.591
WL94D|CD_T_s8602       C2_rcv_fromS1    err_T coded 1.593 true 1.584 (signed true +1.501, half-noise true 0.059) | panel-face err true 1.349 (noise 0.070) | share coded 0.100 true 0.100 pf share true 0.591
WL94D|CD_T_s8602       C0_fromOther     err_T coded 0.061 true 0.061 (signed true -0.027, half-noise true 0.002) | panel-face err true 0.072 (noise 0.009) | share coded 0.900 true 0.900 pf share true 0.409
RUN [Show sender split coded vs true] cd /c/Projects/ReST-GI && sed -n '/^Sender split/,$p' results/audit97/C1/tables_compare.txt | cut -c1-330
RUN [Find WL95/95A statements about W bug and unmatched norms] grep -n -i "zero-operator\|zero operator\|matched norm\|unmatched\|8x8\|8 x 8\|8×8\|an94.W\|Frobenius" docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md docs/worklogs/95A_*.md docs/worklogs/96*.md docs/worklogs/94D_*.md | cut -c1-700
  -> docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md:25:  - Confirmed the `an94.W` layout defect: `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`, whereas the protocol is 4 rings × 16 sectors (`experiments/ecro86/protocol/ecro86.json:47`; `experiments/ecr ...[cut]... ins) all bound a *linear, light-independent, separable, fixed-basis* form, in Frobenius norms that down-weight grazing rows.
docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md:109:- **S1 local operator vs M64** over all 64 incident bins (flux-transfer weighted rel L1): 0.96–1.04 for every model (val geometries and CK2), including T. For reference, WL92's best rank-16 approximation of M64 S1 leaves an Eckart–Young residual of 0.46–0.52 in the flux-Frobenius norm and keeps 47–54 % of the mirror-bin flux; the learned operators keep 21–33 % of the peak on the validation fields.
RUN [Find WL95/95A statements about W bug and unmatched norms] cd /c/Projects/ReST-GI && grep -n -i "zero-operator\|zero operator\|matched norm\|unmatched\|8x8\|8 x 8\|8×8\|an94.W\|Frobenius" docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md docs/worklogs/95A_*.md docs/worklogs/96*.md docs/worklogs/94D_*.md | cut -c1-700
RUN [Read WL95 section 2.1/2.2 known corrections] sed -n 205,250p docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-900; sed -n 150,200p docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-600
  -> - This ceiling is **empirical over the factorisations tried**, not a proof.
  - A peak-targeted rank-16 objective was not tested [HYP].
- [MEAS] Floor B is near-exact at rank 16: field L1 0.012–0.017; 0.065 at rank 8 (svd8). S2 is intermediate: 0.16–0.30 at rank 16, 0.14 at rank 32.
- [MEAS] Panel i ...[cut]... 742 | n/c |

All arm values without another citation are [MEAS, this calculation, `~/wl95_calc/wl95_calc*.json`]. n/c = not computed.

**What the table supports**
- [MEAS] The learned operators are 0.36–0.66 above admissible non-negative rank-16 operators in the same flux-L1 norm. At the field level, NMF16 beats learned T by about 0.31 in field L1 and about 0.11 in field peak, at comparable energy.
- [INF] So the rank-16 bound is **not** what currently binds the learned sender. Five candidate causes are **not** separated:
  - optimisation;
  - objective / normalisation (the WL94D teacher den);
RUN [Read WL95 section 2.1/2.2 known corrections] cd /c/Projects/ReST-GI && sed -n 205,250p docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-900; sed -n 150,200p docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-600
RUN [List mat94 functions and write sites] grep -n "^def \|^    def \|makedirs\|savez\|np.save\|jd(\|open(.*'w'" experiments/ecro_mat94/mat94.py | head -90
  -> 42:def jd(o, fn):
43:    os.makedirs(os.path.dirname(fn), exist_ok=True)
44:    json.dump(o, open(fn, 'w'), indent=1)
47:def jl(fn):
51:def sha_file(fn):
55:def lambert_state(alb):
66:def new_rows():
71:def default_pmat(G):
80:def rect_patches(G, r):
85:def variant_pmat(G, v):
99:def train_pmat_B(G, ...[cut]... t_ok=True)
335:        np.savez_compressed(os.path.join(out, f'{otag}.npz'), hl=PT.pack(acc, cam['res'], int(spp))['hl'])
336:        jd({'variant': v, 'src': src, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, os.path.join(out, f'{otag}.json'))
342:def masks(PT, S, ctx, cam):
414:def pre(R, R92A, R92B, R93D, RCK):
441:    def hemi(k):
554:    np.savez_compressed(os.path.join(R, 'results', 'pre', 'masks94.npz'), **{k: v for k, v in M.items() if k != 'DIFF_rule'})
560:    jd({'status': status, 'tests': T, 'geometry': V, 'patch_sha256': _INST['patch_sha'], 'scene_sha256': sha_file(SCENE_FN),
RUN [List mat94 functions and write sites] cd /c/Projects/ReST-GI && grep -n "^def \|^    def \|makedirs\|savez\|np.save\|jd(\|open(.*'w'" experiments/ecro_mat94/mat94.py | head -90
RUN [Read mat94 install/load/arm table code] sed -n 126,290p experiments/ecro_mat94/mat94.py
  -> def _patch_src(fn, obj, label, probe_k1=False):
    """probe_k1 (field_src only): at k = 1, when the hit lies on the probe patch's own shape it IS the probe point of patch pid0, so its
    material is PMAT[pid0] (patch_bin's cell assignment rounded a few probe points to a neighbouring cell at cell e ...[cut]... S_FROZEN)
            for k, x in t.items():
                out[f'{v}|{k}'] = x[0].cpu().numpy().astype(np.float32)
            for a in (arms or ARMS_FROZEN):
                out[f'{v}|B_{a}'] = out[f'{v}|S_{a}'] + out[f'{v}|G']
    fn = os.path.join(R, 'results', 'tables', f'{otag}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **out)
    print('TABLES', otag, len(out), flush=True)


# ------------------------------------------------------------------------------------------------ references / renders
def fref(R, v):
    PT, S, A, FX, ctx, lt = ctx_ck2(R)
RUN [Read mat94 install/load/arm table code] cd /c/Projects/ReST-GI && sed -n 126,290p experiments/ecro_mat94/mat94.py
RUN [Inspect WL94D run script and model files on server] ssh LabServer63 'cat ~/rest_x94d/code/experiments/ecro_s94d/run94d.sh | head -60; ls ~/rest_x94d/code/experiments | head -40; ls ~/rest_x94d/results/models ~/rest_x94d/results/sender ~/rest_x94d/results/audit94d ~/rest_x94d/data | head -60; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
  -> #!/bin/bash
# WL94D chain: ONE detached chain. setsid nohup bash run94d.sh <R> <W94> <W94A> <W94B> <W94C> <R86> <R92A> [--dev|--feas] > <R>/chain.out 2>&1 &
set -u
R=$1; W=$2; WA=$3; WB=$4; WC=$5; R86=$6; R92A=$7; MODE=${8:-}
DEV=; [ "$MODE" = --dev ] && DEV=--dev
. "$R/env.sh"
CODE=$R/code/experime ...[cut]... esults/models:
A_s8601.json
A_s8601.pt
A_s8602.json
A_s8602.pt
Ar_s8601.json
Ar_s8601.pt
B_s8601.json
B_s8601.pt
C_s8601.json
C_s8601.pt
C_s8602.json
C_s8602.pt
D_s8601.json
D_s8601.pt
D_s8602.json
D_s8602.pt
E_s8601.json
E_s8601.pt
E_s8602.json
E_s8602.pt
P_s8601.json
P_s8601.pt
P_s8602.json
P_s8602.pt
R_s8601.json
R_s8601.pt
R_s8602.json
R_s8602.pt
T_s8601.json
T_s8601.pt
T_s8602.json
T_s8602.pt

/home/nd/rest_x94d/results/sender:
SN_D_s8601.json
SN_D_s8602.json
SN_P_s8601.json
SN_P_s8602.json
SN_R_s8601.json
SN_R_s8602.json
SN_T_s8601.json
SN_T_s8602.json
SN_own_s8601.json
SN_own_s8602.json
RUN [Check production helpers for write side effects] cat experiments/ecro_attr92/fx92.py; grep -n "def build_state" -A 25 experiments/ecro86/ecro86.py | head -50; grep -n "def local_matrices" -A 22 experiments/ecro_attr92/diag92.py; grep -n "def load_own" -A 12 experiments/ecro_stage93/train93.py; ssh LabServer63 'head -3 ~/rest_x94d/chain.status; ps -ef | grep -c "[r]un9"; ls ~/rest_x97/'
  -> """WL92 configuration: imports the frozen WL86 / WL86A / WL87 modules unchanged and adds the prospectively defined condition E4.

E4 is chosen from the never-rendered WL87 candidates E3b / E3c by the geometry / visibility-only rule of the protocol and recorded in
<root>/results/e4_choice.json before ...[cut]...    m.load_state_dict(torch.load(os.path.join(R92A, 'results', 'models', f'own_s{seed}.pt'), map_location=dev))
49-    for p in m.parameters():
50-        p.requires_grad_(False)
51-    return m.eval()
52-
53-
54-def sd_hash(model):
55-    h = hashlib.sha256()
56-    for k, v in sorted(model.state_dict().items()):
57-        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
2026-11-14T16:30:53Z CHAIN_START mode= commit=a08c9bf protocol=e0a389463e172bc7 W94C=/home/nd/rest_x94c
2026-11-14T16:30:53Z START A_rp_own_s8601
2026-11-14T16:30:53Z START A_rp_C_s8601
0
B
C1
C2
D
E3
E4
MAP
RUN [Check production helpers for write side effects] cd /c/Projects/ReST-GI && cat experiments/ecro_attr92/fx92.py; grep -n "def build_state" -A 25 experiments/ecro86/ecro86.py | head -50; grep -n "def local_matrices" -A 22 experiments/ecro_attr92/diag92.py; grep -n "def load_own" -A 12 experiments/ecro_stage93/train93.py; ssh LabServer63 'head -3 ~/rest_x94d/chain.status; ps -ef | grep -c "[r]un9"; ls ~/rest_x97/'
RUN [Read c94 item loading and eval93b.flight] sed -n 96,170p experiments/ecro_mat94/c94.py; grep -n "def flight" -A 22 experiments/ecro_stage93b/eval93b.py | head -50
  -> def gen(R, R86, split, assign, start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = sel_files(R, split)[int(start):int(start) + int(count)]
    out = os.path.join(R, 'data', f'{split}_{assign}'); os.makedirs(out, exist_ok=True)
    ctx = None; t0 = time.time(); r ...[cut]... B, KAPPA, 3) * rw.view(1, 1, NB, KAPPA, 1)).sum(3)
157-
158-
159-def l1s(X, T):
160-    return [float(np.abs(X - T).sum()), float(T.sum())]
161-
162-
163-def xl2(X, A, B):
164-    """cross-product sums: sum (X - A)(X - B), sum A B (A, B independent halves)."""
165-    return [float(((X - A) * (X - B)).sum()), float((A * B).sum())]
166-
167-
168-def targets(R86, R92A, R, c):
169-    """-> t, t_a, t_b, t2, t2_a, t2_b as float64 numpy [L, N, NB, 3] (halves None if not available)."""
170-    if c['kind'] == 'sharp':
171-        D = np.load(os.path.join(R, 'results', 'fields', f"D_{c['tag']}.npz"))
RUN [Read c94 item loading and eval93b.flight] cd /c/Projects/ReST-GI && sed -n 96,170p experiments/ecro_mat94/c94.py; grep -n "def flight" -A 22 experiments/ecro_stage93b/eval93b.py | head -50
RUN [Find R86/R92A roots and stored sender JSON] ssh LabServer63 'ps -o args= -p 1 >/dev/null; grep -h "rest_" ~/rest_x94d/results/logs/S_own_s8601.log | head -3; ls ~/rest_x94d/results/logs | head -30; ls -d ~/rest_ecro86 ~/rest_x92a; cat ~/rest_x94d/results/sender/SN_T_s8601.json; sed -n 1,40p ~/rest_x94d/code/experiments/ecro_s94d/s94.py'
  -> A_audit.log
A_audit.rc
A_repro.log
A_repro.rc
A_rp_C_s8601.log
A_rp_C_s8601.rc
A_rp_C_s8602.log
A_rp_C_s8602.rc
A_rp_own_s8601.log
A_rp_own_s8601.rc
A_rp_own_s8602.log
A_rp_own_s8602.rc
E_pop_P_s8601.log
E_pop_P_s8601.rc
E_pop_P_s8602.log
E_pop_P_s8602.rc
E_pop_R_s8601.log
E_pop_R_s8601.rc
E_pop_R_s ...[cut]... ut>
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_q94c'))
import q94 as Q  # noqa: E402

B, MT, C94, AN, R94 = Q.B, Q.MT, Q.C94, Q.AN, Q.R94
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_s94d.json')))
DEV = MT.DEV
SEEDS = [8601, 8602]
STEPS = 60 if DEV else 1000
WARM, LR0, LR1 = 100, 1e-4, 1e-5
NLAM = 4 if DEV else 16
VAR, TEX = AN.VARIANTS, AN.TEX
PANEL = list(range(1088, 1130))


def jd(o, fn):
    AN.jd(o, fn)


def dev_():
    import torch
    return torch.device('cuda')
RUN [Read imports and model API] sed -n 1,40p experiments/ecro_mat94/mat94.py | cut -c1-200; sed -n 1,40p experiments/ecro_mat94/c94.py | cut -c1-200; grep -n "class GeoState" -A 14 experiments/ecro86/model86.py; grep -n "def factors\|def apply" -A 12 experiments/ecro86/model86.py | head -60; grep -n "class Item" -A 22 experiments/ecro86/train86.py
  -> """WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are a ...[cut]...    def gs(self):
27-        g = type('G', (), {})()
28-        g.N = self.tab['pc'].shape[0]
29-        g.PC, g.PN, g.PT, g.PAREA, g.PMAT = (self.tab[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat'))
30-        return M.GeoState(g, self.hit, self.dbin, dev)
31-
32-    def target(self):
33-        f = torch.as_tensor(self.f.astype(np.float32), device=dev)
34-        if self.fb is not None:
35-            f = 0.5 * (f + torch.as_tensor(self.fb.astype(np.float32), device=dev))
36-        return f
37-
38-
39-def load(root, split, limit=None):
40-    from concurrent.futures import ThreadPoolExecutor
RUN [Read imports and model API] cd /c/Projects/ReST-GI && sed -n 1,40p experiments/ecro_mat94/mat94.py | cut -c1-200; sed -n 1,40p experiments/ecro_mat94/c94.py | cut -c1-200; grep -n "class GeoState" -A 14 experiments/ecro86/model86.py; grep -n "def factors\|def apply" -A 12 experiments/ecro86/model86.py | head -60; grep -n "class Item" -A 22 experiments/ecro86/train86.py
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_sender_dump.py
RUN [Launch short GPU sender dump job detached on server] cd /c/Projects/ReST-GI/experiments/audit97/C1 && sed -i 's/, rw=None if False else np.zeros(1))/)/' c1_sender_dump.py && grep -n "hit=st.hit" c1_sender_dump.py && file c1_sender_dump.py && scp -q c1_sender_dump.py LabServer63:rest_x97/C1/ && ssh LabServer63 'cd ~/rest_x97/C1 && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && mkdir -p snd && . ~/rest_x95a/env.sh && export PYTHONDONTWRITEBYTECODE=1 && setsid nohup $PY c1_sender_dump.py ~/rest_x94d/code ~/rest_x94d ~/res
  -> 49:                    hit=st.hit, dbin=st.dbin)
c1_sender_dump.py: Python script, ASCII text executable
1274 MiB, 26 %
started
[This command modified 1 file you've previously read: experiments\audit97\C1\c1_sender_dump.py. Call Read before editing.]
RUN [Read WL94B audit/exposure gate code] sed -n 131,215p experiments/ecro_exp94b/b94.py
  -> def replay(seed, expanded):
    rng = np.random.default_rng(int(seed) + 94)
    order = rng.permutation(len(expanded)).tolist()
    return [expanded[order.pop()] for _ in range(C94.STEPS)]


def audit(R, R86):
    import scene86 as S
    files = C94.sel_files(R, 'train')[:N_AUD]; names = [r['file']  ...[cut]...  halves.values())
                and (DEV or int(counts.min()) >= 1) and all(s['replayed_counts_equal'] for s in sched.values()))
    T['T_ALLOC'] = {'pass': bool(ok_alloc), 'total': int(counts.sum()), 'orig': int(counts[0::2].sum()), 'div': int(counts[1::2].sum()),
                    'H_counts': sorted({int(c) for c in counts[Hm]}), 'generic_counts': sorted({int(c) for c in counts[~Hm]}),
                    'min_visits': int(counts.min())}
    # CK2 comparison (untextured field reference, box A mask)
    M = AN.masks(R)
    zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
RUN [Read WL94B audit/exposure gate code] cd /c/Projects/ReST-GI && sed -n 131,215p experiments/ecro_exp94b/b94.py
RUN [Read WL94B gate code and worklog results] sed -n 215,282p experiments/ecro_exp94b/b94.py; sed -n 20,110p docs/worklogs/94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
  -> zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    Xr = {p: 0.5 * (zf[f'{p}_a'].astype(np.float64) + zf[f'{p}_b'].astype(np.float64)) for p in PARTS}
    fr = sum(Xr.values())
    den = fr.mean(1, keepdims=True) + 0.05 * fr.mean()
    nmr = {p: (Xr[p] / den).sum((1, 2)) for p in  ...[cut]... quartile | 1.70× |
  | all steps on the single top item | 2.84× |

## Observation

- The relevant physical strata exist, are exactly identified by path labels, and are measured reliably (independent-seed rank correlation 0.999).
- They are a small and diffuse part of the training signal: about 3.4 % of the loss mass and about 0.5 % on box receivers. The item distribution is only moderately skewed, with a top-quartile contrast of 2.2.
- The CK2 condition that drives the sharp ROI is a box face whose target is about 70 % panel-mediated. That condition is almost absent from the 96 training items.
RUN [Read WL94B gate code and worklog results] cd /c/Projects/ReST-GI && sed -n 215,282p experiments/ecro_exp94b/b94.py; sed -n 20,110p docs/worklogs/94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
RUN [Check sender dump job] ssh LabServer63 'cd ~/rest_x97/C1 && tail -5 snd.log; ls -la snd | head; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
  -> DUMP D_s8602 gpu peak MiB 226
DUMP T_s8602 gpu peak MiB 226
DUMP R_s8602 gpu peak MiB 226
DUMP P_s8602 gpu peak MiB 226
DONE
total 99056
drwxr-xr-x 2 nd nd     4096 Nov 15 17:53 .
drwxr-xr-x 3 nd nd     4096 Nov 15 17:53 ..
-rw-r--r-- 1 nd nd    49280 Nov 15 17:53 M64_S1.npy
-rw-r--r-- 1 nd nd   540800 Nov 15 17:53 M64_all.npy
-rw-r--r-- 1 nd nd  9975568 Nov 15 17:53 SND_D_s8601.npz
-rw-r--r-- 1 nd nd 10003605 Nov 15 17:53 SND_D_s8602.npz
-rw-r--r-- 1 nd nd  9977622 Nov 15 17:53 SND_P_s8601.npz
-rw-r--r-- 1 nd nd 10004058 Nov 15 17:53 SND_P_s8602.npz
-rw-r--r-- 1 nd nd  9983959 Nov 15 17:53 SND_R_s8601.npz
602 MiB
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_sender_metrics.py
RUN [Run sender metrics on server CPU and fetch JSON] cd /c/Projects/ReST-GI/experiments/audit97/C1 && scp -q c1_sender_metrics.py LabServer63:rest_x97/C1/ && ssh LabServer63 'cd ~/rest_x97/C1 && . ~/rest_x95a/env.sh && $PY c1_sender_metrics.py snd ~/rest_x94d sender_metrics.json 2>&1 | tail -15' && scp -q LabServer63:rest_x97/C1/sender_metrics.json LabServer63:rest_x97/C1/snd/sender_dump_meta.json LabServer63:rest_x97/C1/snd/M64_S1.npy /c/Projects/ReST-GI/results/audit97/C1/
  -> own_s8601 max |stored - mine| (coded, boxAB): 1.4619760047018815e-07
D_s8601 max |stored - mine| (coded, boxAB): 8.65356288848318e-08
T_s8601 max |stored - mine| (coded, boxAB): 6.367963556463963e-08
R_s8601 max |stored - mine| (coded, boxAB): 6.543979136353073e-08
P_s8601 max |stored - mine| (coded, boxAB): 9.8812485083144e-08
own_s8602 max |stored - mine| (coded, boxAB): 1.3020469369351417e-07
D_s8602 max |stored - mine| (coded, boxAB): 1.9869810330419568e-07
T_s8602 max |stored - mine| (coded, boxAB): 1.0478781575784524e-07
R_s8602 max |stored - mine| (coded, boxAB): 2.505172960187707e-07
P_s8602 max |stored - mine| (coded, boxAB): 9.009751411603872e-08
WROTE sender_metrics.json
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_sender_report.py
RUN [Print sender recompute report] python experiments/audit97/C1/c1_sender_report.py > results/audit97/C1/sender_report.txt 2>&1; cat results/audit97/C1/sender_report.txt | cut -c1-260
  -> REFERENCE OPERATORS vs M64 S1 (matched norms)
  zero           flux-L1 1.000  mass 0.000  flux-Fro(R,G,B) 1.000 / 1.000 / 1.000  op-peak 0.000  min 0.00e+00
  svd8           flux-L1 1.150  mass 0.936  flux-Fro(R,G,B) 0.680 / 0.637 / 0.639  op-peak 0.338  min -3.54e-01
  svd8_clip0     flux-L1 0.9 ...[cut]... s8602    learned_sender panel_face coded 2.110 / -0.390 / 1.293 (0.069)   true 2.101 / -0.395 / 1.283 (0.070)  n=9
  P_s8602    learned_sender receivers  coded 1.551 / +0.587 / 1.372 (0.127)   true 1.546 / +0.584 / 1.373 (0.125)  n=1228
  P_s8602    learned_sender boxA       coded 2.269 / +0.753 / 1.447 (0.058)   true 2.263 / +0.753 / 1.439 (0.059)  n=54
  P_s8602    learned_sender boxAB      coded 2.308 / +0.790 / 1.487 (0.058)   true 2.302 / +0.790 / 1.480 (0.059)  n=108
  P_s8602    learned_sender panel_face coded 1.970 / -0.140 / 1.076 (0.069)   true 1.962 / -0.144 / 1.067 (0.070)  n=9
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_sender_report.py
RUN [Print sender recompute report] cd /c/Projects/ReST-GI && python experiments/audit97/C1/c1_sender_report.py > results/audit97/C1/sender_report.txt 2>&1; cat results/audit97/C1/sender_report.txt | cut -c1-260
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_passivity.py
RUN [Compute learned S1 operator directional albedo] cd /c/Projects/ReST-GI/experiments/audit97/C1 && scp -q c1_passivity.py LabServer63:rest_x97/C1/ && ssh LabServer63 'cd ~/rest_x97/C1 && . ~/rest_x95a/env.sh && $PY c1_passivity.py snd passivity.json 2>&1 | tail -14' && scp -q LabServer63:rest_x97/C1/passivity.json /c/Projects/ReST-GI/results/audit97/C1/
  -> D_s8601    albedo max 1.893 mean 0.563  share>1 0.0812  max by ring 1.89 1.76 1.73 1.22  mean by ring 0.60 0.58 0.54 0.54
D_s8602    albedo max 2.542 mean 0.590  share>1 0.1623  max by ring 2.54 2.46 1.69 1.23  mean by ring 0.65 0.62 0.59 0.51
P_s8601    albedo max 1.866 mean 0.602  share>1 0.0944   ...[cut]... .53 1.28  mean by ring 0.84 0.72 0.64 0.67
T_s8602    albedo max 2.931 mean 0.678  share>1 0.1966  max by ring 2.93 2.33 1.64 1.21  mean by ring 0.78 0.69 0.64 0.59
own_s8601  albedo max 2.264 mean 0.561  share>1 0.1040  max by ring 2.26 2.12 1.77 1.44  mean by ring 0.57 0.58 0.54 0.56
own_s8602  albedo max 3.077 mean 0.552  share>1 0.1539  max by ring 2.98 3.08 2.15 1.31  mean by ring 0.58 0.58 0.55 0.50
M64 S1 albedo max 0.825 mean 0.635 by ring 0.65 0.58 0.64 0.66 ; CK2 lit bins [12, 13, 14, 15] albedo [[0.596, 0.683, 0.677], [0.6, 0.686, 0.68], [0.596, 0.683, 0.677], [0.597, 0.683, 0.677]]
RUN [List render families per WL94 root] ssh LabServer63 'for d in rest_x94 rest_x94r rest_x94b rest_x94c rest_x94d; do echo "== $d $(ls ~/$d/results/renders | wc -l)"; ls ~/$d/results/renders | grep npz | sed -E "s/_r[0-9]+\.npz//" | sed -E "s/^(REF_ER_CK|CK)_(none|U07|U03|bx75|bx38i|bx38)_/\1_V_/" | sort | uniq -c; done'
  -> == rest_x94 384
     12 CK_V_A_s8601_C0_CK2_T1
     12 CK_V_A_s8601_C2_srcS1_CK2_T1
     12 CK_V_B_s8601_C0_CK2_T1
     12 CK_V_B_s8601_C2_srcS1_CK2_T1
     12 CK_V_own_s8601_C0_CK2_T1
     12 CK_V_own_s8601_C2_both_CK2_T1
     12 CK_V_own_s8601_C2_rcv_CK2_T1
     12 CK_V_own_s8601_C2_srcS1_CK2_T1
  ...[cut]... 1
     12 CK_V_R_s8602_C0_CK2_T1
     12 CK_V_R_s8602_C2_srcS1_CK2_T1
     12 CK_V_T_s8601_C0_CK2_T1
     12 CK_V_T_s8601_C2_srcS1_CK2_T1
     12 CK_V_T_s8602_C0_CK2_T1
     12 CK_V_T_s8602_C2_srcS1_CK2_T1
     12 CK_V_own_s8601_C0_CK2_T1
     12 CK_V_own_s8601_C2_both_CK2_T1
     12 CK_V_own_s8601_C2_rcv_CK2_T1
     12 CK_V_own_s8601_C2_srcS1_CK2_T1
     12 CK_V_own_s8601_C2_src_CK2_T1
     12 CK_V_own_s8602_C0_CK2_T1
     12 CK_V_own_s8602_C2_both_CK2_T1
     12 CK_V_own_s8602_C2_rcv_CK2_T1
     12 CK_V_own_s8602_C2_srcS1_CK2_T1
     12 CK_V_own_s8602_C2_src_CK2_T1
     24 REF_ER_CK_V_CK2_T1
WRITE c:\Projects\ReST-GI\experiments\audit97\C1\c1_images.py
RUN [Launch image metric recompute on server CPU] cd /c/Projects/ReST-GI/experiments/audit97/C1 && scp -q c1_images.py LabServer63:rest_x97/C1/ && ssh LabServer63 'cd ~/rest_x97/C1 && . ~/rest_x95a/env.sh && (setsid nohup $PY c1_images.py ~/rest_x94d images_recompute.json > images.log 2>&1; touch images.done) > /dev/null 2>&1 & sleep 45; tail -5 images.log'
  -> Exit code 1
tail: cannot open 'images.log' for reading: No such file or directory
RUN [Relaunch image recompute detached] ssh LabServer63 'cd ~/rest_x97/C1 && ls; . ~/rest_x95a/env.sh; setsid nohup bash -c "$PY c1_images.py $HOME/rest_x94d images_recompute.json > images.log 2>&1; touch images.done" > /dev/null 2>&1 < /dev/null & sleep 20; tail -3 ~/rest_x97/C1/images.log'
  -> c1_images.py
c1_passivity.py
c1_sender_dump.py
c1_sender_metrics.py
c1_tables.py
images.done
images.log
images_recompute.json
passivity.json
sender_metrics.json
snd
snd.log
tables_recompute.json
E_s8601_C0 none ROI_T {'E_H': 0.0858, 'signed': -0.08299, 'nb_coded': 0.00835, 'se_mc': 3e-05, 'floor': 0.00694, 'half1': 0.08595, 'half2': 0.08599}
E_s8601_C2_srcS1 none ROI_T {'E_H': 0.07731, 'signed': -0.07608, 'nb_coded': 0.00836, 'se_mc': 3e-05, 'floor': 0.00694, 'half1': 0.07746, 'half2': 0.07751}
E_s8602_C0 none ROI_T {'E_H': 0.06594, 'signed': 0.05688, 'nb_coded': 0.00848, 'se_mc': 3e-05, 'floor': 0.00696, 'half1': 0.06678, 'half2': 0.06678}
RUN [Wait for image recompute and fetch result] ssh LabServer63 'cd ~/rest_x97/C1 && while pgrep -u nd -f "c1_images.py" > /dev/null; do sleep 3; done; tail -2 images.log; ls -la images_recompute.json' && scp -q LabServer63:rest_x97/C1/images_recompute.json /c/Projects/ReST-GI/results/audit97/C1/
  -> Command did not complete within its 300s timeout and was moved to the background (ID: bpuvy6evt). Output is being written to: C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\tasks\bpuvy6evt.output. If it exits while you are still working you will be notified, but it is terminated when you give your final response and no notification can follow that — so do not end your turn to wait for it; if you need its result, wait for it before giving your final response. It is running now: do not start a second copy while it runs. If it is no longer needed, stop it with TaskStop. No output had been read from it when it was moved. A program that buffers its output, or a pipe into one such as tail, can keep the output file empty until the command finishes. To check interim output, use Read on that file path.
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
