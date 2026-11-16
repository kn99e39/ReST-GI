# Track C2: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List audit dirs and target results] ls docs/audits/wl97_evidence_reliability/ experiments/audit97 results/audit97 2>&1 | head -30; ls results/ecro_attr95a results/ecro_util96a results/wl95_calc results/wl96_calc results/wl96_calc/scratch 2>&1 | head -150
RUN [List worklogs and code files] ls docs/worklogs | sed -n '85,140p'; ls experiments/ecro_attr95a experiments/ecro_util96a experiments/ecro_attr95a/protocol experiments/ecro_util96a/protocol; wc -l experiments/ecro_attr95a/*.py experiments/ecro_util96a/*.py
  -> ls: cannot access 'docs/audits/wl97_evidence_reliability/': No such file or directory
ls: cannot access 'experiments/audit97': No such file or directory
ls: cannot access 'results/audit97': No such file or directory
results/ecro_attr95a:
an.json
chain.status
deliv.npz
gpu.txt
main.json
ops95a.npz
s0.json

results/ecro_util96a:
T96A.npz
chain.status
cost_s1.0.json
final.json
gpu.txt
s0.json
s1.json
tails.json
tails.npz

results/wl95_calc:
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
  -> 80A_WORKLOG_MULTI_AGENT_ARCHITECTURE_THOUGHT_EXPERIMENT.md
80B_WORKLOG_C1_ARCHITECTURE_REASSESSMENT.md
80C_WORKLOG_COMPILED_NEURAL_MATERIAL_RULES_SMOKE.md
80D_WORKLOG_COMPILED_CLOSURE_CORRECTNESS_AND_ENERGY.md
80_WORKLOG_ORBIT_BALANCED_ANGULAR_BASIS.md
81_WORKLOG_COMPILED_NEURAL_MATERIAL_RULES_UTILI ...[cut]... PRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md
96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
attachments
experiments/ecro_attr95a:
a95.py
protocol
run95a.sh
wl95calc

experiments/ecro_attr95a/protocol:
ecro_attr95a.json

experiments/ecro_util96a:
figw3_posthoc.py
protocol
run96a.sh
w96a.py

experiments/ecro_util96a/protocol:
ecro_util96a.json
  628 experiments/ecro_attr95a/a95.py
   29 experiments/ecro_util96a/figw3_posthoc.py
  819 experiments/ecro_util96a/w96a.py
 1476 total
RUN [List audit dirs and target results] cd /c/Projects/ReST-GI && ls docs/audits/wl97_evidence_reliability/ experiments/audit97 results/audit97 2>&1 | head -30; ls results/ecro_attr95a results/ecro_util96a results/wl95_calc results/wl96_calc results/wl96_calc/scratch 2>&1 | head -150
RUN [List worklogs and code files] cd /c/Projects/ReST-GI && ls docs/worklogs | sed -n '85,140p'; ls experiments/ecro_attr95a experiments/ecro_util96a experiments/ecro_attr95a/protocol experiments/ecro_util96a/protocol; wc -l experiments/ecro_attr95a/*.py experiments/ecro_util96a/*.py
RUN [Read WL95A worklog] wc -l docs/worklogs/95A_*.md docs/worklogs/95_*.md docs/worklogs/96_*.md docs/worklogs/96A_*.md docs/worklogs/8[7-9]_*.md docs/worklogs/9[01]_*.md && cat docs/worklogs/95A_*.md
  -> 213 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
   889 docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md
   267 docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
   263 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER ...[cut]... d ~/wl95_calc ~/rest_x92a`. The chain runs `a95.py stage0 | main | an` and `mat94.py render`. Extension to 64 draws re-runs `main` with `A95_EXTEND=1`, triggered by the predeclared rule after the 16-draw pilot.
- **Outputs.**
  - `results/ecro_attr95a/`: s0.json, main.json, an.json, deliv.npz (per-patch exitance per arm, deployed and per draw), ops95a.npz, chain.status, gpu.txt.
  - `results/evaluation/95A/`, with a sha256 manifest:
    - figA1 senders;
    - figA2 delivery;
    - figA3 box-A footprint maps (shared scale);
    - figA4 ROI renders and signed errors (shared scale, one exposure).
RUN [Read WL95A worklog] cd /c/Projects/ReST-GI && wc -l docs/worklogs/95A_*.md docs/worklogs/95_*.md docs/worklogs/96_*.md docs/worklogs/96A_*.md docs/worklogs/8[7-9]_*.md docs/worklogs/9[01]_*.md && cat docs/worklogs/95A_*.md
RUN [Read a95.py first half] sed -n 1,330p experiments/ecro_attr95a/a95.py
  -> """WL95A (protocol ecro_attr95a): sharp sender representability and flight attribution smoke on CK2 'none' (learning-free diagnostics).
Production code (model86 / ecro86 / pt86 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python a95.py stage0 <R> <R92A>     contract tests,  ...[cut]... 'passive_all': bool(all(p['passive'] for p in per)), 'nonneg': bool(min(p['min_entry'] for p in per) >= 0),
                           'mass_mean': float(np.mean([p['mass'] for p in per]))}
    out['sender'] = sm; out['operator'] = om
    # propagation: deployed + pooled draws
    rw0 = rw_of(u0, S)
    TAB = [k_ for k_ in snd]                                   # table senders
    RAY = ['X1c', 'X1']
    ex = {}; tabs_mean = {}; tabs_dep = {}

    def deliver(hit, dbin, seg, rw, store):
        res = {}
        for k_ in TAB:
            Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
RUN [Read a95.py first half] cd /c/Projects/ReST-GI && sed -n 1,330p experiments/ecro_attr95a/a95.py
RUN [Read a95.py second half] sed -n 330,628p experiments/ecro_attr95a/a95.py
  -> Dt = receive(flight(hit, dbin, rw, snd[k_], N, S), Mrcv)
            res[k_] = Dt
        for k_, cen in (('X1c', True), ('X1', False)):
            L, nr = x1_rays(PT, ctx, lt, hit, seg, cen)
            res[k_] = receive(flight_ray(L, rw, N, S), Mrcv)
            store['rays_S1'] = nr
        retu ...[cut]...  + e3 (identical across arms); floor = reference table; '
                 f'G_s = reference e2 + own learned global (diagnostic); shared error scale {E_MAX:.3g}, one exposure', fontsize=9)
    fn = os.path.join(outd, 'figA4_renders.png'); fig.savefig(fn, dpi=70, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'stage0':
        stage0(a[1], a[2])
    elif cmd == 'main':
        main(a[1], a[2])
    elif cmd == 'an':
        an(a[1], a[2])
    os._exit(0)
RUN [Read a95.py second half] cd /c/Projects/ReST-GI && sed -n 330,628p experiments/ecro_attr95a/a95.py
RUN [Find state ray sampling code] grep -n "u_state\|u_src\|def state_rays\|def set_tables\|def source\|def ray_weights\|KAPPA\|def build_state\|86001\|86002" experiments/ecro86/pt86.py experiments/ecro86/model86.py experiments/ecro86/ecro86.py experiments/ecro86/scene86.py | head -80
  -> experiments/ecro86/pt86.py:29:NB, K, KAPPA, PSRC = S.NB, S.K, S.KAPPA, S.PSRC
experiments/ecro86/pt86.py:114:        rs = np.random.default_rng(P['representation'].get('table_seed', 86001))
experiments/ecro86/pt86.py:116:        self.u_state_np = us
experiments/ecro86/pt86.py:117:        self.u_src_ ...[cut]... .py:40:def ray_weights():
experiments/ecro86/model86.py:42:    rs = np.random.default_rng(S.P['representation'].get('table_seed', 86001))
experiments/ecro86/model86.py:45:    z = ((k // KAPPA) // S.NPHI + us[2, 0]) / S.NZ
experiments/ecro86/model86.py:46:    return (z * S.DOMEGA / KAPPA).astype(np.float32)
experiments/ecro86/model86.py:119:            E1 = (Lk.view(Ls, N, NB, KAPPA, 3) * self.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
experiments/ecro86/ecro86.py:43:def build_state(ctx):
experiments/ecro86/scene86.py:17:KAPPA = REP['state_rays_per_bin']
experiments/ecro86/scene86.py:18:K = NB * KAPPA
RUN [Find state ray sampling code] cd /c/Projects/ReST-GI && grep -n "u_state\|u_src\|def state_rays\|def set_tables\|def source\|def ray_weights\|KAPPA\|def build_state\|86001\|86002" experiments/ecro86/pt86.py experiments/ecro86/model86.py experiments/ecro86/ecro86.py experiments/ecro86/scene86.py | head -80
RUN [Read state ray sampling details] sed -n 100,130p experiments/ecro86/pt86.py; sed -n 340,400p experiments/ecro86/pt86.py; sed -n 36,48p experiments/ecro86/model86.py; grep -n "def dir_in_bin" -A 12 experiments/ecro86/pt86.py
  -> self.scene = mi.load_dict(d)
        by = {s.id(): mi.ShapePtr(s) for s in self.scene.shapes()}
        self.ptrs = [by[n] for n in SHAPE_NAMES]
        self.params = mi.traverse(self.scene)
        self.V0 = {}; self.N0 = {}
        for b, name in (('A', 'boxA'), ('B', 'boxB')):
            M = g.b ...[cut]... ) * (1.0 / S.NZ)
197-        phi = (F(bn % S.NPHI) + b) * (TWO_PI / S.NPHI) - math.pi
198-        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
199-        return mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)
200-
201-    # ---------------------------------------------------------------- transport
202-    def trace(self, ray, N, rng, sd, ref, kmax, lights, mode, B=None, probe_y=None, probe_ok=None, plain=False):
203-        scene, mats = self.scene, self.mats
204-        thr = mi.Color3f(1.0)
205-        active = mi.Bool(True) if probe_ok is None else mi.Bool(probe_ok)
206-        out = {}
207-
RUN [Read state ray sampling details] cd /c/Projects/ReST-GI && sed -n 100,130p experiments/ecro86/pt86.py; sed -n 340,400p experiments/ecro86/pt86.py; sed -n 36,48p experiments/ecro86/model86.py; grep -n "def dir_in_bin" -A 12 experiments/ecro86/pt86.py
RUN [Inspect WL95A raw artifacts] mkdir -p docs/audits/wl97_evidence_reliability experiments/audit97/C2 results/audit97/C2 && python -c "
import numpy as np, json
z=np.load('results/ecro_attr95a/deliv.npz')
for k in z: print(k, z[k].shape, z[k].dtype)
o=np.load('results/ecro_attr95a/ops95a.npz')
for k in o: print(k,o[k].shape)
" ; python -c "
import json
a=json.load(open('results/ecro_attr95a/an.json'))
print(a.keys()); print(json.dumps(a['flight'],indent=0)); print(json.dumps(a['rank16'])); print(json.dumps(a['components'])); p
  -> dep|A1 (1270,) float64
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
draws|A2  ...[cut]... ": 0.0010303106008091253,
"H_share": 0.32070131161636745,
"pixels": 58035,
"noise_bound": 0.008345229143743723
},
"pool_X1c": {
"E_H": 0.012775853907539032,
"signed": -0.0067384003480377595,
"H_share": 0.32070131161636745,
"pixels": 58035,
"noise_bound": 0.008329219721851694
},
"pool_q256": {
"E_H": 0.01859747327844655,
"signed": 0.012608530
f2324a32effc80ca *results/ecro_attr95a/deliv.npz
d4d02e5a2f423ce9 *results/ecro_attr95a/ops95a.npz
d8f4ebdc72d78472 *results/ecro_attr95a/an.json
28d104991bcad6f7 *results/ecro_attr95a/main.json
ea95017a511dcb0a *results/ecro_attr95a/s0.json
RUN [Inspect WL95A raw artifacts] cd /c/Projects/ReST-GI && mkdir -p docs/audits/wl97_evidence_reliability experiments/audit97/C2 results/audit97/C2 && python -c "
import numpy as np, json
z=np.load('results/ecro_attr95a/deliv.npz')
for k in z: print(k, z[k].shape, z[k].dtype)
o=np.load('results/ecro_attr95a/ops95a.npz')
for k in o: print(k,o[k].shape)
" ; python -c "
import json
a=json.load(open('results/ecro_attr95a/an.json'))
print(a.keys()); print(json.dumps(a['flight'],indent=0)); print(json.dumps(a['rank16'])); print(json.
RUN [Recompute WL95A delivery statistics independently] cat > experiments/audit97/C2/c2_wl95a_delivery.py <<'EOF'
"""WL97 audit, track C2: independent recomputation of the WL95A delivery statistics from results/ecro_attr95a/deliv.npz.
numpy only; no audited metric function is imported."""
import json, sys
import numpy as np
z = np.load('results/ecro_attr95a/deliv.npz')
an = json.load(open('results/ecro_attr95a/an.json'))
ra, rb = z['ref_a'], z['ref_b']; ref = 0.5 * (ra + rb)
BOXA = np.arange(1130, 1184); PFACE = np.arange(1130, 1139)
ARMS = ['A0_s860
  -> Exit code 2
/usr/bin/bash: -c: line 94: unexpected EOF while looking for matching `''
RUN [Recompute WL95A delivery statistics independently] cd /c/Projects/ReST-GI && cat > experiments/audit97/C2/c2_wl95a_delivery.py <<'EOF'
"""WL97 audit, track C2: independent recomputation of the WL95A delivery statistics from results/ecro_attr95a/deliv.npz.
numpy only; no audited metric function is imported."""
import json, sys
import numpy as np
z = np.load('results/ecro_attr95a/deliv.npz')
an = json.load(open('results/ecro_attr95a/an.json'))
ra, rb = z['ref_a'], z['ref_b']; ref = 0.5 * (ra + rb)
BOXA = np.arange(1130, 1184); PFACE = np.arange(11
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_wl95a_delivery.py
RUN [Run WL95A delivery recomputation] python experiments/audit97/C2/c2_wl95a_delivery.py 2>&1 | tee results/audit97/C2/wl95a_delivery_recompute.log
  -> n0 boxA 0.05926934169469287 pface 0.0697528345058155 ref ratio half-diff -0.04956679776042456
== boxA
A0_s8601   dep 0.495/0.588/0.76 pool 0.517(se 0.005, boot 0.508..0.526) /0.569(jk 0.004; boot 0.564..0.578) corr 0.78 dev 0.0e+00
           draw ratio q0/10/25/50/75/90/100 [0.442 0.472 0.486 0. ...[cut]... ": 0.16953254329570996,
"cv_total": 0.48608660091197864,
"kurtosis_total": 27.112107270786876,
"pool_ratio_without_top5": 0.9342533754964794,
"max_patch_draw_over_ref_patch": 32.22285897077702,
"frac_zero_patch_draws": 0.3333333333333333
},
"X1": {
"top1_share": 0.038975350909005786,
"top5_share": 0.15789186827138066,
"cv_total": 0.3877470127067408,
"kurtosis_total": 7.293915836676807,
"pool_ratio_without_top5": 0.9339729466508353,
"max_patch_draw_over_ref_patch": 28.626846380148802,
"frac_zero_patch_draws": 0.3333333333333333
},
"pface_share_of_boxA_ref": 0.7063426860492208
}
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_wl95a_delivery.py
RUN [Run WL95A delivery recomputation] cd /c/Projects/ReST-GI && python experiments/audit97/C2/c2_wl95a_delivery.py 2>&1 | tee results/audit97/C2/wl95a_delivery_recompute.log
RUN [Read WL95A protocol] python -c "
import json
p=json.load(open('experiments/ecro_attr95a/protocol/ecro_attr95a.json'))
print(json.dumps(p,indent=1))
" | head -200
  -> {
 "protocol": "ecro_attr95a",
 "version": 1,
 "title": "WL95A - sharp sender representability and flight attribution smoke (learning-free; CK2 none)",
 "brief": "user brief 'WL95A - Sharp Sender Representability & Flight Attribution Smoke' (2026-11-15)",
 "batch_class": "suffix of WL95 (depend ...[cut]... rank / bin increases as repair",
  "learned global residuals",
  "e2 / e3 rebalancing",
  "material / light / geometry changes",
  "promoting G1 / G2 / EXP / C3UR",
  "successor batch"
 ],
 "pre_freeze_exposure": "All WL92-WL95 results are known. Dev run ~/rest_x95adev (D = 2 / 4 draws, q Mq = 64, 32-spp renders): Stage 0 test flags were read (all PASS; first attempt crashed on a JSON print of numpy bools after writing s0.json, fixed); the chain exercised main, renders, analysis and the rule-based extension path ('AN EXTEND' printed). No dev metric value, table or figure was opened."
}
RUN [Read WL95A protocol] cd /c/Projects/ReST-GI && python -c "
import json
p=json.load(open('experiments/ecro_attr95a/protocol/ecro_attr95a.json'))
print(json.dumps(p,indent=1))
" | head -200
RUN [Inspect server run dirs] ssh -o BatchMode=yes -o ConnectTimeout=15 LabServer63 'ls ~ | grep rest_ | tr "\n" " "; echo; ls ~/rest_x97 2>/dev/null; ls ~/rest_x95a ~/rest_x95a/results ~/rest_x95a/results/tables ~/rest_x95a/results/lists 2>/dev/null | head -80; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; cat ~/rest_x95a/env.sh'
  -> rest_8dna_baseline rest_arch77a rest_ato74 rest_ato74_design rest_ato74_dev rest_attrib71b rest_b4rep rest_closure80d rest_cmp72 rest_cmp72_dev rest_compiled80c rest_compiled81 rest_compiled81_dev rest_compiled82 rest_compiled82_dev rest_compiled82_probe rest_compiled83 rest_compiled83_dev rest_comp ...[cut]... json
deliv.npz
eval95a
fields
lists
logs
main.json
ops95a.npz
pre
renders
s0.json
senders.npz
tables
wl94d
wl95_calc

/home/nd/rest_x95a/results/lists:
r95a.txt
r95a_e_00
r95a_e_01
r95a_e_02
r95a_e_03
r95a_e_04
r95a_e_05
r95a_p_00
r95a_p_01
r95a_p_02
r95a_p_03
r95a_p_04
r95a_p_05

/home/nd/rest_x95a/results/tables:
FZ_own_s8601.npz
FZ_own_s8602.npz
T95A.npz
0 %, 1 MiB
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
RUN [Read an94 metric functions] grep -n "^def \|^W \|^W=" experiments/ecro_mat94/an94.py | head -50; grep -n "def eh" -A 28 experiments/ecro_mat94/an94.py; grep -n "def load_ref" -A 14 experiments/ecro_mat94/an94.py; grep -n "def load_hl" -A 12 experiments/ecro_mat94/an94.py; grep -n "def masks" -A 30 experiments/ecro_mat94/an94.py | head -60
  -> 23:def jd(o, fn):
28:def jl(fn):
32:W = (((np.arange(NB) // NPHI) + 0.5) / NZ * (2 * np.pi / NB)).astype(np.float64)
35:def ex(X):
40:def slope(m, r):
44:def corr(m, r):
49:def fref(R):
64:def masks(R):
69:def comp_pairs(arm):
74:def response(tab, F, v, comp_m, comp_r, P_):
94:def bin_err(X, Fv, P_, ...[cut]... (rm, rr), 'patches': int(ok.sum()),
86-            'ref_snr_median': float(np.median(np.abs(rr) / np.maximum(nz, 1e-12))) if ok.any() else None,
87-            'baseline_err': float(np.abs(em_0[ok] - er_0[ok]).sum() / max(er_0[ok].sum(), 1e-30)),
88-            'textured_err': float(np.abs(em_v[ok] - er_v[ok]).sum() / dv),
89-            'err_baseline_carried': float(np.abs(r_mod * (em_0[ok] - er_0[ok])).sum() / dv),
90-            'err_response': float(np.abs(er_0[ok] * (r_mod - r_ref)).sum() / dv),
91-            '_rho': (rm.tolist(), rr.tolist())}
92-
93-
94-def bin_err(X, Fv, P_, binmask):
RUN [Read an94 metric functions] cd /c/Projects/ReST-GI && grep -n "^def \|^W \|^W=" experiments/ecro_mat94/an94.py | head -50; grep -n "def eh" -A 28 experiments/ecro_mat94/an94.py; grep -n "def load_ref" -A 14 experiments/ecro_mat94/an94.py; grep -n "def load_hl" -A 12 experiments/ecro_mat94/an94.py; grep -n "def masks" -A 30 experiments/ecro_mat94/an94.py | head -60
RUN [Read an94 header and W definition] sed -n 1,62p experiments/ecro_mat94/an94.py; git log --oneline -5 -- experiments/ecro_mat94/an94.py
  -> """WL94 analysis: frozen Stage B metrics and gate, Stage C fit / outcome mapping per seed, final hypothesis labels, figures.

  python an94.py stageB <R>              -> results/an/stageB.json (prints the gate only)
  python an94.py stageC <R> <seed>       -> results/an/stageC_s<seed>.json (prints F ...[cut]... , 'results', 'fields', f'CKF_{v}.npz'))
        d = {}
        for h in 'ab':
            d[f'Sp_{h}'] = z[f'e2_p_{h}'].astype(np.float64); d[f'So_{h}'] = z[f'e2_o_{h}'].astype(np.float64)
            d[f'S_{h}'] = d[f'Sp_{h}'] + d[f'So_{h}']; d[f'G_{h}'] = (z[f'e3_p_{h}'] + z[f'e3_o_{h}']).astype(np.float64)
            d[f'B_{h}'] = d[f'S_{h}'] + d[f'G_{h}']
        for k in ('Sp', 'So', 'S', 'G', 'B'):
            d[k] = 0.5 * (d[f'{k}_a'] + d[f'{k}_b'])
        out[v] = d
    return out

e347904 WL94 freeze: ECRO material generalization vs sharp angular representation (protocol ecro_mat94)
RUN [Read an94 header and W definition] cd /c/Projects/ReST-GI && sed -n 1,62p experiments/ecro_mat94/an94.py; git log --oneline -5 -- experiments/ecro_mat94/an94.py
RUN [List WL95A renders on server] ssh -o BatchMode=yes LabServer63 'cd ~/rest_x95a/results; ls renders | head -100; du -sh renders pre fields tables; ls -la pre fields; ls ~/rest_x97/*; nproc; uptime'
  -> CK_none_G_s8601_X_CK2_T1_r0.json
CK_none_G_s8601_X_CK2_T1_r0.npz
CK_none_G_s8601_X_CK2_T1_r1.json
CK_none_G_s8601_X_CK2_T1_r1.npz
CK_none_G_s8602_X_CK2_T1_r0.json
CK_none_G_s8602_X_CK2_T1_r0.npz
CK_none_G_s8602_X_CK2_T1_r1.json
CK_none_G_s8602_X_CK2_T1_r1.npz
CK_none_dep_A0_s8601_X_CK2_T1_r0.json
CK ...[cut]... P:
blobs.tgz
blobs_rest_ecro86.tsv
blobs_rest_ecro86a.tsv
blobs_rest_gap90.tsv
blobs_rest_gm187.tsv
blobs_rest_rel88.tsv
blobs_rest_rel89.tsv
blobs_rest_x191.tsv
blobs_rest_x92.tsv
blobs_rest_x92a.tsv
blobs_rest_x92b.tsv
blobs_rest_x93.tsv
blobs_rest_x93a.tsv
blobs_rest_x93b.tsv
blobs_rest_x93c.tsv
blobs_rest_x93d.tsv
blobs_rest_x93d_ck2.tsv
blobs_rest_x94.tsv
blobs_rest_x94b.tsv
blobs_rest_x94c.tsv
blobs_rest_x94d.tsv
blobs_rest_x94r.tsv
blobs_rest_x95a.tsv
blobs_rest_x96a.tsv
norm_rest_ecro86.tsv
norm_rest_ecro86a.tsv
32
 17:49:03 up 108 days, 12:32, 12 users,  load average: 0.24, 0.16, 0.13
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_img95a_server.py
RUN [Run image E_H recomputation on server CPU] file experiments/audit97/C2/c2_img95a_server.py && ssh -o BatchMode=yes LabServer63 'mkdir -p ~/rest_x97/C2/out' && scp -q experiments/audit97/C2/c2_img95a_server.py LabServer63:rest_x97/C2/ && ssh -o BatchMode=yes LabServer63 'cd ~/rest_x97/C2 && file c2_img95a_server.py && . ~/rest_x95a/env.sh && CUDA_VISIBLE_DEVICES= $PY c2_img95a_server.py out/img95a.json'
  -> experiments/audit97/C2/c2_img95a_server.py: Python script, ASCII text executable
c2_img95a_server.py: Python script, ASCII text executable
{"mask_keys": ["DIFF", "PMAT_default", "PSID", "ROI_T", "S1", "T", "T_other_faces", "T_panel_face", "frame", "mirror_patch", "panel", "panel_facing_bins", "pid", ...[cut]... s +0.0006 +0.0006 | L1 vs floor 0.0044 (diff half-noise 0.0002) signed vs floor reps -0.0016 -0.0016 | corr-noise 0.97 | blk8 0.0090
pool_X1c         E_H 0.0128 signed -0.0067 nb 0.0083 | reps 0.0128 0.0128 | dE_H vs floor +0.0009 reps +0.0009 +0.0009 | L1 vs floor 0.0111 (diff half-noise 0.0002) signed vs floor reps -0.0093 -0.0093 | corr-noise 0.96 | blk8 0.0094
pool_q256        E_H 0.0186 signed +0.0126 nb 0.0084 | reps 0.0186 0.0186 | dE_H vs floor +0.0068 reps +0.0068 +0.0068 | L1 vs floor 0.0111 (diff half-noise 0.0002) signed vs floor reps +0.0100 +0.0100 | corr-noise 0.96 | blk8 0.0160
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_img95a_server.py
RUN [Run image E_H recomputation on server CPU] cd /c/Projects/ReST-GI && file experiments/audit97/C2/c2_img95a_server.py && ssh -o BatchMode=yes LabServer63 'mkdir -p ~/rest_x97/C2/out' && scp -q experiments/audit97/C2/c2_img95a_server.py LabServer63:rest_x97/C2/ && ssh -o BatchMode=yes LabServer63 'cd ~/rest_x97/C2 && file c2_img95a_server.py && . ~/rest_x95a/env.sh && CUDA_VISIBLE_DEVICES= $PY c2_img95a_server.py out/img95a.json'
RUN [Compare image recomputation to original and check ref-half sensitivity] scp -q LabServer63:rest_x97/C2/out/img95a.json results/audit97/C2/img95a_recompute.json && python -c "
import json
o=json.load(open('results/audit97/C2/img95a_recompute.json')); a=json.load(open('results/ecro_attr95a/an.json'))['images']
fl=o['arms']['floor']
mx=0
for k,r in o['arms'].items():
    mx=max(mx,abs(r['E_H']-a[k]['E_H']),abs(r['signed']-a[k]['signed']),abs(r['noise_bound']-a[k]['noise_bound']))
    d=[r['E_H_vs_ref_half'][i]-fl['E_H_vs_ref_half'][i] for i in (0,1)]
    print(f\"{k:16
  -> G_s8601          E_H vs ref halves 0.0258 0.0259  dE_H vs floor by ref half +0.0118 +0.0118  (full +0.0133)
G_s8602          E_H vs ref halves 0.0165 0.0166  dE_H vs floor by ref half +0.0024 +0.0025  (full +0.0032)
dep_A0_s8601     E_H vs ref halves 0.0254 0.0254  dE_H vs floor by ref half +0.011 ...[cut]...  +0.0079 +0.0079  (full +0.0086)
pool_A4          E_H vs ref halves 0.0260 0.0260  dE_H vs floor by ref half +0.0120 +0.0119  (full +0.0126)
pool_M64_spool   E_H vs ref halves 0.0239 0.0239  dE_H vs floor by ref half +0.0099 +0.0098  (full +0.0104)
pool_X1          E_H vs ref halves 0.0145 0.0146  dE_H vs floor by ref half +0.0005 +0.0005  (full +0.0006)
pool_X1c         E_H vs ref halves 0.0147 0.0147  dE_H vs floor by ref half +0.0006 +0.0006  (full +0.0009)
pool_q256        E_H vs ref halves 0.0205 0.0204  dE_H vs floor by ref half +0.0064 +0.0063  (full +0.0068)
max dev vs original 0
RUN [Compare image recomputation to original and check ref-half sensitivity] cd /c/Projects/ReST-GI && scp -q LabServer63:rest_x97/C2/out/img95a.json results/audit97/C2/img95a_recompute.json && python -c "
import json
o=json.load(open('results/audit97/C2/img95a_recompute.json')); a=json.load(open('results/ecro_attr95a/an.json'))['images']
fl=o['arms']['floor']
mx=0
for k,r in o['arms'].items():
    mx=max(mx,abs(r['E_H']-a[k]['E_H']),abs(r['signed']-a[k]['signed']),abs(r['noise_bound']-a[k]['noise_bound']))
    d=[r['E_H_vs_ref_half'][i]-fl['E_H_vs_ref_half'][i] for i in
RUN [Grep WL96 worklog for key claims] grep -n -i "ring-0\|ring 0\|counterexample\|rank-16\|1\.015\|0\.068\|Amdahl\|124\.7\|69\.4\|75\.8\|21\.4\|10\.6" docs/worklogs/96_*.md | head -80
  -> 45:  - both match the path-traced CK2 e2_p at ratio 1.015 / rel L1 0.068 (n0 0.059, corr 0.995).
57:  - [FACT, supervisor check] CK2 panel incidence occupies only 4 ring-0 input bins (12–15). M64·P_ring0 has rank 16 per channel, and an exact non-negative rank-16 factorisation reproduces M64·s for an ...[cut]... best lifecycle ratios are then:
145:  - edit→frame 124.7 / 114.1 = **1.09×**;
146:  - source→frame 75.8 / 65.2 = **1.16×**;
256:  - `results/wl96_calc/ck2_support.py` gives the CK2 panel support bins 12–15, all ring 0. It uses numpy geometry, unshadowed (an upper bound), with the 16 deployed source points from rng 86001.
257:  - `results/wl96_calc/ring_alloc.py` writes `ring_alloc.json` (rank 16 per channel; exact non-negative rank-16 factorisation, error 0.0; ring allocation of A1–A3). Its inputs are `results/ecro_attr95a/ops95a.npz` and `ck2_support_E.npy`; random check inputs use rng 96000.
RUN [Grep WL96 worklog for key claims] cd /c/Projects/ReST-GI && grep -n -i "ring-0\|ring 0\|counterexample\|rank-16\|1\.015\|0\.068\|Amdahl\|124\.7\|69\.4\|75\.8\|21\.4\|10\.6" docs/worklogs/96_*.md | head -80
RUN [Read WL96 worklog core] sed -n 30,160p docs/worklogs/96_*.md
  -> ## Verdict

**NO_VIABLE_LEARNED_MECHANISM** — scoped and provisional.

- **Scope.** The scene86 / GM1 room family, point lights, rigid edits and α ≈ 0.03–0.18. The judgment covers the learned mechanisms that could be specified in this study.
- **The physics critic's reading is EVIDENCE_INSUFFICIENT. ...[cut]... 3.9× cheaper source-only update is dominated by EXP's q table: 42–54 ms wall time against 2 ms GPU time.
- **Explicit carriers carry real costs that the designers underestimated:**
  - DQ / CIS per-edit dirty pairs are 60 % on the CK2 box shift (≈ 136–140 ms per edit, about 1.8× ECRO);
  - α⁻⁴ pairs;
  - per-light photon cost ∝ L·F_s;
  - ESC-R memory 18.6–74 MB per facet;
  - per-frame camera continuation;
  - a moving sharp object invalidates every cache of that object.
- **No designer smoke can discriminate learned utility with power.**
  - The cost rules are component-level, not lifecycle.
RUN [Read WL96 worklog core] cd /c/Projects/ReST-GI && sed -n 30,160p docs/worklogs/96_*.md
RUN [Read WL96 calc scripts] cat results/wl96_calc/ring_alloc.py; echo ------; cat results/wl96_calc/ring_alloc.json; echo; echo -----; cat results/wl96_calc/ck2_support.py; ls -la results/wl96_calc/scratch/*
  -> """WL96 CPU-only algebra: support-restricted exactness of rank-limited operators on ring-0 inputs, and the ring allocation of the WL95A bound
operators (inputs: results/ecro_attr95a/ops95a.npz = A1 NMF16, A2 L1-NMF16, A3 rank-32, A4 M64 S1; CK2 support weights from ck2_support_E.npy)."""
import os
i ...[cut]... xt_incidence.json
-rw-r--r-- 1 dna10 197609  5893 Nov 15 22:37 ext_incidence.py
-rw-r--r-- 1 dna10 197609  2905 Nov 15 22:37 lobe_fit.json
-rw-r--r-- 1 dna10 197609  3436 Nov 15 22:37 lobe_fit.py
-rw-r--r-- 1 dna10 197609   779 Nov 15 22:37 lobe_fit2.json
-rw-r--r-- 1 dna10 197609  1688 Nov 15 22:37 lobe_fit2.py
-rw-r--r-- 1 dna10 197609  6309 Nov 15 22:37 run2_alt.log
-rw-r--r-- 1 dna10 197609  6296 Nov 15 22:37 run2_ck2.log
-rw-r--r-- 1 dna10 197609  3231 Nov 15 22:37 run_alt.log
-rw-r--r-- 1 dna10 197609  3208 Nov 15 22:37 run_ck2.log
-rw-r--r-- 1 dna10 197609  1198 Nov 15 22:37 run_ext.log
RUN [Read WL96 calc scripts] cd /c/Projects/ReST-GI && cat results/wl96_calc/ring_alloc.py; echo ------; cat results/wl96_calc/ring_alloc.json; echo; echo -----; cat results/wl96_calc/ck2_support.py; ls -la results/wl96_calc/scratch/*
RUN [Inspect CK2 scene json and geometry code] python -c "
import json
sc=json.load(open('experiments/ecro_local93d/addendum_checker/scene_ck2.json'))
print(json.dumps(sc)[:3000])
"; grep -n "PSRC\|def rect_table\|class Geom\|SR\b\|def patch_table\|PE1\|PO\b" experiments/ecro86/scene86.py | head -40
  -> {"tag": "CK2", "targets": ["A"], "light_set": "LCK2", "camera": {"origin": [0.3, -1.6, 1.3], "target": [-1.3, 0.45, 1.0], "fov_deg": 32.0, "res": 512}, "geom": {"A": {"c": [-1.08, 0.6, 1.0], "h": [0.2, 0.45, 0.45], "yaw": 0.15}, "B": {"c": [1.4, 1.5, 0.5], "h": [0.3, 0.25, 0.5], "yaw": 0.0}}, "light ...[cut]... self.PE2 = np.zeros((N, 3))      # rect patch: p = PO + a PE1 + b PE2
86:                    self.PO[p] = o0 + iu * e1 + iv * e2; self.PE1[p] = e1; self.PE2[p] = e2
88:                    self.PC[p] = self.PO[p] + 0.5 * (e1 + e2)
98:                self.PT[p] = t; self.PB[p] = np.cross(n, t); self.PN[p] = n; self.PC[p] = self.SC + self.SR * n
99:                self.PAREA[p] = 4 * math.pi * self.SR ** 2 / (SPH_NPHI * SPH_NZ)
103:    def rect_table(self):
107:    def patch_table(self):
110:        return np.concatenate([self.PO, self.PE1, self.PE2, self.PT, self.PB, self.PN, sph, self.PSPH], 1)
RUN [Inspect CK2 scene json and geometry code] cd /c/Projects/ReST-GI && python -c "
import json
sc=json.load(open('experiments/ecro_local93d/addendum_checker/scene_ck2.json'))
print(json.dumps(sc)[:3000])
"; grep -n "PSRC\|def rect_table\|class Geom\|SR\b\|def patch_table\|PE1\|PO\b" experiments/ecro86/scene86.py | head -40
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_rank16_ring0.py
RUN [Run rank-16 ring-0 counterexample audit] python experiments/audit97/C2/c2_rank16_ring0.py 2>&1 | tee results/audit97/C2/rank16_ring0.log | head -250
  -> {
 "support": {
  "bins_dense": [
   12,
   13,
   14,
   15
  ],
  "bins_orig_16pts": [
   12,
   13,
   14,
   15
  ],
  "cos_incidence_min_max": [
   0.02438307264534844,
   0.1641819580229075
  ],
  "share_dense": {
   "12": 0.022881999743183123,
   "13": 0.2046854836573043, ...[cut]... 
  },
  "unit_out_flux_rows_rank16": {
   "ring_op_l1": [
    0.5943093001589093,
    0.6389715251111755,
    0.8470646550207888,
    1.0115703666718803
   ],
   "ck2_dense_support": {
    "rel_l1": 0.5852621824670172,
    "energy": 1.3574518044364337,
    "peak": 0.860827057234047
   }
  },
  "unit_out_flux_rows_rank24": {
   "ring_op_l1": [
    0.6500977012518503,
    1.1363437658590498,
    1.1695928813725713,
    1.194049774818337
   ],
   "ck2_dense_support": {
    "rel_l1": 0.41123415254887496,
    "energy": 0.9937441679505656,
    "peak": 0.9532668526199747
   }
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_rank16_ring0.py
RUN [Run rank-16 ring-0 counterexample audit] cd /c/Projects/ReST-GI && python experiments/audit97/C2/c2_rank16_ring0.py 2>&1 | tee results/audit97/C2/rank16_ring0.log | head -250
RUN [Find deterministic oracle validation artifacts] cd results/wl96_calc/scratch && cat d3/res_validate.json; echo; cat d1/val2.py; echo ----; cat d1/val_m64.py; echo ----; cat d1/rep.py; echo; grep -n "1\.015\|0\.068\|1\.486\|0\.487" -r . | head -20
  -> <persisted-output>
Output too large (59.5KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bxjvbolha.txt

Preview (first 2KB):
{
 "toy_vs_ref": {
  "ratio": 1.0151576688591952,
  "rel_l1": 0.06812682344574132,
  "corr ...[cut]...  2:3]).mean((0, 1))
    zb = ((np.arange(64)//16)+0.5)/4
    alb_me = (M.sum(1)/3 * zb).sum()*2*math.pi/64; alb_A = (A4[m].sum(1)/3*zb).sum()*2*math.pi/64
    print(NO, 'ring sums me/A4', [round(M[16*i:16*i+16].sum()/A4[m,16*i:16*i+16].sum(),3) for i in range(4)], 'albedo', round(alb_me,4), round(alb_A,4))
----
import math, numpy as np, sys
from er_np import ER
st = np.load('C:/Projects/ReST-GI/experiments/ecro86/data/mats86.npz')['states']
er = ER(st[1])
A4 = np.load('C:/Projects/ReST-GI/results/ecro_attr95a/ops95a.npz')['A4'].astype(np.float64)
NZ, NP = 4, 16; SUB = 8
...
</persisted-output>
RUN [Find deterministic oracle validation artifacts] cd /c/Projects/ReST-GI && cd results/wl96_calc/scratch && cat d3/res_validate.json; echo; cat d1/val2.py; echo ----; cat d1/val_m64.py; echo ----; cat d1/rep.py; echo; grep -n "1\.015\|0\.068\|1\.486\|0\.487" -r . | head -20
RUN [Read d3 toy oracle script] cd /c/Projects/ReST-GI/results/wl96_calc/scratch && cat d3/toy96.py | head -170
  -> """WL96 D3 CPU toy (scratch, not evidence of the pipeline): CK2 S1 panel -> box A exactly-2-event delivery under one point light.
Exact deterministic quadrature over the panel vs carriers (binned patch x bin table, refined tables, patch-centre exact direction,
receiver-anchored X1 draws, portal/imag ...[cut]... p.floor(V / (H_P / nv)), 0, nv - 1).astype(np.int64)
        b = bin_of_local(to_local(v), nz, nphi)
        return T[np.broadcast_to(iu, b.shape), np.broadcast_to(iv, b.shape), b]
    return f


def metrics(E, R):
    """flux ratio and abs rel L1 of per-patch exitance (sum over channels, box A Lambert albedo 0.5) against R."""
    e, r = 0.5 * E.sum(1), 0.5 * R.sum(1)
    return {'ratio': float(e.sum() / r.sum()), 'rel_l1': float(np.abs(e - r).sum() / r.sum()), 'corr': float(np.corrcoef(e, r)[0, 1])}


LIGHT0 = (np.array(sc['lights'][0]['pos'], float), np.array(sc['lights'][0]['rgb'], float))
RUN [Inspect toy96b entry points] cd /c/Projects/ReST-GI/results/wl96_calc/scratch && grep -n "validate\|argv\|def \|deliv.npz\|json.dump" d3/toy96b.py | head -60; sed -n 170,215p d3/toy96.py
  -> 14:def seg_hits_obb(p0, p1):
25:def L1s(y, v, light):
32:def rad_exact(Y, v, light, U, V):
36:def rad_centre(Y, v, light, U, V):
42:def deliver_table(light, sub, nz, nphi, n_side=4, h=0.01, ns=None, nd=None):
89:def x1_draws(light, D=64, kappa=8, seed=96100, centre=False):
118:def portal(light, Kf,  ...[cut]... ref.sum()), 'rel_l1': float(np.abs(e - ref).sum() / ref.sum()),
                             'corr': float(np.corrcoef(e, ref)[0, 1]), 'ref_half_noise': float(np.abs(z['ref_a'][BA] - z['ref_b'][BA]).sum() / 2 / ref.sum())}
        for k in ('X1', 'X1c', 'q256', 'A4'):
            p = np.asarray(z[f'draws|{k}']).mean(0)[BA]
            res[f'pooled_{k}_vs_toy'] = {'ratio': float(p.sum() / e.sum()), 'rel_l1': float(np.abs(p - e).sum() / e.sum())}
        res['t'] = time.time() - t0
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f'res_{mode}.json'), 'w'), indent=1)
RUN [Re-run copied deterministic oracle validation] mkdir -p results/audit97/C2/toy_copy && cp results/wl96_calc/scratch/d3/toy96.py results/audit97/C2/toy_copy/toy96_copy.py && sha256sum results/wl96_calc/scratch/d3/toy96.py | cut -c1-16 && cd results/audit97/C2/toy_copy && (time python toy96_copy.py validate) 2>&1 | tail -30
  -> 044efed109a6c299
{
 "toy_vs_ref": {
  "ratio": 1.0151576688591952,
  "rel_l1": 0.06812682344574132,
  "corr": 0.9950740723641449,
  "ref_half_noise": 0.05926934169469287
 },
 "pooled_X1_vs_toy": {
  "ratio": 1.0071750840743148,
  "rel_l1": 0.09764760093486582
 },
 "pooled_X1c_vs_toy": {
  "ratio": 1.0215993195279753,
  "rel_l1": 0.2307867377534812
 },
 "pooled_q256_vs_toy": {
  "ratio": 1.4008252563441086,
  "rel_l1": 0.41004681356905526
 },
 "pooled_A4_vs_toy": {
  "ratio": 1.5210302059232415,
  "rel_l1": 0.5218074525944975
 },
 "t": 23.232704401016235
}

real	0m23.382s
user	0m0.015s
sys	0m0.000s
RUN [Re-run copied deterministic oracle validation] cd /c/Projects/ReST-GI && mkdir -p results/audit97/C2/toy_copy && cp results/wl96_calc/scratch/d3/toy96.py results/audit97/C2/toy_copy/toy96_copy.py && sha256sum results/wl96_calc/scratch/d3/toy96.py | cut -c1-16 && cd results/audit97/C2/toy_copy && (time python toy96_copy.py validate) 2>&1 | tail -30
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_wl95a_consistency.py
RUN [Run WL95A consistency checks] python experiments/audit97/C2/c2_wl95a_consistency.py 2>&1 | tee results/audit97/C2/wl95a_consistency.log
  -> {
 "active_patches": 36,
 "oracle_vs_ref": {
  "ratio": 1.0151576688591952,
  "rel_l1": 0.06812682344574132,
  "n0": 0.05926934169469287,
  "total_diff_over_ref_total_se": -0.6116057338405495,
  "ref_total_half_diff_rel": -0.02478339888021228,
  "l1_half_a": 0.082419525970243,
  "l1_half_b" ...[cut]... _a_better_single_draw": 0.5625
  },
  "A4-X1": {
   "dL1_pooled_vs_ref": 0.43586689567135495,
   "se_jk": 0.05396222158284601,
   "z": 8.077260032783975,
   "dL1_pooled_vs_oracle": 0.4241598516596315,
   "dL1_deployed_vs_ref": -0.10350342429545445,
   "frac_draws_a_better_single_draw": 0.3125
  },
  "A0_s8601-M64_spool": {
   "dL1_pooled_vs_ref": 0.06962933414851347,
   "se_jk": 0.03443023357215993,
   "z": 2.0223311585319976,
   "dL1_pooled_vs_oracle": 0.10279900194748381,
   "dL1_deployed_vs_ref": 0.10476708743134455,
   "frac_draws_a_better_single_draw": 0.703125
  }
 }
}
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_wl95a_consistency.py
RUN [Run WL95A consistency checks] cd /c/Projects/ReST-GI && python experiments/audit97/C2/c2_wl95a_consistency.py 2>&1 | tee results/audit97/C2/wl95a_consistency.log
RUN [Read WL96A worklog] sed -n 1,263p docs/worklogs/96A_*.md
  -> # Worklog — WL96A: Learned Higher-Order Transport Utility Ceiling (2026-11-16)

## Origin and scope

- **Brief.** "WL96A — Learned Higher-Order Transport Utility Ceiling" (user). A suffix of WL96: the dependent, learning-free feasibility test of the one learned element WL96 left standing (a broad ≥  ...[cut]... ; only the render spp was reduced. The figure therefore showed qualitatively that |XOD − F| and |XOP − F| were the smallest tail differences, and that G and Z were larger. No number was read.
  - All criteria were written before any WL96A code ran and were not changed.
- **Post-dev fixes before the freeze:** a module import order, numpy-bool JSON serialisation, and pool sizes.
- **No amendment.**
- **Post-hoc, figure only:** figW3 colours (`figw3_posthoc.py`).
- **The full-run `T_READOUT` record** stores the masks check under a key containing "sha". The pass flag includes it (masks identical).
RUN [Read WL96A worklog] cd /c/Projects/ReST-GI && sed -n 1,263p docs/worklogs/96A_*.md
RUN [Read WL96A protocol and code structure] grep -n "^def \|^    def " experiments/ecro_util96a/w96a.py | head -60; python -c "
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in p:
    if k not in ('preserved','not_authorised'): print(k, ':', json.dumps(p[k],indent=1))
" | sed -n 1,200p
  -> 45:def _js(o):
49:def jd(o, fn):
54:def jl(fn):
58:def say(*a):
63:def setup(R):
69:def regions(G):
75:def ref_fields(R, ext=False):
94:def e2(rf, s=''):
98:def e3(rf, s=''):
103:def explicit_op(ctx, us, seed_ray):
119:def tail(op, mp, L2):
130:def realizations(ctx):
140:def fmet(X, ref, ra, rb, idx ...[cut]... itatively that |arm - F| in the ROI is visibly smallest for XOD / XOP among the tail arms and larger for the learned G arms and for Z (no number was read). (2) full-scale timing probe ~/rest_x96aprobe (stage 0 at full spp: all PASS incl. T_READOUT; tails 26 s; one render ~10 s; one extension half 18 s; Stage 2 two lights 21 s, two fits 178 s at 98 % GPU; lifecycle 18 s, boundary_ok); only walls, GPU utilisation and pass flags were read, no metric or timing value of an arm. No criterion was changed after either run; the only post-dev changes are the JSON-bool / import fixes and the pool sizes."
RUN [Read WL96A protocol and code structure] cd /c/Projects/ReST-GI && grep -n "^def \|^    def " experiments/ecro_util96a/w96a.py | head -60; python -c "
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in p:
    if k not in ('preserved','not_authorised'): print(k, ':', json.dumps(p[k],indent=1))
" | sed -n 1,200p
RUN [Read w96a.py tails and image decision code] sed -n 60,166p experiments/ecro_util96a/w96a.py; sed -n 238,405p experiments/ecro_util96a/w96a.py
  -> print(*a, flush=True)


def setup(R):
    PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    return PT, S, A, FX, ctx, lt


def regions(G):
    pm = np.asarray(G.PMAT)
    return {'boxA': BOXA, 'panel_face': PFACE, 'S1': S1, 'nonS1': np.nonzero(pm != 1 ...[cut]... E'
    sfx = '_ext' if EXT else ''
    jd({'images': img, 'decision': dec, 'ext': EXT}, os.path.join(R, 'results', f's1{sfx}.json'))
    print('S1', 'DEV' if DEV else dec['outcome'], flush=True)                # dev: no outcome exposure; extension path always exercised
    if (dec['outcome'] == 'S1_UNRESOLVED' or DEV) and not EXT:
        print('AN EXTEND', flush=True)
    say(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in dec.items()}, default=_js))


# ------------------------------------------------------------------------------------------------ Stage 2 (conditional)
RUN [Read w96a.py tails and image decision code] cd /c/Projects/ReST-GI && sed -n 60,166p experiments/ecro_util96a/w96a.py; sed -n 238,405p experiments/ecro_util96a/w96a.py
RUN [Inspect WL96A cost JSON] python -c "
import json
c=json.load(open('results/ecro_util96a/cost_s1.0.json'))
def show(o,d=0,mx=3):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and d<mx:
                print(' '*d+str(k)+':'); show(v,d+1,mx)
            else:
                s=json.dumps(v); print(' '*d+str(k)+': '+s[:300])
    elif isinstance(o,list):
        print(' '*d+json.dumps(o)[:400])
show(c)
" | head -150
  -> s: 1.0
N: 1270
gpu: "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB"
foreign_before:
 []
frame_spp: 64
res: 512
arms:
 R0:
  edit:
   wall_s: 0.14413186814635992
   wall_raw_s: 0.14413186814635992
   compile_s_total: 0.0
   events_with_compilation: 0
   cpu_s: 0.148891754000001
   g ...[cut]... 
 lambda24_apply_s: 0.0001558726653456688
 lambda64_apply_s: 0.0002428051084280014
 generator_factors_s: 0.0052984319627285
 ecro_apply_s: 0.0007731951773166656
 frame_s: 0.05733892507851124
rays:
 relation_rays: 650240
 q_shadow_rays_per_light: 20320
 render_rays_nominal_per_frame: 67108864
 dirty_rays_per_edit_median: 49936.0
process_gpu_mem:
 ["3430383, 1452 MiB"]
U:
 LAM24:
  edit: 0.9321302015681585
  source: 1.040492233086946
 LAM64:
  edit: 0.9209615109936242
  source: 1.038301742024448
 ZERO:
  edit: 1.038028573853174
  source: 1.0428847966464583
boundary_ok: true
RUN [Inspect WL96A cost JSON] cd /c/Projects/ReST-GI && python -c "
import json
c=json.load(open('results/ecro_util96a/cost_s1.0.json'))
def show(o,d=0,mx=3):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and d<mx:
                print(' '*d+str(k)+':'); show(v,d+1,mx)
            else:
                s=json.dumps(v); print(' '*d+str(k)+': '+s[:300])
    elif isinstance(o,list):
        print(' '*d+json.dumps(o)[:400])
show(c)
" | head -150
RUN [Inspect WL91 lifecycle JSON] ls results/gm1_c3urx1_91/results/econ/ && python -c "
import json
c=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
def show(o,d=0,mx=4):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and d<mx:
                print(' '*d+str(k)+':'); show(v,d+1,mx)
            else:
                s=json.dumps(v); print(' '*d+str(k)+': '+s[:260])
    elif isinstance(o,list):
        print(' '*d+json.dumps(o)[:300])
show(c)
" | head -230
  -> cold91_C3UR.json
cold91_C3URX1.json
cold91_PT.json
cold91_R0.json
ladder91.json
life91.json
scale91.json
foreign_before: ""
gpu: "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB"
frame_spp: 64
arms:
 C3URX1:
  edit:
   wall_s: 0.12465548515319824
   wall_raw_s: 0.12465548515319824
   events ...[cut]...  0.03095261938869953
   parts_sum_over_wall: 0.9996962710821007
   n: 15
  source_two_lights:
   wall_s: 0.14367064833641052
   wall_raw_s: 0.14367064833641052
   events_with_compilation: 0
   compile_s_total: 0.0
   cpu_s: 0.1448608410000034
   gpu_kernel_ms: 57.334878614638
   parts:
    render: 0.09706029295921326
    source_q_table: 0.040642041712999344
    table_handover: 0.00020400434732437134
    transport_orders: 0.005737693980336189
   wall_minus_render_s: 0.046610355377197266
   parts_sum_over_wall: 0.9998147475712991
   n: 1
  frame:
   wall_s: 0.05395129323005676
RUN [Inspect WL91 lifecycle JSON] cd /c/Projects/ReST-GI && ls results/gm1_c3urx1_91/results/econ/ && python -c "
import json
c=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
def show(o,d=0,mx=4):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)) and d<mx:
                print(' '*d+str(k)+':'); show(v,d+1,mx)
            else:
                s=json.dumps(v); print(' '*d+str(k)+': '+s[:260])
    elif isinstance(o,list):
        print(' '*d+json.dumps(o)[:300])

RUN [Fetch CKF reference and inspect WL96A artifacts] scp -q LabServer63:rest_x94/results/fields/CKF_none.npz results/audit97/C2/CKF_none.npz && sha256sum results/audit97/C2/CKF_none.npz results/ecro_util96a/* | cut -c1-16,65- && python -c "
import numpy as np, json
z=np.load('results/audit97/C2/CKF_none.npz'); print({k:(z[k].shape,str(z[k].dtype)) for k in z})
t=np.load('results/ecro_util96a/tails.npz'); print({k:(t[k].shape,str(t[k].dtype)) for k in t})
T=np.load('results/ecro_util96a/T96A.npz'); print({k:T[k].shape for k in T})
s=json.load(open(
  -> 3a6238a8a42d1db4 *results/audit97/C2/CKF_none.npz
8019cb39d03421dd *results/ecro_util96a/T96A.npz
fa6888fa1fcf237e *results/ecro_util96a/chain.status
103b775ed571ed26 *results/ecro_util96a/cost_s1.0.json
1c625581262250e4 *results/ecro_util96a/final.json
589634a73cfff5ab *results/ecro_util96a/gpu.txt ...[cut]... "e3_o": 135.8935318547684}, "nonS1": {"e2": 4397.706049614216, "e3_p": 76.30197916922758, "e3_o": 3825.2911649209937}, "all": {"e2": 4557.582972135144, "e3_p": 76.30197916922758, "e3_o": 3961.1846967757624}}
{"stage0": true, "S1": "S1_NO_HEADROOM", "S1_ext": false, "S2": null, "accuracy_confirmed": false, "cost_met": false, "U": {"1.0": {"LAM24": {"edit": 0.9321302015681585, "source": 1.040492233086946}, "LAM64": {"edit": 0.9209615109936242, "source": 1.038301742024448}, "ZERO": {"edit": 1.038028573853174, "source": 1.0428847966464583}}}, "LEARNED_RESOLVENT_UTILITY": "NO_MEANINGFUL_HEADROOM"}
RUN [Fetch CKF reference and inspect WL96A artifacts] cd /c/Projects/ReST-GI && scp -q LabServer63:rest_x94/results/fields/CKF_none.npz results/audit97/C2/CKF_none.npz && sha256sum results/audit97/C2/CKF_none.npz results/ecro_util96a/* | cut -c1-16,65- && python -c "
import numpy as np, json
z=np.load('results/audit97/C2/CKF_none.npz'); print({k:(z[k].shape,str(z[k].dtype)) for k in z})
t=np.load('results/ecro_util96a/tails.npz'); print({k:(t[k].shape,str(t[k].dtype)) for k in t})
T=np.load('results/ecro_util96a/T96A.npz'); print({k:T[k].shape for 
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_wl96a_fields.py
RUN [Run WL96A field recomputation] python experiments/audit97/C2/c2_wl96a_fields.py 2>&1 | tee results/audit97/C2/wl96a_fields_recompute.log
  -> max abs dev vs tails.json 2.905661822261152e-17
T96A identity {'F': 3.748958247275225e-08, 'XOD': 3.7561158847165665e-08, 'XD': 3.85000794849556e-08, 'XOP': 3.751578826334989e-08, 'Z': 4.109687712722704e-08, 'Fa': 4.0851971125990217e-08, 'Fb': 2.349142717283387e-08}
XD|e3_o|boxA           flux 0.9 ...[cut]... ": 0.010905398759892704,
  "XP_se_table": 0.0004726780110578782,
  "secondary_rule_bound_max(2SE,n0/2)": 0.012048774955142063
 },
 "all": {
  "XD_table_excess_linear": 0.007403904728897879,
  "XD_table_excess_quadrature": 0.020575273399834396,
  "XD_cv_rel_l2_bias": 0.026654898481422728,
  "XD_rel_l2_noise": 0.06213547175902252,
  "XP_table_excess_linear": 0.0026658903233867833,
  "XP_table_excess_quadrature": 0.011823694776476738,
  "XP_cv_rel_l2_bias": 0.013148361370221738,
  "XP_se_table": 0.000507416796365012,
  "secondary_rule_bound_max(2SE,n0/2)": 0.012443571457806763
 }
}
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_wl96a_fields.py
RUN [Run WL96A field recomputation] cd /c/Projects/ReST-GI && python experiments/audit97/C2/c2_wl96a_fields.py 2>&1 | tee results/audit97/C2/wl96a_fields_recompute.log
RUN [Inspect mat94 render entry and run script] grep -n "^def \|argv\|results'" experiments/ecro_mat94/mat94.py | head -90; cat experiments/ecro_util96a/run96a.sh | head -80
  -> 23:DEV = '--dev' in sys.argv
42:def jd(o, fn):
47:def jl(fn):
51:def sha_file(fn):
55:def lambert_state(alb):
66:def new_rows():
71:def default_pmat(G):
80:def rect_patches(G, r):
85:def variant_pmat(G, v):
99:def train_pmat_B(G, idx):
109:def val_pmat_ck(G, idx):
121:def pm_digest(pm):
126:def _pat ...[cut]... 
  pool 4 F2a "$A s2ref $R 0 40 $DEV" F2b "$A s2ref $R 40 80 $DEV" F2c "$A s2ref $R 80 120 $DEV" F2d "$A s2ref $R 120 160 $DEV"
  run F2fit "$A s2fit $R $DEV"
  renders s2 "$O/lists/r96a_s2.txt"
else
  log "STAGE2_NOT_TRIGGERED"
fi
run C1 "$A cost $R $R92A 1.0 $DEV"
if grep -q '^COST EXTEND' $O/logs/C1.log || [ -n "$DEV" ]; then   # dev exercises the scaling path regardless
  log "COST EXTEND (rule: U_ZERO >= 1.15)"
  run C15 "$A cost $R $R92A 1.5 $DEV"
  run C20 "$A cost $R $R92A 2.0 $DEV"
fi
run FIN "$A final $R $O/eval96a $DEV"
log "$(grep '^FINAL' $O/logs/FIN.log | tail -1)"
log CHAIN_DONE
RUN [Inspect mat94 render entry and run script] cd /c/Projects/ReST-GI && grep -n "^def \|argv\|results'" experiments/ecro_mat94/mat94.py | head -90; cat experiments/ecro_util96a/run96a.sh | head -80
RUN [Read mat94 render function and server layout] sed -n 183,201p experiments/ecro_mat94/mat94.py; sed -n 300,345p experiments/ecro_mat94/mat94.py; sed -n 562,600p experiments/ecro_mat94/mat94.py; ssh -o BatchMode=yes LabServer63 'ls ~/rest_x96a ~/rest_x96a/results ~/rest_x96a/results/renders | head -80; ls ~/rest_x96a/code; cat ~/rest_x96a/results/lists/r96a.txt | head -3'
  -> def set_pmat(PT, ctx, pm):
    ctx.G.PMAT[:] = pm
    ctx.PMATU = PT.U(np.asarray(pm, np.uint32))


def ctx_ck2(R):
    """Ctx of the CK2 scene (geometry, light set, camera of the WL93D addendum rev. 2), default material map."""
    import fx92
    fx92.inject(R)
    PT, S, A = install()
    FX = fx ...[cut]... X_CK2_T1_r0.json
CK_none_XD_X_CK2_T1_r0.npz
CK_none_XD_X_CK2_T1_r1.json
CK_none_XD_X_CK2_T1_r1.npz
CK_none_XD_X_CK2_T1_r2.json
CK_none_XD_X_CK2_T1_r2.npz
CK_none_XD_X_CK2_T1_r3.json
CK_none_XD_X_CK2_T1_r3.npz
CK_none_XOD_X_CK2_T1_r0.json
CK_none_XOD_X_CK2_T1_r0.npz
CK_none_XOD_X_CK2_T1_r1.json
CK_none_XOD_X_CK2_T1_r1.npz
CK_none_XOD_X_CK2_T1_r2.json
CK_none_XOD_X_CK2_T1_r2.npz
AGENTS.md
COMMIT
experiments
none results/tables/T96A.npz:F 86600 4096 CK_none_F_X_CK2_T1_r0
none results/tables/T96A.npz:F 86601 4096 CK_none_F_X_CK2_T1_r1
none results/tables/T96A.npz:F 86602 4096 CK_none_F_X_CK2_T1_r2
RUN [Read mat94 render function and server layout] cd /c/Projects/ReST-GI && sed -n 183,201p experiments/ecro_mat94/mat94.py; sed -n 300,345p experiments/ecro_mat94/mat94.py; sed -n 562,600p experiments/ecro_mat94/mat94.py; ssh -o BatchMode=yes LabServer63 'ls ~/rest_x96a ~/rest_x96a/results ~/rest_x96a/results/renders | head -80; ls ~/rest_x96a/code; cat ~/rest_x96a/results/lists/r96a.txt | head -3'
RUN [Read fx92.inject and model loading helpers] grep -rn "def inject" -A 22 experiments/ecro_attr92/fx92.py | head -60; sed -n 202,236p experiments/ecro_mat94/mat94.py; sed -n 142,182p experiments/ecro_mat94/mat94.py; sed -n 24,41p experiments/ecro_mat94/mat94.py
  -> 23:def inject(root=None, name=None):
24-    """Make E4 addressable. name: candidate id; default: the recorded choice in <root>/results/e4_choice.json."""
25-    if name is None:
26-        fn = os.path.join(root, 'results', 'e4_choice.json')
27-        if not os.path.exists(fn):
28-            retur ...[cut]...  9: 0.4, 10: 0.9}
NEW_IDS = [6, 7, 8, 9, 10]
TARGET_RECTS = list(range(6, 12))            # box A faces
PANEL_FACE = 6                               # outward normal -x (towards the panel)
LAMBERT_RECTS = [1, 2, 3, 4] + list(range(6, 18))
VARIANTS = ['none', 'U07', 'U03', 'bx75', 'bx38', 'bx38i']
VDEF = {'none': None, 'U07': ('u', 3), 'U03': ('u', 6), 'bx75': ('c', 3, 4), 'bx38': ('c', 6, 7), 'bx38i': ('c', 7, 6)}
SEEDS = [8601, 8602]
ARMS_FROZEN = ['C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1', 'C2_srcNonS1']
FREF_SPP, FREF_SEEDS = 4096, (94301, 94302)
IREF_SPP, ASPP = 8192, 4096
_INST = {}
RUN [Read fx92.inject and model loading helpers] cd /c/Projects/ReST-GI && grep -rn "def inject" -A 22 experiments/ecro_attr92/fx92.py | head -60; sed -n 202,236p experiments/ecro_mat94/mat94.py; sed -n 142,182p experiments/ecro_mat94/mat94.py; sed -n 24,41p experiments/ecro_mat94/mat94.py
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_srv_tables96a.py
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_srv_learned95a.py
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_img96a_server.py
WRITE c:\Projects\ReST-GI\experiments\audit97\C2\c2_srv_chain.sh
RUN [Deploy and launch audit server chain] cd /c/Projects/ReST-GI/experiments/audit97/C2 && sed -i 's|( cd $CODE/ecro_mat94 \&\& $PY mat94.py render $R $R/results/lists/c2.txt ) \|\| echo RENDER_FAILED|$PY $CODE/ecro_mat94/mat94.py render $R $R/results/lists/c2.txt \|\| echo RENDER_FAILED|' c2_srv_chain.sh && grep -n "render" c2_srv_chain.sh && file c2_srv_chain.sh c2_srv_*.py c2_img96a_server.py && scp -q c2_srv_chain.sh c2_srv_tables96a.py c2_srv_learned95a.py c2_img96a_server.py LabServer63:rest_x97/C2/ && ssh -o BatchMode=yes LabServ
  -> 2:# WL97 audit C2 server chain (own directory only): audit tables -> 4 renders (1 rep, 4,096 spp, ~10 s GPU each) -> learned-sender statistics
7:mkdir -p $R/results/renders $R/results/tables $R/results/lists $B/out
13:$PY $CODE/ecro_mat94/mat94.py render $R $R/results/lists/c2.txt || echo RENDER_FAILED
c2_srv_chain.sh:      Bourne-Again shell script, ASCII text executable
c2_srv_learned95a.py: Python script, ASCII text executable
c2_srv_tables96a.py:  Python script, ASCII text executable
c2_img96a_server.py:  Python script, ASCII text executable
c2_srv_chain.sh: Bourne-Again shell script, ASCII text executable
24 %, 890 MiB
3469366, 880 MiB
launched
[This command modified 1 file you've previously read: experiments\audit97\C2\c2_srv_chain.sh. Call Read before editing.]
RUN [Locate WL87/WL90 quoted claims] grep -n -i "36\b.*equal\|0 better\|3 worse\|everywhere\|better / \|equal /" docs/worklogs/87_*.md | head -30; echo ----; grep -n -i "0 better\|36 equal\|0\.0009\|P2plus\|headroom" docs/worklogs/96_*.md docs/worklogs/90_*.md docs/plans/wl96_spatio_angular_carrier/*.md 2>/dev/null | head -40
  -> ----
docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md:114:| deterministic sender-cell × receiver-point quadrature (DQ; d2, d4) | exact ER at cell centres, one visibility per pair | 5 cm × 4²: 0.026; 1.25 cm × 4²: 0.0009 | pairs ∝ A_sharp·A_rcv, cells ∝ α⁻², pairs ∝ α⁻⁴. [MEAS-S, c ...[cut]... esigner4_continuous.md:147:| 16 × 16 | 4 × 4 | none | 0.0009 | 1,548,288 |
docs/plans/wl96_spatio_angular_carrier/designer4_continuous.md:259:- *α → 0:* the cell count grows as α⁻² because cells must stay below the lobe footprint α·r (measured: g4 0.026, g8 0.0075, g16 0.0009 at α ≈ 0.03–0.075). A planar mirror needs the image-source special case; a curved mirror (caustic) has no deterministic shortcut.
docs/plans/wl96_spatio_angular_carrier/designer4_continuous.md:313:- Prior expectation is weak: sweeps 11.6 ms vs generator ≈ 12.5 ms, and P2plus ≤ 0.0009 in GM1 [MEAS cross-regime, WL95 §3.2].
RUN [Locate WL87/WL90 quoted claims] cd /c/Projects/ReST-GI && grep -n -i "36\b.*equal\|0 better\|3 worse\|everywhere\|better / \|equal /" docs/worklogs/87_*.md | head -30; echo ----; grep -n -i "0 better\|36 equal\|0\.0009\|P2plus\|headroom" docs/worklogs/96_*.md docs/worklogs/90_*.md docs/plans/wl96_spatio_angular_carrier/*.md 2>/dev/null | head -40
RUN [Read WL87 verdict section and WL90 table] sed -n 325,373p docs/worklogs/87_*.md; sed -n 150,200p docs/worklogs/90_*.md
  -> - 이 결손은 같은 예산의 sender-side 배치나 조밀한 physical table fill로 사라집니다. 다만 현재 구현에서는 둘 다 edit당 0.4–4.6 s로 비쌉니다.
  - "더 조밀한 수신측 state가 필요하다"는 결론은 지지되지 않습니다. pooled 16×도 평균 결손의 일부를 남기기 때문입니다.
- **추론 표시.** 위 내용 가운데 "학습 operator 고유의 오차"는 같은 state, 같은 seed의 쌍 비교에서 나온 측정입니다. 그 원인 메커니즘(WL86A의 single flight와 global t ...[cut]... ration 모두.
- patch 평균(공간 사영)은 E_H를 늘리지 않습니다.

**Order.**
- m* = 1: first gather만 exact로 바꾸면(M₁) 0.0146 / 0.0159 / 0.0149로, ORACLE과 0.001–0.002 차이입니다.
- first-order share는 0.96 / 1.01 / 1.00이고, 누적 몫은 gap의 9 % / −1 % / 0 %입니다.
- field 수준: W1 T1에서 table 오차가 truth noise의 2배를 넘는 첫 곳은 **panel의 order 2**입니다.

  | panel order 2 | field relative L1 |
  |---|---|
  | M₀ | 0.137 |
  | EE | 0.053 |
  | angular only | 0.148 |
  | spatial only | 0.089 |
  | truth noise | 0.043 |

- 다른 class와 다른 order는 noise 수준입니다.

**고정 해상도 grid (single-read, EE 대비 잔여, W1 T1 / T0 / W2 T1).**

| 사영 해상도 | EE 대비 잔여 |
|---|---|
RUN [Read WL87 verdict section and WL90 table] cd /c/Projects/ReST-GI && sed -n 325,373p docs/worklogs/87_*.md; sed -n 150,200p docs/worklogs/90_*.md
RUN [Locate WL87 utility counting and check server chain] grep -n "우세\|열세\|39\b\|동등" docs/worklogs/87_*.md | head -30; ls results/gm1_explicit87 | head -40; ssh -o BatchMode=yes LabServer63 'cat ~/rest_x97/C2/chain.out; ls ~/rest_x97/C2/out ~/rest_x97/C2/R/results/renders 2>/dev/null'
  -> 209:| panel object | 0.0185 / 0.0400 / 0.0339 / 0.0121 | 0.0390 / 0.0555 / 0.0515 / 0.0115 | 0.0104 / 0.0118 / 0.0104 / 0.0080 |
215:  - F4 LMIX 0.0039, LSPOT 0.0069. 같은 조건에서 ECRO는 0.026 / 0.035 입니다.
310:  - 미검사 조건 E3에서도 학습 operator와 동등합니다.
346:| LEARNED OPERATOR ADDITIONAL UTILITY | **SUPPORTED (사전 규칙) — 근거는 source만 바뀌는 갱신 비용 하나** | fidelity 효용 없음(우세 0 / 열세 3 / 36). edit-to-first-frame ≤ 15 % 차이(기준 미달). source-only table 갱신 3.5–3.9× 저렴(약 40–50 ms) |
chain.status
code
gpu_util_trace.txt
results
2026-11-15T17:58:45Z
TABLES OK
CK_none_C2F_X_CK2_T1_r0 14.27
/home/nd/rest_x97/C2/R/results/renders:
CK_none_C2F_X_CK2_T1_r0.json
CK_none_C2F_X_CK2_T1_r0.npz

/home/nd/rest_x97/C2/out:
img95a.json
RUN [Locate WL87 utility counting and check server chain] cd /c/Projects/ReST-GI && grep -n "우세\|열세\|39\b\|동등" docs/worklogs/87_*.md | head -30; ls results/gm1_explicit87 | head -40; ssh -o BatchMode=yes LabServer63 'cat ~/rest_x97/C2/chain.out; ls ~/rest_x97/C2/out ~/rest_x97/C2/R/results/renders 2>/dev/null'
RUN [Find WL87 utility computation code] ls results/gm1_explicit87/results | head -40; grep -rn "worse\|better" experiments/gm1_explicit87/*.py | head -30
  -> e3_admission.json
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
experiments/gm1_explicit87/analyze87.py:524:    better, worse, cmp_rows = [], [], []
experiments/gm1_explicit87/analyze87.py:535:                cmp_rows.append({'where': f'{e}_{t}:{k}', 'EXP': x['E_H'], 'ECRO': [y['E_H'] for y in ys], 'ECRO_better': b, 'ECRO_worse': w})
experiments/gm1_explicit87/analyze87.py:536:                (better if b else worse if w else []).append(f'{e}_{t}:{k}')
experiments/gm1_explicit87/analyze87.py:551:    if (better and not worse) or cheaper:
experiments/gm1_explicit87/analyze87.py:553:    elif not better and not exp_costlier:
experiments/gm1_explicit87/analyze87.py:571:    out = {'verdicts': V, 'sharp_by_edit': sharp, 'learned_vs_explicit_regions': cmp_rows, 'ECRO_better': better, 'ECRO_worse': worse,
RUN [Find WL87 utility computation code] cd /c/Projects/ReST-GI && ls results/gm1_explicit87/results | head -40; grep -rn "worse\|better" experiments/gm1_explicit87/*.py | head -30
RUN [Read WL87 utility rule and rows] sed -n 505,575p experiments/gm1_explicit87/analyze87.py; python -c "
import json
f=json.load(open('results/gm1_explicit87/results/final87.json'))
print(f.keys())
rows=f['learned_vs_explicit_regions']; print(len(rows), 'better',f['ECRO_better'],'worse',f['ECRO_worse'])
import numpy as np
for r in rows: print(r['where'], round(r['EXP'],4), [round(x,4) for x in r['ECRO']], r['ECRO_better'], r['ECRO_worse'])
"
  -> V['FROZEN_STATE_ADEQUACY'] = 'CONDITIONALLY SUPPORTED'
    elif not broad_ok:
        V['FROZEN_STATE_ADEQUACY'] = 'NOT SUPPORTED'
    else:
        V['FROZEN_STATE_ADEQUACY'] = 'INDETERMINATE'
    sharp = {}
    for e in E:
        pe = s2['per_edit'][e]
        dep = s1[e]['fidelity']['T1']['EXP'] ...[cut]... False
E3_T0:stationary_all 0.0038 [0.0045, 0.0041] False False
E3_T0:A 0.0058 [0.0066, 0.006] False False
E3_T0:C 0.0049 [0.0048, 0.0055] False False
E3_T0:D 0.0051 [0.0056, 0.0053] False False
E3_T0:obj_panel 0.0105 [0.0101, 0.0085] False False
E3_T0:obj_sphere 0.0108 [0.0088, 0.01] False False
E3_T1:stationary_all 0.0038 [0.0047, 0.0041] False False
E3_T1:A 0.009 [0.0097, 0.0089] False False
E3_T1:C 0.0052 [0.0051, 0.0063] False False
E3_T1:D 0.005 [0.0059, 0.0052] False False
E3_T1:obj_panel 0.0104 [0.0118, 0.0104] False False
E3_T1:obj_sphere 0.0112 [0.0088, 0.0107] False False
RUN [Read WL87 utility rule and rows] cd /c/Projects/ReST-GI && sed -n 505,575p experiments/gm1_explicit87/analyze87.py; python -c "
import json
f=json.load(open('results/gm1_explicit87/results/final87.json'))
print(f.keys())
rows=f['learned_vs_explicit_regions']; print(len(rows), 'better',f['ECRO_better'],'worse',f['ECRO_worse'])
import numpy as np
for r in rows: print(r['where'], round(r['EXP'],4), [round(x,4) for x in r['ECRO']], r['ECRO_better'], r['ECRO_worse'])
"
RUN [Count WL87 learned-vs-explicit comparisons by sign] python -c "
import json, numpy as np
f=json.load(open('results/gm1_explicit87/results/final87.json'))
rows=f['learned_vs_explicit_regions']
bw=bb=mx=0; l=[]
for r in rows:
    d=[y-r['EXP'] for y in r['ECRO']]
    if all(x>0 for x in d): bw+=1
    elif all(x<0 for x in d): bb+=1; l.append((r['where'], r['EXP'], r['ECRO']))
    else: mx+=1; l.append((r['where'], r['EXP'], r['ECRO']))
print('rows',len(rows),'both seeds ECRO worse',bw,'both better',bb,'mixed',mx)
for x in l: print(x)
# seed-level
a
  -> rows 36 both seeds ECRO worse 23 both better 5 mixed 8
('E1_T1:C', 0.01126277709730662, [0.013153214396359483, 0.010996022652351377])
('E2_T0:stationary_all', 0.004043563030507395, [0.004268397948538619, 0.0034471742832331543])
('E2_T0:D', 0.0038080015597572064, [0.003925543447996353, 0.002890975 ...[cut]... , 24, 0.053), 'E3': (15, 24, 0.05)}
dict_keys(['E1', 'E2', 'E3', 'sources', 'family', 'gates', 'tables_meta', 'EXP_classification'])
dict_keys(['EXP', 'BINQ', 'ENC', 'RAYSEND', 'C3', 'ECRO_ecro_s8601', 'ECRO_ecro_s8602', 'ORACLE', 'STALE_ecro_s8601', 'STALE_ecro_s8602'])
{"E_H": 0.01126277709730662, "H_share": 0.395615823473676, "noise_bound": 0.006483562276785664, "hl_over_h": 0.9868870463661182, "capture": 0.028468975276101503}
{"E_H": 0.013153214396359483, "H_share": 0.395615823473676, "noise_bound": 0.006349409717928945, "hl_over_h": 0.9742613029024868, "capture": 0.033247442634798174}
RUN [Count WL87 learned-vs-explicit comparisons by sign] cd /c/Projects/ReST-GI && python -c "
import json, numpy as np
f=json.load(open('results/gm1_explicit87/results/final87.json'))
rows=f['learned_vs_explicit_regions']
bw=bb=mx=0; l=[]
for r in rows:
    d=[y-r['EXP'] for y in r['ECRO']]
    if all(x>0 for x in d): bw+=1
    elif all(x<0 for x in d): bb+=1; l.append((r['where'], r['EXP'], r['ECRO']))
    else: mx+=1; l.append((r['where'], r['EXP'], r['ECRO']))
print('rows',len(rows),'both seeds ECRO worse',bw,'both better',bb,'mixed',mx)
for x in 
RUN [Inspect WL87 render naming and chain status] grep -n "def fidelity\|def load_\|def eh\|renders'\|noise_bound\|REF_\|_r{" experiments/gm1_explicit87/analyze87.py | head -40; ssh -o BatchMode=yes LabServer63 'cat ~/rest_x97/C2/chain.out; ls ~/rest_gm187/results/renders | sed -e "s/_r[0-9]*\.\(npz\|json\)$//" | sort | uniq -c | head -70; ls ~/rest_gm187/results/masks | head; ls ~/rest_ecro86/results/renders | sed -e "s/_r[0-9]*\.\(npz\|json\)$//" | sort | uniq -c | head -40'
  -> 29:def load_hl(root, pat):
30:    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
54:    p0, p1 = A.load(R87, f'REF_P_{e}_T0'), A.load(R87, f'REF_P_{e}_T1')
55:    r1 = A.load(R87, f'REF_ER_{e}_T1'); r0 = A.load(R87, f'REF_ER_{e}_T0')
137:        refs = {t: A.load( ...[cut]... E1_T1_LMIX
      4 EC_ecro_s8602_ecro_E1_T1_LSPOT
      4 EC_ecro_s8602_ecro_E2_T0
      4 EC_ecro_s8602_ecro_E2_T1
      4 EC_ecro_s8602_nog_E1_T1
      4 EC_ecro_s8602_nog_E2_T1
      4 EC_ecro_s8602_nos_E1_T1
      4 EC_ecro_s8602_nos_E2_T1
      4 EC_ecro_s8602_stale_E1_T1
      4 EC_ecro_s8602_stale_E2_T1
      4 ORA_E1_T0
      4 ORA_E1_T1
      4 ORA_E2_T0
      4 ORA_E2_T1
     16 REF_ER_E1_T0
     16 REF_ER_E1_T1
      8 REF_ER_E1_T1_LMIX
      8 REF_ER_E1_T1_LSPOT
     16 REF_ER_E2_T0
     16 REF_ER_E2_T1
     16 REF_P_E1_T0
     16 REF_P_E1_T1
     16 REF_P_E2_T0
     16 REF_P_E2_T1
RUN [Inspect WL87 render naming and chain status] cd /c/Projects/ReST-GI && grep -n "def fidelity\|def load_\|def eh\|renders'\|noise_bound\|REF_\|_r{" experiments/gm1_explicit87/analyze87.py | head -40; ssh -o BatchMode=yes LabServer63 'cat ~/rest_x97/C2/chain.out; ls ~/rest_gm187/results/renders | sed -e "s/_r[0-9]*\.\(npz\|json\)$//" | sort | uniq -c | head -70; ls ~/rest_gm187/results/masks | head; ls ~/rest_ecro86/results/renders | sed -e "s/_r[0-9]*\.\(npz\|json\)$//" | sort | uniq -c | head -40'
RUN [Read WL87 fidelity computation] sed -n 22,48p experiments/gm1_explicit87/analyze87.py; sed -n 128,170p experiments/gm1_explicit87/analyze87.py; grep -n "def load\b" -A 12 experiments/ecro86/analyze86.py | head -40; grep -n "def eh\|def region_metrics\|def fid" -A 16 experiments/ecro86/analyze86.py | head -60
  -> FAMILY = [('E2_T1', 'C'), ('E2_T0', 'C'), ('E1_T1', 'obj_panel')]


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def load_hl(root, pat):
    fns = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
 ...[cut]... )
123-    hl, hla, hlb = halves(ec['hl'])
124-    T = {}
125-    for k in REG:
126-        m = M[k]
127-        if m.sum() < 30:
128-            continue
129-        den = max(tot[m].sum(), 1e-30)
130-        T[k] = {'E_H': float(np.abs(hl[m] - h[m]).sum() / den), 'H_share': float(h[m].sum() / den),
131-                'noise_bound': float((0.5 * np.abs(hla[m] - hlb[m]).sum() + 0.5 * np.abs(ha[m] - hb[m]).sum()) / den),
132-                'hl_over_h': float(hl[m].sum() / max(h[m].sum(), 1e-30))}
133-        T[k]['capture'] = T[k]['E_H'] / max(T[k]['H_share'], 1e-30)
134-    return T
135-
136-
RUN [Read WL87 fidelity computation] cd /c/Projects/ReST-GI && sed -n 22,48p experiments/gm1_explicit87/analyze87.py; sed -n 128,170p experiments/gm1_explicit87/analyze87.py; grep -n "def load\b" -A 12 experiments/ecro86/analyze86.py | head -40; grep -n "def eh\|def region_metrics\|def fid" -A 16 experiments/ecro86/analyze86.py | head -60
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
