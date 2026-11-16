# Critic X1: action log of the interrupted attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List WL98 design files and sizes] ls -la docs/plans/wl98_learned_directional_transport/ && wc -l docs/plans/wl98_learned_directional_transport/* && ls experiments/wl98_calc results/wl98_calc 2>&1
  -> total 237
drwxr-xr-x 1 dna10 197121     0 Nov 16 13:19 .
drwxr-xr-x 1 dna10 197121     0 Nov 16 12:49 ..
-rw-r--r-- 1 dna10 197121 67175 Nov 16 13:15 D1_physics_measure_first.md
-rw-r--r-- 1 dna10 197121 65455 Nov 16 13:17 D2_learning_generalization_first.md
-rw-r--r-- 1 dna10 197121 58724 Nov 16 13 ...[cut]... md
-rw-r--r-- 1 dna10 197121 28850 Nov 16 12:49 DESIGN_REVIEW.md
-rw-r--r-- 1 dna10 197121   321 Nov 16 13:19 X4_critic_prior_art.md
   527 docs/plans/wl98_learned_directional_transport/D1_physics_measure_first.md
   845 docs/plans/wl98_learned_directional_transport/D2_learning_generalization_first.md
   686 docs/plans/wl98_learned_directional_transport/D3_systems_cost_first.md
   307 docs/plans/wl98_learned_directional_transport/DESIGN_REVIEW.md
     5 docs/plans/wl98_learned_directional_transport/X4_critic_prior_art.md
  2370 total
experiments/wl98_calc:
D1
D2
D3

results/wl98_calc:
D1
D2
D3
RUN [List WL98 design files and sizes] cd /c/Projects/ReST-GI && ls -la docs/plans/wl98_learned_directional_transport/ && wc -l docs/plans/wl98_learned_directional_transport/* && ls experiments/wl98_calc results/wl98_calc 2>&1
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\DESIGN_REVIEW.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\X4_critic_prior_art.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\DESIGN_REVIEW.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\X4_critic_prior_art.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\D1_physics_measure_first.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\D1_physics_measure_first.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\D2_learning_generalization_first.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\D2_learning_generalization_first.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\D3_systems_cost_first.md 
READ c:\Projects\ReST-GI\docs\plans\wl98_learned_directional_transport\D3_systems_cost_first.md 
RUN [List ECRO86 files and grep pt86a for tags] ls experiments/ecro86/ && grep -n "def \|field_src\|event\|cls\b\|class_" experiments/ecro86/pt86a.py 2>/dev/null | head -80
  -> __pycache__
analyze86.py
data
diag_v3.py
econ86.py
ecro86.py
gen86.py
masks86.py
mats86.py
model86.py
protocol
pt86.py
run86.sh
scene86.py
tests86.py
train86.py
viz86.py
watch86.sh
RUN [List ECRO86 files and grep pt86a for tags] cd /c/Projects/ReST-GI && ls experiments/ecro86/ && grep -n "def \|field_src\|event\|cls\b\|class_" experiments/ecro86/pt86a.py 2>/dev/null | head -80
RUN [Find field_src definitions] grep -rln "field_src" experiments | head -20; grep -rn "def field_src" experiments | head
  -> experiments/audit97/E3/ana97.py
experiments/audit97/E4/e4_kmax.py
experiments/audit97/MAP/build_map.py
experiments/ecro86a/attrib86a.py
experiments/ecro86a/d2_86a.py
experiments/ecro86a/pt86a.py
experiments/ecro_attr92/diag92.py
experiments/ecro_attr92/interv92.py
experiments/ecro_attr92/protocol/ecro_attr92.json
experiments/ecro_attr95a/a95.py
experiments/ecro_attr95a/protocol/ecro_attr95a.json
experiments/ecro_exp94b/b94.py
experiments/ecro_exp94b/protocol/ecro_exp94b.json
experiments/ecro_mat94/c94.py
experiments/ecro_mat94/mat94.py
experiments/ecro_mat94/protocol/ecro_mat94.json
experiments/ecro_own92a/own92a.py
experiments/ecro_own92a/protocol/ecro_own92a.json
experiments/ecro_util96a/protocol/ecro_util96a.json
experiments/ecro_util96a/w96a.py
experiments/ecro86a/pt86a.py:73:def field_src(ctx, src, spp, seed, kmax=PT.D - 1, b=128, split=False):
RUN [Find field_src definitions] cd /c/Projects/ReST-GI && grep -rln "field_src" experiments | head -20; grep -rn "def field_src" experiments | head
RUN [Read pt86a field_src reference tracer] sed -n 1,200p experiments/ecro86a/pt86a.py
  -> """WL86A diagnostic sources for the frozen WL86 transport module (pt86 is imported unchanged; nothing here is a runtime input of ECRO).

Field probes (radiance leaving a surface point made of 2..5 scattering events, exactly pt86.Ctx.field) under controlled incident fields:
  'true'            the li ...[cut]... out[f'{nm}1'] = parts[q].y; out[f'{nm}2'] = parts[q].z
        a = ctx.accum(out, slot, ns)
        acc = a if acc is None else {k_: acc[k_] + v for k_, v in a.items()}
        done += bb; launch += 1
    cnt = acc['cnt'].reshape(ctx.N, NB)
    f = np.stack([acc[f'f{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / np.maximum(cnt, 1)[..., None]
    if split:
        return {nm: (np.stack([acc[f'{nm}{c}'] for c in range(3)], -1).reshape(ctx.N, NB, 3) / np.maximum(cnt, 1)[..., None]).astype(np.float32)
                for nm in ('e2_p', 'e2_o', 'e3_p', 'e3_o')}
    return f.astype(np.float32)
RUN [Read pt86a field_src reference tracer] cd /c/Projects/ReST-GI && sed -n 1,200p experiments/ecro86a/pt86a.py
RUN [Inspect fixture and scene86 panel/bin definitions] python -c "
import json;d=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(json.dumps(d.get('fixture'),indent=1)[:3000])" ; grep -n "S1\|panel\|NZ\|NPHI\|def dir_in_bin\|def bin_of\|^D *=\|PATCH\|patch_size\|0.1\b" experiments/ecro86/scene86.py | head -60
  -> {
 "units": "metres, z up; room x,y in [-2,2], z in [0,3], open towards -y; camera outside the opening",
 "rationale": "semi-enclosed room so that >= 3-event transport is a meaningful part of the radiance; stationary sharp panel and moderate sphere; two movable Lambert boxes whose rigid edits move ...[cut]... nge(SPH_NPHI):
94:                p = self.SBASE + ip + SPH_NPHI * iz
95:                z = (iz + 0.5) / SPH_NZ * 2 - 1; phi = (ip + 0.5) / SPH_NPHI * 2 * math.pi - math.pi
99:                self.PAREA[p] = 4 * math.pi * self.SR ** 2 / (SPH_NPHI * SPH_NZ)
132:    """Bin-centre local directions [NB, 3] (bin = iz * NPHI + iphi)."""
134:    for iz in range(NZ):
135:        for ip in range(NPHI):
136:            z = (iz + 0.5) / NZ; phi = (ip + 0.5) / NPHI * 2 * math.pi - math.pi
137:            out[iz * NPHI + ip] = [math.sqrt(1 - z * z) * math.cos(phi), math.sqrt(1 - z * z) * math.sin(phi), z]
RUN [Inspect fixture and scene86 panel/bin definitions] cd /c/Projects/ReST-GI && python -c "
import json;d=json.load(open('experiments/ecro86/protocol/ecro86.json'));print(json.dumps(d.get('fixture'),indent=1)[:3000])" ; grep -n "S1\|panel\|NZ\|NPHI\|def dir_in_bin\|def bin_of\|^D *=\|PATCH\|patch_size\|0.1\b" experiments/ecro86/scene86.py | head -60
RUN [Find CK2 workload definition] grep -rn "CK2\|ck2" experiments/ecro_util96a/protocol/ecro_util96a.json | head -10; grep -rln "'none'" experiments/ecro_attr95a/*.py | head; python - <<'EOF'
import json
d=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
def walk(o,path=''):
    if isinstance(o,dict):
        for k,v in o.items():
            if 'ck' in k.lower() or 'light' in k.lower(): print(path+'/'+k, str(v)[:300])
            walk(v,path+'/'+k)
    elif isinstance(o,list):
        for i,v in enumerate(
  -> 4: "title": "WL96A - learned higher-order transport utility ceiling (learning-free; CK2 none)",
12:  "CK2 geometry, materials (default map 'none'), light LCK2, camera, masks94",
27:  "T_PMAT": "CK2 'none' material map digest = CKF_none.json pmat_digest (3d894e73cef701a4)",
29:  "T_SPLIT_FIELD": "mat ...[cut]... e86.light_ok on the CK2 geometry) from rng 96200, additionally rejected within 0.3 m of the CK2 light; field_src split, 2 x 512 spp each (seeds 96300 + 2 l, + 1); estimated 12 GPU-min",
59:  "evaluation": "held-out CK2 light (x = CKF e2 mean, y = CKF e3_o mean) and in-sample fit lights on half b; field rel L1 per region and image dI as in Stage 1 (table = e2 + reference e3_p + Lambda x)",
61:  "ACCURACY_HEADROOM_CONFIRMED": "for rho = 24 or 64 on the held-out CK2 light: dI(fit) <= dI(XOD) - M_img AND boxA e3_o field rel L1(fit) < rel L1(XOD) - max(n0 / 2, 0.02)"
experiments/ecro_attr95a/a95.py
RUN [Find CK2 workload definition] cd /c/Projects/ReST-GI && grep -rn "CK2\|ck2" experiments/ecro_util96a/protocol/ecro_util96a.json | head -10; grep -rln "'none'" experiments/ecro_attr95a/*.py | head; python - <<'EOF'
import json
d=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
def walk(o,path=''):
    if isinstance(o,dict):
        for k,v in o.items():
            if 'ck' in k.lower() or 'light' in k.lower(): print(path+'/'+k, str(v)[:300])
            walk(v,path+'/'+k)
    elif isinstance(o,list):
   
RUN [Locate CK2 light position] grep -rn "LCK2\|CK2_LIGHT\|ck2_light" --include=*.py --include=*.json experiments | head -8
  -> experiments/audit97/D/edit_validity.py:38:                 'light_ok_LCK2': bool(S.light_ok(ck['lights'][0]['pos'], g)),
experiments/ecro_local93d/addendum_checker/protocol_checker93d.json:28:   "light_set": "LCK2",
experiments/ecro_local93d/addendum_checker/scene_ck2.json:6: "light_set": "LCK2",
experiments/ecro_mat94/protocol/ecro_mat94.json:65:  "scene": "CK2 geometry, light LCK2 and camera of the addendum (unchanged)",
experiments/ecro_nearmirror93e/scene.json:2:  "tag": "CK2NM93E", "targets": ["A"], "light_set": "LCK2NM93E",
experiments/ecro_util96a/protocol/ecro_util96a.json:12:  "CK2 geometry, materials (default map 'none'), light LCK2, camera, masks94",
RUN [Locate CK2 light position] cd /c/Projects/ReST-GI && grep -rn "LCK2\|CK2_LIGHT\|ck2_light" --include=*.py --include=*.json experiments | head -8
RUN [Read CK2 scene definition] cat experiments/ecro_local93d/addendum_checker/scene_ck2.json
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
    0.45
   ],
   "yaw": 0.15
  },
  "B": {
   "c": [
    1.4,
    1.5,
    0.5
   ],
   "h": [
    0.3,
    0.25,
    0.5
   ],
   "yaw": 0.0
  }
 },
 "lights": [
  {
   "pos": [
    -1.45,
    -0.6,
    1.7
   ],
   "rgb": [
    30.0,
    30.0,
    30.0
   ]
  }
 ]
}
RUN [Read CK2 scene definition] cd /c/Projects/ReST-GI && cat experiments/ecro_local93d/addendum_checker/scene_ck2.json
RUN [Check in-bin direction sampling and depth] grep -n "def dir_in_bin" -A8 experiments/ecro86/pt86.py; grep -n "^D\b\|^D =\|D = " experiments/ecro86/pt86.py | head -3; grep -n "def patch_bin" -A6 experiments/ecro86/pt86.py
  -> 195:    def dir_in_bin(bn, a, b):
196-        z = (F(bn // S.NPHI) + a) * (1.0 / S.NZ)
197-        phi = (F(bn % S.NPHI) + b) * (TWO_PI / S.NPHI) - math.pi
198-        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
199-        return mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)
200-
201-    # ------- ...[cut]... ):
203-        scene, mats = self.scene, self.mats
6:  camera modes  : e1 (k = 1) = q read by the camera, e2 (k = 2) = T q, h (3 <= k <= D) = H[q] = sum_{j=2}^{D-1} T^j q, D = 6
26:D = 6
158:    def patch_bin(self, si, sid, w):
159-        """Patch id of the hit and direction bin of the world direction w in that patch's frame."""
160-        isb = sid >= S.SID_BOXA
161-        b = dr.select(isb, sid - S.SID_BOXA, U(0))
162-        gb = lambda k: dr.gather(F, self.BX, b * 9 + k)
163-        n = si.n
164-        dd = [n.x * gb(3 * a) + n.y * gb(3 * a + 1) + n.z * gb(3 * a + 2) for a in range(3)]
RUN [Check in-bin direction sampling and depth] cd /c/Projects/ReST-GI && grep -n "def dir_in_bin" -A8 experiments/ecro86/pt86.py; grep -n "^D\b\|^D =\|D = " experiments/ecro86/pt86.py | head -3; grep -n "def patch_bin" -A6 experiments/ecro86/pt86.py
RUN [Run X1 scratch: CK2 specular spread per panel patch] mkdir -p /c/Projects/ReST-GI/experiments/wl98_calc/X1 /c/Projects/ReST-GI/results/wl98_calc/X1 && cat > /c/Projects/ReST-GI/experiments/wl98_calc/X1/ck2_spread.py <<'EOF'
"""WL98 critic X1 scratch (design input only, not evidence).
(1) CK2 geometry: across each 0.2 m S1 panel patch, how far does the mirror (specular) direction of the single CK2
    point light move? Compared with the S1 grazing lobe width, this is how many direction-separable lobes a
    per-element outgoing lobe state (FLS) wou
  -> {'patch': (0, 6), 'cos_i_med': 0.1193224610059469, 'cos_i_min': 0.09260416717734395, 'cos_i_max': 0.16409569219266537, 'spec_spread_deg_p95': 13.494634298189144, 'spec_spread_deg_max': 19.35878796886352, 'outgoing_elev_deg_med': 6.853001503966392, 'E_share': 0.254318157027575}
{'patch': (0, 5), 'co ...[cut]... g_med': 3.5325538593680883, 'E_share': 0.03220916208871128}
lobe half-angle deg 0.6624793435021039
ratios [20.4, 18.7, 13.9, 14.3, 13.4, 12.1, 11.4, 10.3]
[{'band': 0, 'polar_span_deg': 59.999999999999986, 'azimuth_span_deg': 45.0}, {'band': 1, 'polar_span_deg': 30.000000000000014, 'azimuth_span_deg': 45.0}, {'band': 2, 'polar_span_deg': 29.999999999999993, 'azimuth_span_deg': 45.0}, {'band': 3, 'polar_span_deg': 60.00000000000001, 'azimuth_span_deg': 45.0}]
{'t=0.5': 1.0, 't=0.6': 0.6666666666666667, 't=0.7': 0.42857142857142866, 't=0.8': 0.24999999999999994, 't=0.9': 0.11111111111111108}
RUN [Grep Architecture.md for ownership rules] grep -n "same-asset\|Object identity\|visibility\|never learned\|scene-baked\|Gate A\|Gate B" docs/Architecture.md | head -40
  -> 20:- **C** maps an outgoing boundary event through editable free space to the next boundary entry. It includes exposed same-asset and same-domain exit/re-entry paths.
25:Every path segment has exactly one owner. Object identity must not freeze exposed interreflection or merge editable air transport into K. Internal response must not be gated by visibility of the output point to a direct source.
53:| Free-space state | Current visibility, directions and transport relations for C | Update for affected editable air paths |
103:A free-space radiosity solver, SH2-only Lambertian model, scene-baked monolithic radiance MLP or SPT-1 repair is not an interchangeable implementation of this contract. A genuine contract conflict must be reported before changing scope.
RUN [Grep Architecture.md for ownership rules] cd /c/Projects/ReST-GI && grep -n "same-asset\|Object identity\|visibility\|never learned\|scene-baked\|Gate A\|Gate B" docs/Architecture.md | head -40
RUN [Read Architecture.md core sections] sed -n 1,110p docs/Architecture.md
  -> # Architecture — FNTO Dynamic Transport Reuse

**Current as of:** 2026-11-13. **Role:** current physical/representation contracts, candidate status and unresolved design decisions. See [document ownership](README.md).

## 1. Objective and design boundary

Maintain physically credible global illumina ...[cut]... ent; none is selected by this baseline decision.
- **Component-to-global boundary:** ER's pointwise local response is qualified only for the tested material family. Nonlocal intrinsic transport, mixed materials, first-exit/boundary-port fields and composed multi-bounce error remain unvalidated with ER; do not silently infer them from WL84.

- **INCONCLUSIVE — learned K adequacy:** which spatial/angular representation and supervision can meet physical, directional and deformation fidelity while remaining passive under composed incidence? Passing algebraic constraints alone does not answer this.
RUN [Read Architecture.md core sections] cd /c/Projects/ReST-GI && sed -n 1,110p docs/Architecture.md
RUN [Read ECRO generator inputs and apply] sed -n 78,126p experiments/ecro86/model86.py
  -> self.head_P = mlp(d, d, 3 * self.R * self.RHO, 1)
        with torch.no_grad():
            self.head_c[-1].bias.fill_(-2.0); self.head_a[-1].bias.fill_(-2.0)
            self.head_Q[-1].bias.fill_(-6.0); self.head_P[-1].bias.fill_(-6.0)

    def form_factors(self, gs):
        """Cosine-weighted vi ...[cut]... cr,lncr->lndc', f['a'], u).reshape(Ls, N * NB, 3)
            ok = (gs.hit >= 0).view(1, N * K, 1)
            idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
            Lk = Bq[:, idx] * ok
            E1 = (Lk.view(Ls, N, NB, KAPPA, 3) * self.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
            y = y + torch.einsum('nmcr,lnmc->lncr', f['c'], E1)
        if use_global:
            wq = torch.einsum('ncpr,lncr,n->lcp', f['Q'], u, gs.area)
            y = y + torch.einsum('ncrp,lcp->lncr', f['P'], wq)
        return torch.einsum('ndcr,lncr->lndc', f['a'], y)

    def forward(self, gs, s, **kw):
RUN [Read ECRO generator inputs and apply] cd /c/Projects/ReST-GI && sed -n 78,126p experiments/ecro86/model86.py
RUN [Hash scratch files and check D2 lobe width data] sha256sum experiments/wl98_calc/X1/ck2_spread.py results/wl98_calc/X1/ck2_spread.json | cut -c1-16; python -c "
import json;d=json.load(open('results/wl98_calc/D2/angular_capacity.json'));print(list(d.keys())[:20])" 2>&1 | head; grep -n "half" results/wl98_calc/D2/angular_capacity.log | head
  -> 64bdadcb0618ca03
6392c9f21c6a1a49
['cos_i=0.025', 'cos_i=0.05', 'cos_i=0.1', 'cos_i=0.25', 'cos_i=0.4', 'cos_i=0.8']
1:0.025 half-max sr 0.00042 | 4x16 sa L1 1.647 flux 1.458 | 4x16 cos L1 1.269 | 8x32 cos L1 1.081 | 32x128 cos L1 0.705 L2 0.951 | qt1024 L1 0.106 L2 0.119 | mix1 L1 1.021 | mix2 L1  ...[cut]... 07 | 32x128 cos L1 0.159 L2 0.218 | qt1024 L1 0.083 L2 0.076 | mix1 L1 0.414 | mix2 L1 0.191 L2 0.489 pk 2.28 | mix4 L1 0.096 L2 0.599 | t1 L1 0.197 L2 1.440 pk 6.38 | t2 L1 0.113 L2 0.987 | t4 L1 0.109 L2 0.739 hm 0.983 | qt256 L1 0.180 qt4096 L1 0.041
6:0.8 half-max sr 0.02833 | 4x16 sa L1 0.807 flux 1.018 | 4x16 cos L1 0.799 | 8x32 cos L1 0.383 | 32x128 cos L1 0.105 L2 0.121 | qt1024 L1 0.069 L2 0.058 | mix1 L1 0.350 | mix2 L1 0.122 L2 0.114 pk 0.80 | mix4 L1 0.044 L2 0.087 | t1 L1 0.036 L2 0.249 pk 0.95 | t2 L1 0.040 L2 0.238 | t4 L1 0.050 L2 0.216 hm 0.996 | qt256 L1 0.143 qt4096 L1 0.034
RUN [Hash scratch files and check D2 lobe width data] cd /c/Projects/ReST-GI && sha256sum experiments/wl98_calc/X1/ck2_spread.py results/wl98_calc/X1/ck2_spread.json | cut -c1-16; python -c "
import json;d=json.load(open('results/wl98_calc/D2/angular_capacity.json'));print(list(d.keys())[:20])" 2>&1 | head; grep -n "half" results/wl98_calc/D2/angular_capacity.log | head
