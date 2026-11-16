# Track E5: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List WL96A code and result files] ls experiments/ecro_util96a/ results/ecro_util96a/ docs/audits/ experiments/audit97/ 2>&1 | head -80; wc -l experiments/ecro_util96a/w96a.py; grep -n "def " experiments/ecro_util96a/w96a.py | head -80
  -> docs/audits/:
wl97_evidence_reliability

experiments/audit97/:
A
B
C1
C2
D
E1
E2
E3
E4
F0
MAP

experiments/ecro_util96a/:
figw3_posthoc.py
protocol
run96a.sh
w96a.py

results/ecro_util96a/:
T96A.npz
chain.status
cost_s1.0.json
final.json
gpu.txt
s0.json
s1.json
tails.json
tails.npz
819 experiments/e ...[cut]... 497:    def kh(*a, **k):
510:def refine(S, s):
521:def cost(R, R92A, s=1.0):
533:    def sync():
536:    def drain():
542:        def __init__(self):
547:        def _take(self):
551:        def part(self, name):
555:        def done(self):
569:    def lam_apply(rho, L2):
576:        def __init__(self, arm):
582:        def _table(self, lt_, ck):
595:        def _frame(self, lt_, seed, ck):
598:        def edit(self, geom, lt_, seed):
613:        def source(self, lt_, seed):
640:        def summ(rows):
651:            def tm(fn, n=20):
689:def final(R, outdir):
740:def figures(R, outdir, ext):
RUN [List WL96A code and result files] cd /c/Projects/ReST-GI && ls experiments/ecro_util96a/ results/ecro_util96a/ docs/audits/ experiments/audit97/ 2>&1 | head -80; wc -l experiments/ecro_util96a/w96a.py; grep -n "def " experiments/ecro_util96a/w96a.py | head -80
READ c:\Projects\ReST-GI\experiments\ecro_util96a\w96a.py 
READ c:\Projects\ReST-GI\experiments\ecro_util96a\w96a.py 
RUN [Print original WL96A cost results] python -c "
import json
d=json.load(open('results/ecro_util96a/cost_s1.0.json'))
def show(o,ind=0,depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,)) and depth<4:
                print(' '*ind+str(k)+':'); show(v,ind+2,depth+1)
            else:
                s=json.dumps(v); print(' '*ind+str(k)+': '+s[:400])
show(d)
"; cat results/ecro_util96a/gpu.txt | head -20; ls docs/audits/wl97_evidence_reliability/
  -> s: 1.0
N: 1270
gpu: "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB"
foreign_before: []
frame_spp: 64
res: 512
arms:
  R0:
    edit:
      wall_s: 0.14413186814635992
      wall_raw_s: 0.14413186814635992
      compile_s_total: 0.0
      events_with_compilation: 0
      cpu_s: 0.14889 ...[cut]... 2 0 %, 576 MiB
17:05:57 67 %, 3546 MiB
17:06:02 97 %, 3546 MiB
17:06:07 96 %, 3546 MiB
17:06:12 94 %, 3546 MiB
17:06:17 0 %, 3546 MiB
17:06:22 91 %, 3546 MiB
17:06:27 16 %, 3546 MiB
17:06:32 85 %, 3546 MiB
17:06:37 85 %, 3546 MiB
A_math_physics_contract.md
B_data_and_protocol_integrity.md
C1_measurement_statistics_wl92_wl94d.md
C2_measurement_statistics.md
D_systems_performance_cost.md
E1_learned_sender_error_and_passivity.md
E2_relation_sampling_reproduction.md
E3_deterministic_delivery_oracle.md
E4_resolvent_accuracy_headroom.md
F0_inference_chain.md
MAP_evidence_dependency_and_provenance.md
READ c:\Projects\ReST-GI\experiments\ecro_util96a\w96a.py 
READ c:\Projects\ReST-GI\experiments\ecro_util96a\w96a.py 
RUN [Print original WL96A cost results] cd /c/Projects/ReST-GI && python -c "
import json
d=json.load(open('results/ecro_util96a/cost_s1.0.json'))
def show(o,ind=0,depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,)) and depth<4:
                print(' '*ind+str(k)+':'); show(v,ind+2,depth+1)
            else:
                s=json.dumps(v); print(' '*ind+str(k)+': '+s[:400])
show(d)
"; cat results/ecro_util96a/gpu.txt | head -20; ls docs/audits/wl97_evidence_reliability/
  -> s: 1.0
N: 1270
gpu: "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB"
foreign_before: []
frame_spp: 64
res: 512
arms:
  R0:
    edit:
      wall_s: 0.14413186814635992
      wall_raw_s: 0.14413186814635992
      compile_s_total: 0.0
      events_with_compilation: 0
      cpu_s: 0.14889 ...[cut]... 2 0 %, 576 MiB
17:05:57 67 %, 3546 MiB
17:06:02 97 %, 3546 MiB
17:06:07 96 %, 3546 MiB
17:06:12 94 %, 3546 MiB
17:06:17 0 %, 3546 MiB
17:06:22 91 %, 3546 MiB
17:06:27 16 %, 3546 MiB
17:06:32 85 %, 3546 MiB
17:06:37 85 %, 3546 MiB
A_math_physics_contract.md
B_data_and_protocol_integrity.md
C1_measurement_statistics_wl92_wl94d.md
C2_measurement_statistics.md
D_systems_performance_cost.md
E1_learned_sender_error_and_passivity.md
E2_relation_sampling_reproduction.md
E3_deterministic_delivery_oracle.md
E4_resolvent_accuracy_headroom.md
F0_inference_chain.md
MAP_evidence_dependency_and_provenance.md
RUN [Outline production modules used by the cost stage] grep -n "^def \|^class \|^    def " experiments/ecro_mat94/mat94.py | head -60; grep -n "^def \|^class \|^    def " experiments/gm1_explicit87/quad87.py experiments/gm1_explicit87/xport87.py | head -60; ls experiments/ecro_mat94/ ; grep -rn "E87" experiments/*/fx92.py | head
RUN [Check server state and WL96A run directory] ssh -o ConnectTimeout=20 LabServer63 'uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls ~/rest_x97/ 2>/dev/null; cat ~/rest_x95a/env.sh; ls ~/rest_x96a ~/rest_x96a/code ~/rest_x96a/results | head -60; nproc; ps -eo pid,user,pcpu,etime,cmd --sort=-pcpu | head -8'
  -> 42:def jd(o, fn):
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
126:def _patch_src(fn, obj, label, probe_ ...[cut]... ce(ops, q, first=None):
experiments/gm1_explicit87/xport87.py:89:def raysend_first(ctx, st, lt, W):
experiments/gm1_explicit87/xport87.py:107:def enc_q(G, s, Mq):
experiments/gm1_explicit87/xport87.py:118:def _cosdir_in_bin(bn, u1, u2):
experiments/gm1_explicit87/xport87.py:128:def deposit_order(ctx, L, dense, R, seed, chunk=400_000):
experiments/gm1_explicit87/xport87.py:181:def sender_side(ctx, q, dense, R, seed):
experiments/gm1_explicit87/xport87.py:191:def build(ctx, lt, quad, P, Mq, q_seed, st=None):
an94.py
c94.py
mat94.py
posthoc94.py
protocol
run94.sh
viz94.py
19:E87 = P87['explicit']
  -> 01:19:49 up 108 days, 20:03, 12 users,  load average: 0.06, 0.06, 0.28
0 %, 1 MiB
A
B
C1
C2
D
E1
E2
E3
E4
MAP
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShad ...[cut]... vents.zeromq.ZeroMqEventBus --reject-unsupported-caps true --tracing false
   2134 nobody    0.8 108-20:02:35 /bin/node_exporter --path.procfs=/host/proc --path.sysfs=/host/sys --path.rootfs=/rootfs --collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($|/)
2771535 root      0.6  2-19:38:23 /usr/sbin/tailscaled --state=/var/lib/tailscale/tailscaled.state --socket=/run/tailscale/tailscaled.sock --port=41641
   1803 root      0.2 108-20:02:45 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
1216655 1200      0.1 97-18:34:30 python3 /opt/bin/video_recorder.py
RUN [Outline production modules used by the cost stage] cd /c/Projects/ReST-GI && grep -n "^def \|^class \|^    def " experiments/ecro_mat94/mat94.py | head -60; grep -n "^def \|^class \|^    def " experiments/gm1_explicit87/quad87.py experiments/gm1_explicit87/xport87.py | head -60; ls experiments/ecro_mat94/ ; grep -rn "E87" experiments/*/fx92.py | head
RUN [Check server state and WL96A run directory] ssh -o ConnectTimeout=20 LabServer63 'uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; ls ~/rest_x97/ 2>/dev/null; cat ~/rest_x95a/env.sh; ls ~/rest_x96a ~/rest_x96a/code ~/rest_x96a/results | head -60; nproc; ps -eo pid,user,pcpu,etime,cmd --sort=-pcpu | head -8'
RUN [Read mat94 setup functions and fx92] cd /c/Projects/ReST-GI/experiments && sed -n 1,41p ecro_mat94/mat94.py; sed -n 140,232p ecro_mat94/mat94.py; cat ecro_mat94/fx92.py 2>/dev/null | head -40; ls */fx92.py
  -> """WL94 (protocol ecro_mat94): per-patch material machinery, Stage A tests, Stage B frozen material counterfactuals on CK2.

Per-patch materials are consumed identically by
  (i)   the learned ECRO operator: its only material input is mat_states[PMAT[patch]] (model86.factors); new Lambert rows are a ...[cut]... rain93 as TR
    return extend_buffer(TR.load_own(R92A, int(seed), torch.device('cuda')))


def load_cand(R, tag):
    import torch
    import model86 as M86
    dev = torch.device('cuda')
    m = extend_buffer(M86.ECRO().to(dev))
    m.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'{tag}.pt'), map_location=dev))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def load_spec(R, R92A, spec):
    k, v = spec.split(':')
    return load_frozen(R92A, v) if k == 'own' else load_cand(R, v)


def mq_ext():
    import diag92 as DG
ecro_attr92/fx92.py
RUN [Read fx92, quad87 Quad/qtable, and XOp] cd /c/Projects/ReST-GI/experiments && cat ecro_attr92/fx92.py | head -70; sed -n 186,300p gm1_explicit87/quad87.py; sed -n 25,75p gm1_explicit87/xport87.py
  -> """WL92 configuration: imports the frozen WL86 / WL86A / WL87 modules unchanged and adds the prospectively defined condition E4.

E4 is chosen from the never-rendered WL87 candidates E3b / E3c by the geometry / visibility-only rule of the protocol and recorded in
<root>/results/e4_choice.json before ...[cut]... [(m, torch.as_tensor(np.nonzero(wid == m)[0], device=dev)) for m in range(self.W.shape[0]) if (wid == m).any()]

    def gather(self, L):
        """L [N, NB, 3] -> radiance arriving along every state ray [N, K, 3]."""
        return L.reshape(-1, 3)[self.idx].view(self.N, K, 3) * self.ok

    def scatter(self, Lr):
        out = torch.zeros(self.N, NB, 3, device=dev)
        for m, ix in self.groups:
            out[ix] = torch.einsum('nkc,kbc->nbc', Lr[ix], self.W[m])
        return out

    def apply(self, L):
        return self.scatter(self.gather(L))


def recurrence(ops, q, first=None):
RUN [Read rest of qtable and outlines of pt86/ecro86] cd /c/Projects/ReST-GI/experiments && sed -n 300,345p gm1_explicit87/quad87.py; grep -n "def \|^class " ecro86/pt86.py | head -70; grep -n "def \|^class " ecro86/ecro86.py | head -40
  -> G = ctx.G
        sph = np.nonzero(G.PSID == S.SID_SPHERE)[0]
        spts = (sph[:, None] * P + np.arange(P)[None]).ravel()
        nsn = np.stack([np.asarray(nn[k], np.float64)[spts] for k in range(3)], 1)
        pp_ = spts // P
        om = valid_omega(nsn, (G.PT[pp_], G.PB[pp_], G.PN[pp_]), 256 ...[cut]... elf, a, b, active):
441:def pack(acc, res, spp):
453:def config(name):
459:def gpu_mem():
463:def main():
22:def sync():
26:def load_model(root, tag):
32:class State:
35:    def __init__(self, G, hit, dbin, seg):
38:    def digest(self):
43:def build_state(ctx):
48:def _seg_obb_t(o, e, obb):
57:def dirty_rays(st, G_old, G_new, moved):
70:def update_state(ctx, st, moved):
82:def respond(model, st, s, use_single=True, use_global=True, factors=None):
90:def render_cfg(root, cfg, model_tag, variant, seed, spp, ls=None, plain=False, cam=None, res=None):
118:def contracts(root, tag):
186:def main():
RUN [Read pt86 Ctx set_geom/render/state_rays and ecro86 state code] cd /c/Projects/ReST-GI/experiments && sed -n 84,141p ecro86/pt86.py; sed -n 200,207p ecro86/pt86.py;  sed -n 255,326p ecro86/pt86.py; sed -n 353,378p ecro86/pt86.py; sed -n 18,82p ecro86/ecro86.py
  -> class Ctx:
    """One Mitsuba scene of the family + Dr.Jit tables of the current geometry; set_geom() edits the boxes in place."""

    def __init__(self, geom, bsdfs=None, emitters=None):
        self.mats = Mats()
        g = S.Geom(geom)
        df = {'type': 'diffuse'}
        bs = bsdfs or [df] ...[cut]... _interleave(K)
    return torch.nonzero(d)[:, 0].cpu().numpy()


def update_state(ctx, st, moved):
    """Selective update after ctx.set_geom(new geometry): re-trace only the dirty rays."""
    idx = dirty_rays(st, st.G, ctx.G, moved)
    if len(idx) == 0:
        return State(ctx.G, st.hit, st.dbin, st.seg), 0
    hit, bh, seg = ctx.state_rays(idx)
    h2, b2, s2 = st.hit.copy(), st.dbin.copy(), st.seg.copy()
    h2[idx] = hit; b2[idx] = bh; s2[idx] = seg
    return State(ctx.G, h2, b2, s2), len(idx)


@torch.no_grad()
def respond(model, st, s, use_single=True, use_global=True, factors=None):
RUN [Read model86 GeoState and factors] grep -n "class GeoState" -A 30 experiments/ecro86/model86.py | head -60; grep -n "def factors" -A 45 experiments/ecro86/model86.py | head -80
RUN [Locate cost claims in WL96A worklog] ls docs/worklogs/ | grep -i "^96"; grep -n "B4\|Stage 3\|lifecycle\|1\.04\|0\.92\|0\.93\|Amdahl\|1\.09\|1\.16\|1\.25\|1\.15" docs/worklogs/96A_*.md | head -70
  -> 49:class GeoState:
50-    """Tensors of one geometry on a device."""
51-
52-    def __init__(self, geom, hit, dbin, dev):
53-        g = S.Geom(geom) if isinstance(geom, dict) else geom
54-        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
55-        self.N  ...[cut]... r,n->lcp', f['Q'], u, gs.area)
123-            y = y + torch.einsum('ncrp,lcp->lncr', f['P'], wq)
124-        return torch.einsum('ndcr,lncr->lndc', f['a'], y)
125-
126-    def forward(self, gs, s, **kw):
127-        return self.apply(self.factors(gs), gs, s, **kw)
128-
129-
130-def loss_fn(pred, tgt):
131-    """Relative L2 with a denominator from the target's patch mean over bins (64-bin average: low-noise) + a floor."""
132-    den = tgt.mean(2, keepdim=True) + 0.05 * tgt.mean(dim=(1, 2, 3), keepdim=True)
133-    return (((pred - tgt) / den) ** 2).mean()
134-
135-
136-def rel_l1(pred, tgt):
  -> 96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
6:- **Question.** Given a fixed, physically accurate transport prefix, is there meaningful representability or lifecycle-cost headroom for a learned higher-order ECRO operator over explicit recu ...[cut]...  explicit sweeps on the same state. There is nothing for learning to recover in accuracy. There is also nothing to amortise in cost, because the sweeps are not the lifecycle's bottleneck, and a geometry-conditioned generator costs more than the sweeps.
231:  - lifecycle sources rng 96500; random Λ factors 96600.
243:  - figW3 (lifecycle; colour-inconsistent chain version, kept) and figW3_lifecycle_posthoc (one colour per component, figure-only fix).
250:  - Utilisation by stage: tails 7 %, one render 23 %, one extension field half 3 % (CPU-bound accumulation), Stage 2 fit 98 %, lifecycle 15 %.
RUN [Read model86 GeoState and factors] cd /c/Projects/ReST-GI && grep -n "class GeoState" -A 30 experiments/ecro86/model86.py | head -60; grep -n "def factors" -A 45 experiments/ecro86/model86.py | head -80
RUN [Locate cost claims in WL96A worklog] cd /c/Projects/ReST-GI && ls docs/worklogs/ | grep -i "^96"; grep -n "B4\|Stage 3\|lifecycle\|1\.04\|0\.92\|0\.93\|Amdahl\|1\.09\|1\.16\|1\.25\|1\.15" docs/worklogs/96A_*.md | head -70
RUN [Read WL96A cost section and protocol cost rules] sed -n 122,162p docs/worklogs/96A_*.md; sed -n 176,205p docs/worklogs/96A_*.md | cut -c1-900; python -c "
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in p:
    s=json.dumps(p[k])
    if 'cost' in k.lower() or 'stage3' in k.lower() or 'lifecycle' in s.lower()[:3000]: print(k, s[:2500]); print()
"
  -> ### B4. Lifecycle cost (Stage 3; `cost_s1.0.json`)

Setup:
- Hardware: RTX 3080 Ti 12 GB, run alone (no foreign process).
- Scene: N = 1,270; frame 512² at 64 spp.
- Events: 6 timing-only rigid edits (box A ± 0.15 m in y) and 16 source updates.
- Timing: medians of warm events; no timed event compil ...[cut]... itatively that |arm - F| in the ROI is visibly smallest for XOD / XOP among the tail arms and larger for the learned G arms and for Z (no number was read). (2) full-scale timing probe ~/rest_x96aprobe (stage 0 at full spp: all PASS incl. T_READOUT; tails 26 s; one render ~10 s; one extension half 18 s; Stage 2 two lights 21 s, two fits 178 s at 98 % GPU; lifecycle 18 s, boundary_ok); only walls, GPU utilisation and pass flags were read, no metric or timing value of an arm. No criterion was changed after either run; the only post-dev changes are the JSON-bool / import fixes and the pool sizes."
RUN [Read WL96A cost section and protocol cost rules] cd /c/Projects/ReST-GI && sed -n 122,162p docs/worklogs/96A_*.md; sed -n 176,205p docs/worklogs/96A_*.md | cut -c1-900; python -c "
import json
p=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
for k in p:
    s=json.dumps(p[k])
    if 'cost' in k.lower() or 'stage3' in k.lower() or 'lifecycle' in s.lower()[:3000]: print(k, s[:2500]); print()
"
RUN [Inspect WL91 lifecycle anchors and WL96 ceiling text] python -c "
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
def show(o,ind=0,depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,dict) and depth<3:
                print(' '*ind+str(k)+':'); show(v,ind+2,depth+1)
            else:
                s=json.dumps(v); print(' '*ind+str(k)+': '+s[:300])
show(d)
" | head -120; grep -n "1\.09\|1\.16" docs/worklogs/96_*.md | cut -c1-600 | head -12
  -> foreign_before: ""
gpu: "NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB"
frame_spp: 64
arms:
  C3URX1:
    edit:
      wall_s: 0.12465548515319824
      wall_raw_s: 0.12465548515319824
      events_with_compilation: 0
      compile_s_total: 0.0
      cpu_s: 0.1289848919999983
      gpu_ ...[cut]... "cpu_s": 0.14949977799999914, "gpu_kernel_ms": 24.306943592149764, "parts": {"geometry_update": 0.03034917265176773, "relation_update": 0.014811171218752861, "source_q_table": 0.0317001827
    source_rows: [{"wall_s": 0.15166817232966423, "wall_raw_s": 0.15166817232966423, "compile_s": 0.0, "kernel_cache_misses": 0, "cpu_s": 0.1534472560000033, "gpu_kernel_ms": 57.30959959095344, "parts": {"source_q_table": 0.04079555533826351, "transport_orders": 0.011516692116856575, "table_handover": 0.000209847465
145:  - edit→frame 124.7 / 114.1 = **1.09×**;
146:  - source→frame 75.8 / 65.2 = **1.16×**;
RUN [Inspect WL91 lifecycle anchors and WL96 ceiling text] cd /c/Projects/ReST-GI && python -c "
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
def show(o,ind=0,depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,dict) and depth<3:
                print(' '*ind+str(k)+':'); show(v,ind+2,depth+1)
            else:
                s=json.dumps(v); print(' '*ind+str(k)+': '+s[:300])
show(d)
" | head -120; grep -n "1\.09\|1\.16" docs/worklogs/96_*.md | cut -c1-600 | head -12
RUN [Read refine rule, pmat variants, form_factors and model loader] cd /c/Projects/ReST-GI/experiments && grep -n "def refine" -A 14 gm1_relation88/econ88.py | head -30; sed -n 66,100p ecro_mat94/mat94.py; grep -n "def form_factors" -A 12 ecro86/model86.py; grep -rn "def load_own" -A 12 ecro_stage93/train93.py | head -20; grep -n "^K\b\|^K =\|^NB\|^KAPPA\|SPH_NPHI\|SPH_NZ\|^PSRC\|^D =" ecro86/scene86.py ecro86/pt86.py | head -20
  -> 230:def refine(s, glossy_B=False):
231-    """Patch refinement by factor s (stationary grids, box faces, sphere) and optional second glossy mover-class surface (box B = S1)."""
232-    if not hasattr(S, '_orig_rects'):
233-        S._orig_rects, S._orig_faces = S.stationary_rects, S.box_faces
234-   ...[cut]... (SPH_NPHI * SPH_NZ)
ecro86/pt86.py:26:D = 6
ecro86/pt86.py:29:NB, K, KAPPA, PSRC = S.NB, S.K, S.KAPPA, S.PSRC
ecro86/pt86.py:177:        izs = dr.clamp(dr.floor((q.z + 1.0) * (0.5 * S.SPH_NZ)), 0, S.SPH_NZ - 1)
ecro86/pt86.py:178:        ips = dr.clamp(dr.floor((dr.atan2(q.y, q.x) + math.pi) * (S.SPH_NPHI / TWO_PI)), 0, S.SPH_NPHI - 1)
ecro86/pt86.py:179:        pid = dr.select(dr.eq(sid, S.SID_SPHERE), U(self.SBASE + ips + S.SPH_NPHI * izs), pid_r)
ecro86/pt86.py:186:        z = (g(20) + b) * (2.0 / S.SPH_NZ) - 1.0
ecro86/pt86.py:187:        phi = (g(19) + a) * (TWO_PI / S.SPH_NPHI) - math.pi
RUN [Check server software versions and threading config] ssh LabServer63 '. ~/rest_x95a/env.sh; cat ~/rest_x96a/code/COMMIT; ls ~/rest_x96a/code/experiments | tr "\n" " "; ls ~/rest_x96a/code/experiments/ecro86/data; ls ~/rest_x92a/results/models | head; $PY -c "
import torch, drjit, mitsuba
print(torch.__version__, drjit.__version__, mitsuba.__version__, torch.get_num_threads())
print(torch.__config__.parallel_info())
"; env | grep -i "omp\|kmp\|mkl" ; lscpu | grep -i "model name\|^CPU(s)\|Thread\|Core"; sed -n 1,60p ~/rest_x96a/code/experiments/ecro
  -> a1718f7
8dna_deformation_replication baseline_sup67 dynamic_transport_failure dynamic_transport_validity ecro86 ecro86a ecro_attr92 ecro_attr95a ecro_cf92b ecro_exp94b ecro_local93d ecro_mat94 ecro_mat94r ecro_own92a ecro_q94c ecro_s94d ecro_stage93 ecro_stage93a ecro_stage93b ecro_stage93c ecro_uti ...[cut]... kages, you must '
ImportError: Before importing any packages, you must specify the desired variant of Mitsuba using "mitsuba.set_variant(..)".
The following variants are available: scalar_rgb, scalar_spectral, cuda_ad_rgb, llvm_ad_rgb.
You can also use the MI_DEFAULT_VARIANT environment variable to define a default variant.
CPU(s):                                  32
Model name:                              Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz
Thread(s) per core:                      2
Core(s) per socket:                      8
CPU(s) scaling MHz:                      41%
8:. "$R/env.sh"
RUN [Read original run script for the cost stage] ssh LabServer63 'cat ~/rest_x96a/code/experiments/ecro_util96a/run96a.sh | head -60; cat ~/rest_x96a/env.sh; ls ~/rest_x96a/results/logs | head -20; grep -c . ~/rest_x96a/chain.out; grep -n "COST\|cost" ~/rest_x96a/chain.out | head'
  -> #!/bin/bash
# WL96A chain (ONE detached chain, end to end): links (read-only) -> Stage 0 -> explicit tails -> renders -> Stage 1 analysis
# [-> predeclared extension: 6 field halves, tails, 12-rep renders, analysis] [-> Stage 2 only on S1_HEADROOM]
# -> Stage 3 lifecycle (ALONE on the GPU) [-> scali ...[cut]... /N2.log | tail -1 | cut -d' ' -f2); log "S1_EXT $S1"
fi
if [ "$S1" = S1_HEADROOM ] || [ -n "$DEV" ]; then                # dev exercises the Stage 2 path regardless
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export C80_MI_VARIANT=cuda_ad_rgb
export PYTHONUNBUFFERED=1
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
C1.log
C1.rc
FIN.log
FIN.rc
N1.log
N1.rc
R_p_p_00.log
R_p_p_00.rc
R_p_p_01.log
R_p_p_01.rc
R_p_p_02.log
R_p_p_02.rc
R_p_p_03.log
R_p_p_03.rc
R_p_p_04.log
R_p_p_04.rc
R_p_p_05.log
R_p_p_05.rc
S0.log
S0.rc
1
WRITE c:\Projects\ReST-GI\experiments\audit97\E5\e5_time.py
EDIT c:\Projects\ReST-GI\experiments\audit97\E5\e5_time.py
RUN [Create run wrapper, upload harness, check server idle] cd /c/Projects/ReST-GI/experiments/audit97/E5 && cat > run_e5.sh <<'EOF'
#!/bin/bash
# WL97 E5 timing run: bash run_e5.sh <tag> <s> [extra flags]   (one GPU process; writes ~/rest_x97/E5/out/<tag>.json)
set -u
B=$HOME/rest_x97/E5
. $HOME/rest_x95a/env.sh
tag=$1; s=$2; shift 2
mkdir -p $B/out
rm -f $B/out/$tag.done
{
  echo "PRE $(date -u +%FT%TZ) $(uptime)"
  echo "PRE apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  ( while true; do echo "$(date -u
  -> run_e5.sh:  Bourne-Again shell script, ASCII text executable
e5_time.py: Python script, ASCII text executable
e5_time.py: Python script, ASCII text executable
run_e5.sh:  Bourne-Again shell script, ASCII text executable
13ef6abb3e9211f2
 01:24:20 up 108 days, 20:07, 12 users,  load average: 0.01, 0.03, 0.21
0 %, 1 MiB
[This command modified 1 file you've previously read: experiments\audit97\E5\e5_time.py. Call Read before editing.]
EDIT c:\Projects\ReST-GI\experiments\audit97\E5\e5_time.py
RUN [Create run wrapper, upload harness, check server idle] cd /c/Projects/ReST-GI/experiments/audit97/E5 && cat > run_e5.sh <<'EOF'
#!/bin/bash
# WL97 E5 timing run: bash run_e5.sh <tag> <s> [extra flags]   (one GPU process; writes ~/rest_x97/E5/out/<tag>.json)
set -u
B=$HOME/rest_x97/E5
. $HOME/rest_x95a/env.sh
tag=$1; s=$2; shift 2
mkdir -p $B/out
rm -f $B/out/$tag.done
{
  echo "PRE $(date -u +%FT%TZ) $(uptime)"
  echo "PRE apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  ( while true; do echo "$(date -u
RUN [Launch first timing run at s=1.0 and wait] ssh LabServer63 'cd ~/rest_x97/E5 && setsid nohup bash code/run_e5.sh run1_s1.0 1.0 > /dev/null 2>&1 & sleep 1; echo started'; ssh LabServer63 'cd ~/rest_x97/E5/out; n=0; while [ ! -f run1_s1.0.done ] && [ $n -lt 100 ]; do sleep 5; n=$((n+1)); done; tail -c 1500 run1_s1.0.log; ls -la'
  -> started
 01:25:28 3543661:3543661 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
STAGE:2026-11-16 01:25:28 3543661:3543661 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2026-11-16 01:25:28 3543661:3543661 ActivityProfilerController.cpp:320] Completed Stage: ...[cut]... ofilerController.cpp:320] Completed Stage: Collection
STAGE:2026-11-16 01:25:29 3543661:3543661 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
E5 DONE 1.0 1270 63.2
RC 0
POST 2026-11-16T01:25:30Z  01:25:30 up 108 days, 20:08, 13 users,  load average: 1.27, 0.47, 0.35
total 252
drwxr-xr-x 2 nd nd   4096 Nov 16 01:25 .
drwxr-xr-x 4 nd nd   4096 Nov 16 01:24 ..
-rw-r--r-- 1 nd nd      0 Nov 16 01:25 run1_s1.0.done
-rw-r--r-- 1 nd nd    779 Nov 16 01:25 run1_s1.0.gpu.txt
-rw-r--r-- 1 nd nd 240899 Nov 16 01:25 run1_s1.0.json
-rw-r--r-- 1 nd nd   2793 Nov 16 01:25 run1_s1.0.log
RUN [Fetch run 1 results and summarise events] mkdir -p results/audit97/E5 && scp -q LabServer63:rest_x97/E5/out/run1_s1.0.json LabServer63:rest_x97/E5/out/run1_s1.0.log LabServer63:rest_x97/E5/out/run1_s1.0.gpu.txt results/audit97/E5/ && head -3 results/audit97/E5/run1_s1.0.log && cat results/audit97/E5/run1_s1.0.gpu.txt | tr '\n' '|' && python - <<'EOF'
import json
d=json.load(open('results/audit97/E5/run1_s1.0.json'))
e=d['env']; print({k:e[k] for k in e if k not in('parallel_info',)}); print(e['parallel_info'][:400])
print(d['probe_check
  -> PRE 2026-11-16T01:24:25Z  01:24:25 up 108 days, 20:07, 12 users,  load average: 0.01, 0.03, 0.20
PRE apps: 
STAGE:2026-11-16 01:25:26 3543661:3543661 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
01:24:25 0 %, 1 MiB|01:24:27 1 %, 312 MiB|01:24:29 18 %, 1082 MiB|01:24:31 27 %, 1460 MiB ...[cut]... coefficients': 6.02, 'generator': 9.19, 'source_q_table': 32.45, 'prefix_order': 1.94, 'lambda_apply': 0.2, 'table_handover': 0.19, 'render': 60.41}
     cpu {'geometry_update': 32.3, 'relation_update': 20.7, 'relation_coefficients': 6.0, 'generator': 135.0, 'source_q_table': 477.1, 'prefix_order': 1.9, 'lambda_apply': 0.2, 'table_handover': 0.2, 'render': 61.4}
 LAM24 source  n=4 TOTAL med 85.99 [85.90,86.14] min 85.86 max 86.36 cpu 87.4 sumparts 85.55 drjit_exec 24.23 miss 0.0
     {'source_q_table': 24.43, 'prefix_order': 2.0, 'lambda_apply': 0.2, 'table_handover': 0.19, 'render': 58.73}
RUN [Fetch run 1 results and summarise events] cd /c/Projects/ReST-GI && mkdir -p results/audit97/E5 && scp -q LabServer63:rest_x97/E5/out/run1_s1.0.json LabServer63:rest_x97/E5/out/run1_s1.0.log LabServer63:rest_x97/E5/out/run1_s1.0.gpu.txt results/audit97/E5/ && head -3 results/audit97/E5/run1_s1.0.log && cat results/audit97/E5/run1_s1.0.gpu.txt | tr '\n' '|' && python - <<'EOF'
import json
d=json.load(open('results/audit97/E5/run1_s1.0.json'))
e=d['env']; print({k:e[k] for k in e if k not in('parallel_info',)}); print(e['parallel_info'][:
RUN [Summarise isolated component timings from run 1] python - <<'EOF'
import json
d=json.load(open('results/audit97/E5/run1_s1.0.json'))
def tab(C):
    for k,v in C.items():
        w=v['wall_ms']
        print(f"{k:22s} n={v['n']:2d} wall {w['med']:8.3f} [{w['q25']:.3f},{w['q75']:.3f}] min {w['min']:.3f} max {w['max']:.3f} | cpu {v['cpu_ms']['med']:7.2f} main {v['cpu_main_ms']['med']:7.2f} | drjit exec {v['exec_ms']['med']:7.3f} codegen {v['codegen_ms']['med']:6.3f} nk {v['nk']['med']:.0f} miss {v['miss']['sum']:.0f} compile {v['compile_ms']['su
  -> q_table                n=18 wall   24.131 [24.086,24.189] min 24.052 max 24.893 | cpu   24.64 main   24.15 | drjit exec   1.844 codegen  0.419 nk 6 miss 0 compile 0
q_table_src3           n=18 wall   24.113 [24.100,24.134] min 24.022 max 24.290 | cpu   24.67 main   24.13 | drjit exec   1.842 codege ...[cut]... 
"gpu_ms_per_call": 1.5819,
"gpu_events_per_call": 144.0
},
"lambda24_apply": {
"gpu_ms_per_call": 0.10465,
"gpu_events_per_call": 2.0
},
"lambda64_apply": {
"gpu_ms_per_call": 0.19169999999999998,
"gpu_events_per_call": 2.0
},
"generator_factors": {
"gpu_ms_per_call": 4.204,
"gpu_events_per_call": 107.0
},
"xop_build_residentW": {
"gpu_ms_per_call": 2.0145999999999997,
"gpu_events_per_call": 16.0
},
"q_upload": {
"gpu_ms_per_call": 0.12279999999999999,
"gpu_events_per_call": 1.0
},
"dirty_rays": {
"gpu_ms_per_call": 4.433649999999999,
"gpu_events_per_call": 63.0
}
}
RUN [Summarise isolated component timings from run 1] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/audit97/E5/run1_s1.0.json'))
def tab(C):
    for k,v in C.items():
        w=v['wall_ms']
        print(f"{k:22s} n={v['n']:2d} wall {w['med']:8.3f} [{w['q25']:.3f},{w['q75']:.3f}] min {w['min']:.3f} max {w['max']:.3f} | cpu {v['cpu_ms']['med']:7.2f} main {v['cpu_main_ms']['med']:7.2f} | drjit exec {v['exec_ms']['med']:7.3f} codegen {v['codegen_ms']['med']:6.3f} nk {v['nk']['med']:.0f} miss {v['miss']['sum']:.0f} co
RUN [Patch harness flags, upload, launch control and scaling chain] cd /c/Projects/ReST-GI/experiments/audit97/E5 && python - <<'EOF'
p='e5_time.py'; s=open(p,newline='').read()
a="LAM = {r: (torch.rand(3, D, r, generator=gen).cuda(), torch.rand(3, D, r, generator=gen).cuda()) for r in (24, 64)}"
b="RANKS = (24, 64) if D < 400000 else (24,)                # keep the process below ~4 GB at the largest refinement\n    LAM = {r: (torch.rand(3, D, r, generator=gen).cuda(), torch.rand(3, D, r, generator=gen).cuda()) for r in RANKS}"
assert a in s; s=s.replace(a,b)
a=
  -> chain_e5.sh: Bourne-Again shell script, ASCII text executable
e5_time.py:  Python script, ASCII text executable
9675abd00f06e747  e5_time.py
98082d44653cbbb2  run_e5.sh
0283252cf7327bfe  chain_e5.sh
 01:26:50 up 108 days, 20:10, 12 users,  load average: 0.39, 0.38, 0.33
apps: 
started
[This command modified 1 file you've previously read: experiments\audit97\E5\e5_time.py. Call Read before editing.]
WRITE c:\Projects\ReST-GI\experiments\audit97\E5\analyze_e5.py
RUN [Wait for the control and scaling chain] ssh LabServer63 'cd ~/rest_x97/E5/out; n=0; while [ ! -f chain2.done ] && [ $n -lt 105 ]; do sleep 5; n=$((n+1)); done; ls; for f in run*.log; do echo "== $f"; grep -h "PRE\|E5 DONE\|RC\|POST\|Error\|Traceback" $f | cut -c1-200; done'
  -> chain2.done
chain2.log
run1_s1.0.done
run1_s1.0.gpu.txt
run1_s1.0.json
run1_s1.0.log
run2_passive_s1.0.done
run2_passive_s1.0.gpu.txt
run2_passive_s1.0.json
run2_passive_s1.0.log
run3_s1.0.done
run3_s1.0.gpu.txt
run3_s1.0.json
run3_s1.0.log
run4_s1.5.done
run4_s1.5.gpu.txt
run4_s1.5.json
run4_s1.5.l ...[cut]... 
POST 2026-11-16T01:29:08Z  01:29:08 up 108 days, 20:12, 13 users,  load average: 1.65, 1.17, 0.66
== run5_s2.0.log
PRE 2026-11-16T01:29:08Z  01:29:08 up 108 days, 20:12, 13 users,  load average: 1.65, 1.17, 0.66
PRE apps: 
E5 DONE 2.0 5080 62.5
RC 0
POST 2026-11-16T01:30:13Z  01:30:13 up 108 days, 20:13, 13 users,  load average: 1.33, 1.18, 0.70
== run6_s3.0.log
PRE 2026-11-16T01:30:13Z  01:30:13 up 108 days, 20:13, 13 users,  load average: 1.33, 1.18, 0.70
PRE apps: 
E5 DONE 3.0 11430 120.4
RC 0
POST 2026-11-16T01:32:16Z  01:32:16 up 108 days, 20:15, 13 users,  load average: 1.07, 1.13, 0.75
RUN [Fetch all runs and consolidate] scp -q "LabServer63:rest_x97/E5/out/run*" results/audit97/E5/ && rm -f results/audit97/E5/*.done && python experiments/audit97/E5/analyze_e5.py 2>&1 | cut -c1-1500
  -> <persisted-output>
Output too large (33.9KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\b12o5w1hd.txt

Preview (first 2KB):
## 1. Original vs reproduced, R0 (ms; medians)
R0 edit: orig 144.1 {'geometry_update': 30.18, ...[cut]... 'geometry_update': 31.06, 'relation_update': 18.14, 'relation_coefficients': 4.93, 'source_q_table': 25.55, 'prefix_order': 1.88, 'tail_orders': 3.32, 'table_handover': 0.2, 'render': 60.07}
   run3_s1.0|D_R0_after_LAM: 146.0 [145.8,146.2] ratio 1.013 {'geometry_update': 31.24, 'relation_update': 18.13, 'relation_coefficients': 4.93, 'source_q_table': 25.42, 'prefix_order': 1.88, 'tail_orders': 3.31, 'table_handover': 0.2, 'render': 60.1}
R0 source: orig 88.5 {'source_q_table': 24.37, 'prefix_order': 2.05, 'tail_orders': 3.31, 'table_handover': 0.21, 'render': 58.51}
...
</persisted-output>
RUN [Fetch all runs and consolidate] cd /c/Projects/ReST-GI && scp -q "LabServer63:rest_x97/E5/out/run*" results/audit97/E5/ && rm -f results/audit97/E5/*.done && python experiments/audit97/E5/analyze_e5.py 2>&1 | cut -c1-1500
RUN [Print U, generator and scaling sections] python experiments/audit97/E5/analyze_e5.py > results/audit97/E5/e5_tables.txt 2>&1; python - <<'EOF'
import re
t=open('results/audit97/E5/e5_tables.txt').read()
secs=t.split('\n## ')
for s in secs:
    h=s.split('\n')[0]
    if h.startswith('2.') or h.startswith('3.') or h.startswith('6.'):
        print('##',s[:9000])
    if h.startswith('1.'):
        i=s.find('R0 source'); print(s[i:i+2400])
EOF
  -> ## 2. U (R0 / arm), medians
{
"orig": {
"LAM24": {
"edit": 0.9321302015681585,
"source": 1.040492233086946
},
"LAM64": {
"edit": 0.9209615109936242,
"source": 1.038301742024448
},
"ZERO": {
"edit": 1.038028573853174,
"source": 1.0428847966464583
}
},
"run1_s1.0|A_orig_order": {
"LAM ...[cut]... _apply': 0.86, 'lambda24_apply_gpu': 0.89, 'build_state_full': 325.175, 'sg_Geom_construct': 206.107, 'q_table_exec': 16.674, 'render_512_64_exec': 23.543, 'R0_edit': 640.261, 'ZERO_edit': 634.765, 'U_ZERO_edit': 1.009, 'tail_in_edit': 7.247, 'R0_source': 128.887, 'ZERO_source': 122.23, 'U_ZERO_source': 1.054, 'tail_in_source': 6.569, 'U_ZERO_edit_noframe': 1.014, 'R0_edit_noframe': 573.513, 'U_ZERO_source_noframe': 1.116, 'R0_source_noframe': 70.413, 'dirty_rays': 449970, 'relation_rays': 5852160, 'selective_equals_full': True}

written c:\Projects\ReST-GI\results\audit97\E5\e5_summary.json
RUN [Print U, generator and scaling sections] cd /c/Projects/ReST-GI && python experiments/audit97/E5/analyze_e5.py > results/audit97/E5/e5_tables.txt 2>&1; python - <<'EOF'
import re
t=open('results/audit97/E5/e5_tables.txt').read()
secs=t.split('\n## ')
for s in secs:
    h=s.split('\n')[0]
    if h.startswith('2.') or h.startswith('3.') or h.startswith('6.'):
        print('##',s[:9000])
    if h.startswith('1.'):
        i=s.find('R0 source'); print(s[i:i+2400])
EOF
RUN [Print convention ceilings and remaining component details] python - <<'EOF'
t=open('results/audit97/E5/e5_tables.txt').read()
i=t.find('## 5.'); j=t.find('## 6.'); print(t[i:j])
import json
d=json.load(open('results/audit97/E5/run2_passive_s1.0.json'))
for k,v in d['events']['A_orig_order']['summary'].items():
    if k.startswith('R0|edit') or k.startswith('LAM24|edit'): print(k, round(v['wall_ms']['med'],2), round(v['cpu_ms']['med'],1))
s=json.load(open('results/audit97/E5/e5_summary.json'))
for rk in ('run1_s1.0','run3_s1.0','run6_s3.0'):
    print(rk
  -> ## 5. Ceilings under conventions (ZERO-cost replacement of the three tail orders)
-- run1_s1.0 N=1270  inputs {'tail_wall': 3.168, 'tail_gpu': 1.582, 'src_table_wall': 29.022, 'edit_table_wall': 83.436, 'src_table_gpu': 4.077, 'edit_table_gpu_hi': 13.307, 'edit_table_gpu_lo': 4.176}
  512x512x64   ...[cut]... .4.6', 'mitsuba': '3.5.1', 'torch_threads': 16, 'env': {}, 'total_s': 62.53603525273502, 'torch_peak_mib': 1453.849609375, 'code_commit': 'a1718f7'}, 'run6_s3.0': {'uptime_before': '01:30:15 up 108 days, 20:13, 13 users,  load average: 1.30, 1.18, 0.71', 'uptime_after': '01:32:15 up 108 days, 20:15, 13 users,  load average: 1.07, 1.13, 0.75', 'apps_before': '3546400, 282 MiB', 'apps_after': '3546400, 6324 MiB', 'torch': '2.3.1+cu121', 'drjit': '0.4.6', 'mitsuba': '3.5.1', 'torch_threads': 16, 'env': {}, 'total_s': 120.3592035304755, 'torch_peak_mib': 2447.4970703125, 'code_commit': 'a1718f7'}}
RUN [Print convention ceilings and remaining component details] cd /c/Projects/ReST-GI && python - <<'EOF'
t=open('results/audit97/E5/e5_tables.txt').read()
i=t.find('## 5.'); j=t.find('## 6.'); print(t[i:j])
import json
d=json.load(open('results/audit97/E5/run2_passive_s1.0.json'))
for k,v in d['events']['A_orig_order']['summary'].items():
    if k.startswith('R0|edit') or k.startswith('LAM24|edit'): print(k, round(v['wall_ms']['med'],2), round(v['cpu_ms']['med'],1))
s=json.load(open('results/audit97/E5/e5_summary.json'))
for rk in ('run1_s1.0','run3_s1.0',
RUN [Check WL96 cross-workload ceiling arithmetic against WL91 anchors] sed -n 138,148p docs/worklogs/96_*.md | cut -c1-700; grep -n "transport_orders" experiments/gm1_c3urx1_91/*.py | head -8; grep -n "D_ORD\|n_orders\|orders" experiments/gm1_explicit87/protocol/*.json | head -5; python -c "
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
for a in d['arms']:
    for k in ('edit','source_single_light'):
        x=d['arms'][a][k]; t=x['parts']['transport_orders']; print(a,k,round(x['wall_s']*1e3,2),round(t*1e3,2),round(x['wall_s']/(x['
  -> Exit code 1
  - The CK2 grazing omni light, Lambert receivers, no shadow edge across the panel.
  - The other lights are oracle-only.
  - Agreement of 0.068 against n0 0.059 leaves up to about 0.03 of unresolved oracle bias.
- **Image-level relevance on CK2 is unresolved** (§A).

**Cost / scaling.** ...[cut]... , 0.0) + T_['transport_s']
experiments/gm1_c3urx1_91/e2e91.py:143:            B = self.C.c3ur_table(self.rc, q); ck.part('transport_orders')
experiments/gm1_c3urx1_91/e2e91.py:147:            B = RL.recurrence([self.op], q)[0]; ck.part('transport_orders')
45:  "orders": 4,
Traceback (most recent call last):
  File "<string>", line 6, in <module>
KeyError: 'transport_orders'
C3URX1 edit 124.66 10.57 1.0927
C3URX1 source_single_light 75.81 9.4 1.1416
C3UR edit 138.19 11.41 1.09
C3UR source_single_light 92.22 10.94 1.1346
R0 edit 145.41 5.57 1.0399
R0 source_single_light 85.44 5.56 1.0697
RUN [Check WL96 cross-workload ceiling arithmetic against WL91 anchors] cd /c/Projects/ReST-GI && sed -n 138,148p docs/worklogs/96_*.md | cut -c1-700; grep -n "transport_orders" experiments/gm1_c3urx1_91/*.py | head -8; grep -n "D_ORD\|n_orders\|orders" experiments/gm1_explicit87/protocol/*.json | head -5; python -c "
import json
d=json.load(open('results/gm1_c3urx1_91/results/econ/life91.json'))
for a in d['arms']:
    for k in ('edit','source_single_light'):
        x=d['arms'][a][k]; t=x['parts']['transport_orders']; print(a,k,round(x['wall_s']*1e3,2),round(t*1e3
RUN [Time generator sub-steps on the server] cd /c/Projects/ReST-GI/experiments/audit97/E5 && cat > e5_gen.py <<'EOF'
"""WL97 E5: sub-steps of the ECRO-class generator (model86.ECRO.factors, OWN s8601) on CK2, N = 1,270 (own copy of the factor
construction split into steps, for timing only; the full production call is timed alongside as the control).
  python e5_gen.py <R_own> <CODE> <R92A> <out.json>"""
import json, os, sys, time
import numpy as np
R_OWN, CODE, R92A, OUT = sys.argv[1:5]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.j
  -> 01:34:02 up 108 days, 20:17, 12 users,  load average: 0.25, 0.82, 0.68
apps: 
511f2c6715e6d2cf
{
"N": 1270,
"R": 16,
"RHO": 24,
"rounds": 3,
"apps": "3547516, 656 MiB",
"uptime": "01:34:04 up 108 days, 20:17, 12 users,  load average: 0.25, 0.82, 0.68",
"split_equals_production": {
"c": 0.0,
"a": 0.0 ...[cut]... : {
"wall_med_ms": 1.9829953089356422,
"wall_q25": 1.9791275262832642,
"wall_q75": 1.986396498978138,
"gpu_ms": 0.85435,
"n": 20
},
"heads_per_bin_c_a": {
"wall_med_ms": 2.8332360088825226,
"wall_q25": 2.832002006471157,
"wall_q75": 2.834885846823454,
"gpu_ms": 2.75775,
"n": 20
},
"heads_global_Q_P": {
"wall_med_ms": 0.2612629905343056,
"wall_q25": 0.25879405438899994,
"wall_q75": 0.2682753838598728,
"gpu_ms": 0.11865,
"n": 20
},
"global_only_generator(tokens+Q,P)": {
"wall_med_ms": 2.2156108170747757,
"wall_q25": 2.209507394582033,
"wall_q75": 2.2217873483896255,
"gpu_ms": 0.9729,
"n": 20
}
}
RUN [Record input and output hashes] ssh LabServer63 'cd ~/rest_x96a/code/experiments; sha256sum ecro86/pt86.py ecro86/ecro86.py ecro86/model86.py ecro86/scene86.py gm1_explicit87/quad87.py gm1_explicit87/xport87.py ecro_mat94/mat94.py ecro_local93d/addendum_checker/scene_ck2.json ecro_util96a/w96a.py ~/rest_x92a/results/models/own_s8601.pt ~/rest_x96a/results/cost_s1.0.json | cut -c1-12,65-; cd ~/rest_x97/E5; sha256sum code/* out/*.json | cut -c1-12,65-; uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  -> 912e68acd314  ecro86/pt86.py
0ef09a869397  ecro86/ecro86.py
72a728b81e49  ecro86/model86.py
b75a2d9cbae1  ecro86/scene86.py
cfa9c5c4e854  gm1_explicit87/quad87.py
aba6e5765c22  gm1_explicit87/xport87.py
15268a6a598e  ecro_mat94/mat94.py
977ba122ab32  ecro_local93d/addendum_checker/scene_ck2.json
afb ...[cut]... nts/audit97/E5/chain_e5.sh
511f2c6715e6 *experiments/audit97/E5/e5_gen.py
9675abd00f06 *experiments/audit97/E5/e5_time.py
98082d44653c *experiments/audit97/E5/run_e5.sh
6a8f66ac308c *results/audit97/E5/e5_summary.json
a9a010a6481c *results/audit97/E5/gen_s1.0.json
064728f31b25 *results/audit97/E5/run1_s1.0.json
1a8557b5f9e2 *results/audit97/E5/run2_passive_s1.0.json
e50632ef6a4b *results/audit97/E5/run3_s1.0.json
9b953cdecb75 *results/audit97/E5/run4_s1.5.json
6bddb47a8fb3 *results/audit97/E5/run5_s2.0.json
0e8ec749378c *results/audit97/E5/run6_s3.0.json
?? docs/audits/
?? experiments/audit97/
RUN [Record input and output hashes] cd /c/Projects/ReST-GI && ssh LabServer63 'cd ~/rest_x96a/code/experiments; sha256sum ecro86/pt86.py ecro86/ecro86.py ecro86/model86.py ecro86/scene86.py gm1_explicit87/quad87.py gm1_explicit87/xport87.py ecro_mat94/mat94.py ecro_local93d/addendum_checker/scene_ck2.json ecro_util96a/w96a.py ~/rest_x92a/results/models/own_s8601.pt ~/rest_x96a/results/cost_s1.0.json | cut -c1-12,65-; cd ~/rest_x97/E5; sha256sum code/* out/*.json | cut -c1-12,65-; uptime; nvidia-smi --query-compute-apps=pid,used_me
