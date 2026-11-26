"""WL115 Stage C training of ECRO-CAR (frozen protocol sr115.json stage_B_frozen.stage_C).

  python train115.py ROOT ARM [STEPS]     ARM in CAR1 (seed 11330001), CAR2 (seed 11330002)
Dense / coarse parts: the WL114 B arm recipe unchanged (train113 OWN losses, schedule, 24,000 steps, final step) on the D114 files with
full-depth targets (identical values: D114 float16 targets are kept in float16 in host memory and converted on the device, exactly as
train114.Item's float32 conversion). Added: the same OWN form on the CAR-16 active children (D115). Matched controls: WL114 B1 / B2.
"""
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for d in (HERE, HERE.parent / 'ecro_fd114', HERE.parent / 'ecro_sss113', HERE.parent / 'ecro86', HERE.parent / 'ecro_material100'):
    sys.path.insert(0, str(d))
import common115 as C  # noqa: E402
import train113 as TR  # noqa: E402
import model86 as M  # noqa: E402
import model115 as MC  # noqa: E402
import candidate115 as CA  # noqa: E402
import common100 as C100  # noqa: E402

DEV = TR.DEV
ARMS = {'CAR1': 11330001, 'CAR2': 11330002}
D114 = C.WL114_ROOT / 'stageC' / 'data'
CHILD = CA.child_index_maps()          # [16 cells, 16 children] -> fine 16 x 16 index


class Item:
    def __init__(self, p114, p115, val=False):
        z = np.load(p114); y = np.load(p115)
        self.g = {k: z[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat', 'hit', 'dbin', 's', 'desc', 'support')}
        def avg(k, zz):
            a = zz[k]
            return (0.5 * (a.astype(np.float32) + zz[k + 'b'].astype(np.float32))) if val else a
        self.c = (avg('c_f2', z), avg('c_fhF', z), z['c_m'] & (z['c_mb'] if val else True))
        self.s = (avg('s_f2', z), avg('s_fhF', z), z['s_m'] & (z['s_mb'] if val else True))
        w = y['w']; pids = y['pids']; loc = -np.ones(len(w), np.int64); loc[pids] = np.arange(len(pids))
        ap, asub = np.nonzero(w > 0)
        fi = CHILD[asub]                                                        # [A, 16]
        ip = loc[ap]
        self.act_pid, self.act_sub, self.act_w = ap, asub, w[ap, asub]
        self.cdesc = y['cdesc'][ip[:, None], fi]                                # [A, 16, 5]
        g = lambda k: y[k][:, ip[:, None], fi]                                   # [L, A, 16, NB, (3)]
        if val:
            self.ch = (0.5 * (g('c16_f2').astype(np.float32) + g('c16_f2b').astype(np.float32)), 0.5 * (g('c16_fhF').astype(np.float32) + g('c16_fhFb').astype(np.float32)),
                       g('c16_m') & g('c16_mb'))
        else:
            self.ch = (g('c16_f2'), g('c16_fhF'), g('c16_m'))
        self.path = (p114, p115)

    def tensors(self):
        z = self.g
        g = type('G', (), {})(); g.N = len(z['pc'])
        g.PC, g.PN, g.PT, g.PAREA, g.PMAT = z['pc'], z['pn'], z['pt'], z['parea'], z['pmat']
        gs = M.GeoState(g, z['hit'], z['dbin'], DEV)
        gs.sup = torch.as_tensor(np.flatnonzero(z['support']), device=DEV); gs.desc = torch.as_tensor(z['desc'], device=DEV)
        gs.act_pid = torch.as_tensor(self.act_pid, device=DEV); gs.act_sub = torch.as_tensor(self.act_sub, device=DEV)
        gs.cdesc = torch.as_tensor(self.cdesc, device=DEV)
        t = lambda a: torch.as_tensor(np.asarray(a), device=DEV).float()
        tm = lambda a: torch.as_tensor(np.asarray(a), device=DEV)
        c = (t(self.c[0]), t(self.c[1]), tm(self.c[2]))
        s = (t(self.s[0]), t(self.s[1]), tm(self.s[2]))
        ch = (t(self.ch[0]), t(self.ch[1]), tm(self.ch[2]))
        return gs, t(z['s']), c, s, ch


def forward(model, gs, s):
    f = model.factors(gs)
    Ba, Bsa, Bca = model.outputs_car(f, gs, s, use_global=False)
    Bb, Bsb, Bcb = model.outputs_car(f, gs, s, use_single=False)
    return Ba, Bb, Bsa, Bsb, Bca, Bcb


@torch.no_grad()
def evaluate(model, items):
    err = np.zeros(4); tru = np.zeros(4)
    for it in items:
        gs, s, c, sb, ch = it.tensors()
        a, b, sa, sbb, ca, cb = forward(model, gs, s)
        for i, (p, q, tt) in enumerate(((a, b, c), (sa, sbb, sb), (ca, cb, ch))):
            mm = tt[2].unsqueeze(-1)
            err[i] += float(((p + q - tt[0] - tt[1]).abs() * mm).sum()); tru[i] += float(((tt[0] + tt[1]) * mm).sum())
        up = (sa + sbb)[:, gs.act_pid, gs.act_sub].unsqueeze(2)                    # dense parent cell vs the child targets
        mm = ch[2].unsqueeze(-1)
        err[3] += float(((up - ch[0] - ch[1]).abs() * mm).sum()); tru[3] += float(((ch[0] + ch[1]) * mm).sum())
    r = err / np.maximum(tru, 1e-20)
    return {'coarse_whole_rel_l1': float(r[0]), 'refined_whole_rel_l1': float(r[1]), 'child_whole_rel_l1': float(r[2]),
            'dense_parent_vs_child_targets_rel_l1': float(r[3])}


def noise_floor(items):
    e = 0.; t = 0.
    for it in items:
        y = np.load(it.path[1]); w = y['w']; pids = y['pids']
        loc = -np.ones(len(w), np.int64); loc[pids] = np.arange(len(pids))
        ap, asub = np.nonzero(w > 0); fi = CHILD[asub]; ip = loc[ap]
        g = lambda k: y[k][:, ip[:, None], fi].astype(np.float32)
        a = g('c16_f2') + g('c16_fhF'); b = g('c16_f2b') + g('c16_fhFb'); m = (y['c16_m'][:, ip[:, None], fi] & y['c16_mb'][:, ip[:, None], fi])[..., None]
        e += float((np.abs(a - b) * m).sum()) / 2; t += float((0.5 * (a + b) * m).sum())
    return {'child_half_noise_rel_l1': e / max(t, 1e-20)}


def main(root, arm, steps=None):
    root = Path(root); seed = ARMS[arm]
    P = dict(TR.P, seed=seed)
    steps = int(steps or C.dv(P['steps'], 200))
    torch.manual_seed(P['seed']); np.random.seed(P['seed'])
    dd = root / 'stageC' / 'd115'
    trn = [f'g{i:05d}.npz' for i in range(C.dv(192, 6))]; van = [f'g{i:05d}.npz' for i in range(C.dv(16, 2))]
    train = [Item(D114 / 'train' / n, dd / 'train' / n) for n in trn]
    val = [Item(D114 / 'val' / n, dd / 'val' / n, val=True) for n in van]
    model = MC.ECROCAR().to(DEV)
    model.mat_states = torch.as_tensor(C100.materials('sss'), device=DEV)
    shared = [p for n_, p in model.named_parameters() if not (n_.startswith('head_r') or n_ == 'child_emb')]
    init_sha_shared = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in shared)).hexdigest()
    ref = json.loads((C.WL114_ROOT / 'stageC' / ('B1' if arm == 'CAR1' else 'B2') / 'training.json').read_text())
    init_match = init_sha_shared == ref['init_param_sha256']
    print('INIT_SHARED_MATCH', init_match, flush=True)
    assert init_match or C.DEV, 'STOP_INIT_MISMATCH'
    count = sum(x.numel() for x in model.parameters())
    out = root / 'stageC' / arm; out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=P['lr']); rng = np.random.default_rng(P['seed']); order = []; log = []; run = []
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    print('TRAIN_START', arm, seed, count, len(train), flush=True)
    for step in range(steps):
        if not order:
            order = rng.permutation(len(train)).tolist()
        it = train[order.pop()]
        lr = P['lr'] * (step + 1) / P['warmup'] if step < P['warmup'] else \
            P['lr_final'] + .5 * (P['lr'] - P['lr_final']) * (1 + math.cos(math.pi * (step - P['warmup']) / max(steps - P['warmup'], 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        gs, s, c, sb, ch = it.tensors()
        a, b, sa, sbb, ca, cb = forward(model, gs, s)
        losses = TR.own(a, b, *c) + TR.own(sa, sbb, *sb) + TR.own(ca, cb, *ch)
        loss = sum(losses); assert torch.isfinite(loss), 'STOP_NONFINITE_LOSS'
        opt.zero_grad(set_to_none=True); loss.backward()
        nrm = torch.nn.utils.clip_grad_norm_(model.parameters(), P['clip']); assert torch.isfinite(nrm), 'STOP_NONFINITE_GRADIENT'
        opt.step(); run.append([float(x.detach()) for x in losses])
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'losses': np.mean(run, 0).tolist(), 'seconds': time.perf_counter() - t0}; run = []
            if (step + 1) % 4000 == 0 or step + 1 == steps:
                row['val'] = evaluate(model, val)
            log.append(row); print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); train_s = time.perf_counter() - t0
    ck = out / 'model.pt'; torch.save(model.state_dict(), ck)
    C.jd({'arm': arm, 'seed': seed, 'steps': steps, 'params': count, 'init_shared_sha256': init_sha_shared, 'init_shared_matches_B': init_match,
          'train_s': train_s, 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'log': log, 'validation': evaluate(model, val),
          'target_noise': noise_floor(val), 'checkpoint_sha256': C.sha(ck), 'recipe': P, 'torch': torch.__version__,
          'gpu': torch.cuda.get_device_name(0)}, out / 'training.json')
    print('TRAIN_COMPLETE', arm, train_s, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
