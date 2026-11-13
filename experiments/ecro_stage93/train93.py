"""WL93 controlled training of the local-operator arms (global branch = frozen WL92A own_s<seed>).

  python train93.py <R> <R86> <R92A> <arm> <seed> [--dev]
arms: T0 (tied, OWN contract), TQ (tied, OWN + source-stage), U0 (untied, OWN), UQ (untied, OWN + source-stage),
      C0 / CQ (capacity-matched tied control, OWN / OWN + source-stage)
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93.json')))
ARMS = P['arms']


class Item:
    """One WL86 training geometry with the WL92A component target t2 and the WL93 source-stage target q (no f2p, to save memory)."""

    def __init__(self, R, R92A, sp, fn):
        import train86 as TR
        self.it = TR.Item(fn)
        z2 = np.load(os.path.join(R92A, 'data', sp + '_f2', os.path.basename(fn)))
        self.f2 = z2['f2']; self.f2b = z2['f2b'] if 'f2b' in z2.files else None
        self.q = np.load(os.path.join(R, 'data', sp + '_q', os.path.basename(fn)))['q']

    def tens(self, dev):
        T = lambda a: torch.as_tensor(a.astype(np.float32), device=dev)
        t2 = T(self.f2) if self.f2b is None else 0.5 * (T(self.f2) + T(self.f2b))
        return torch.as_tensor(self.it.s, device=dev), self.it.target(), t2, T(self.q)


def population(R, R86, R92A, sp, limit=None):
    from concurrent.futures import ThreadPoolExecutor
    ex = {n for n, _, _ in json.load(open(os.path.join(R92A, 'results', 'exclusion.json')))['excluded'].get(sp, [])}
    fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))) if os.path.basename(f) not in ex
           and os.path.exists(os.path.join(R, 'data', sp + '_q', os.path.basename(f))) and os.path.exists(os.path.join(R92A, 'data', sp + '_f2', os.path.basename(f)))][:limit]
    with ThreadPoolExecutor(16) as e:
        return list(e.map(lambda f: Item(R, R92A, sp, f), fns))


def load_own(R92A, seed, dev):
    import model86 as M
    m = M.ECRO().to(dev)
    m.load_state_dict(torch.load(os.path.join(R92A, 'results', 'models', f'own_s{seed}.pt'), map_location=dev))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def sd_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    R, R86, R92A, arm, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    import model93 as MM
    dev = torch.device('cuda')
    A = ARMS[arm]; TRc = P['training']
    out = os.path.join(R, 'results', 'models'); os.makedirs(out, exist_ok=True)
    tag = f'{arm}_s{seed}'
    torch.manual_seed(seed); np.random.seed(seed)
    own = load_own(R92A, seed, dev)
    for p in own.parameters():
        p.requires_grad_(False)
    net = MM.Local(own, A['mode'], cm_seed=seed + 9300).to(dev)
    init_hash = sd_hash(net)
    t0 = time.time()
    items = population(R, R86, R92A, 'train', 48 if DEV else None)
    val = population(R, R86, R92A, 'val', 8 if DEV else None)
    print('loaded', len(items), len(val), round(time.time() - t0, 1), 's', 'trainable', net.n_trainable(), flush=True)
    steps = 200 if DEV else TRc['steps']
    if os.environ.get('WL93_PROBE_STEPS'):                     # pre-freeze full-scale probe only (never set in the evidence chain)
        steps = int(os.environ['WL93_PROBE_STEPS'])
    opt = torch.optim.Adam(net.trainable(), lr=TRc['lr'])
    wu = TRc['warmup_steps']

    def lr(i):
        if i < wu:
            return TRc['lr'] * (i + 1) / wu
        return TRc['lr_final'] + 0.5 * (TRc['lr'] - TRc['lr_final']) * (1 + math.cos(math.pi * (i - wu) / max(steps - wu, 1)))
    rng = np.random.default_rng(seed + TRc['order_seed_offset'])
    order = []; log = []; run = []; gn = []; nonfinite = 0
    t0 = time.time()
    for i in range(steps):
        if not order:
            order = rng.permutation(len(items)).tolist()
        it = items[order.pop()]
        for g in opt.param_groups:
            g['lr'] = lr(i)
        gs = it.it.gs(); s, t, t2, q = it.tens(dev)
        G = MM.global_term(own, gs, s)
        f = net.factors(gs)
        S_, Bq, _ = net.single(f, gs, s)
        l_sum, l_s, l_g, l_q = MM.losses(S_, G, Bq, t, t2, q)
        loss = l_sum + l_s + l_g + (A['w_source_stage'] * l_q)
        opt.zero_grad(set_to_none=True); loss.backward()
        g_ = float(torch.nn.utils.clip_grad_norm_(net.trainable(), TRc['clip']))
        if not (math.isfinite(float(loss)) and math.isfinite(g_)):
            nonfinite += 1
        opt.step()
        run.append((float(loss), float(l_sum), float(l_s), float(l_g), float(l_q))); gn.append(g_)
        if (i + 1) % 250 == 0 or i + 1 == steps:
            m = np.mean(run, 0)
            rec = {'step': i + 1, 'objective': float(m[0]), 'L_sum': float(m[1]), 'L_single': float(m[2]), 'L_global': float(m[3]), 'L_source_stage': float(m[4]),
                   'grad_norm_mean': float(np.mean(gn)), 'grad_norm_max': float(np.max(gn)), 'clipped_fraction': float(np.mean(np.array(gn) > TRc['clip'])), 's': round(time.time() - t0, 1)}
            log.append(rec); run = []; gn = []
            print(json.dumps(rec), flush=True)
    # validation (component, source-stage and whole-table errors on the WL92A validation geometries)
    a = {k: 0.0 for k in ('w_n', 'w_d', 's_n', 's_d', 'g_n', 'g_d', 'q_n', 'q_d', 'qp_n', 'qp_d', 'Ls', 'Lq')}
    with torch.no_grad():
        for it in val:
            gs = it.it.gs(); s, t, t2, q = it.tens(dev)
            G = MM.global_term(own, gs, s)
            S_, Bq, _ = net.single(net.factors(gs), gs, s)
            pm = torch.as_tensor(it.it.tab['pmat'] == 1, device=dev).view(1, -1, 1, 1)
            a['w_n'] += float((S_ + G - t).abs().sum()); a['w_d'] += float(t.sum())
            a['s_n'] += float((S_ - t2).abs().sum()); a['s_d'] += float(t2.sum())
            a['g_n'] += float((G - (t - t2)).abs().sum()); a['g_d'] += float((t - t2).sum())
            a['q_n'] += float((Bq - q).abs().sum()); a['q_d'] += float(q.sum())
            a['qp_n'] += float(((Bq - q).abs() * pm).sum()); a['qp_d'] += float((q * pm).sum())
            l_sum, _, _, l_q = MM.losses(S_, G, Bq, t, t2, q); a['Ls'] += float(l_sum); a['Lq'] += float(l_q)
    n = max(len(val), 1)
    valr = {'whole_rel_l1': a['w_n'] / a['w_d'], 'single_vs_L2_rel_l1': a['s_n'] / a['s_d'], 'global_vs_L3_rel_l1': a['g_n'] / a['g_d'],
            'Bq_vs_q_rel_l1': a['q_n'] / a['q_d'], 'Bq_vs_q_panel_rel_l1': a['qp_n'] / max(a['qp_d'], 1e-30), 'L_sum': a['Ls'] / n, 'L_source_stage': a['Lq'] / n, 'n': len(val)}
    res = {'arm': arm, 'mode': A['mode'], 'seed': seed, 'steps': steps, 'trainable_params': net.n_trainable(), 'total_params': sum(p.numel() for p in net.parameters()),
           'init_sha256': init_hash, 'own_sha256': sd_hash(own), 'train_s': time.time() - t0, 'items': len(items), 'nonfinite_steps': nonfinite,
           'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'val': valr, 'log': log}
    torch.save(net.state_dict(), os.path.join(out, f'{tag}.pt'))
    json.dump(res, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print('FINAL', tag, json.dumps({k: res[k] for k in ('train_s', 'nonfinite_steps', 'trainable_params')}), json.dumps({k: round(v, 4) for k, v in valr.items() if k != 'n'}), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
