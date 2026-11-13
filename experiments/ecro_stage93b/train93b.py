"""WL93B smoke training (WL93A training contract unchanged; global branch = frozen WL92A own_s<seed>).

  python train93b.py <R> <R86> <R92A> <U0|UQ|C0|CQ|R_T0|R_TQ> <seed> [--dev]
R_T0 / R_TQ: reproduction controls of the reused WL93A T0 / TQ checkpoints (same mode and weight).
Data: the 128-geometry train subset (R/data/train_q exists only for it), q = one realization at the protocol train budget;
validation monitoring with independent training-validation targets (R/data/val_q).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_stage93'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93b.json')))


def main():
    R, R86, R92A, arm, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    import model93 as MM
    import train93 as TR
    dev = torch.device('cuda')
    A = P['arms'][arm[2:] if arm.startswith('R_') else arm]
    TRc = json.load(open(os.path.join(HERE, '..', 'ecro_stage93a', 'protocol', 'ecro_stage93a.json')))['task_b']['training']   # WL93A contract, unchanged
    out = os.path.join(R, 'results', 'models'); os.makedirs(out, exist_ok=True)
    tag = f'{arm}_s{seed}'
    torch.manual_seed(seed); np.random.seed(seed)
    own = TR.load_own(R92A, seed, dev)
    net = MM.Local(own, A['mode'], cm_seed=seed + 9300).to(dev)                 # WL93 tiedcm seed convention (unused for tied / untied)
    init_hash = TR.sd_hash(net)
    t0 = time.time()
    items = TR.population(R, R86, R92A, 'train')
    val = TR.population(R, R86, R92A, 'val')
    print('loaded', len(items), len(val), round(time.time() - t0, 1), 's', 'trainable', net.n_trainable(), flush=True)
    steps = 100 if DEV else TRc['steps']
    opt = torch.optim.Adam(net.trainable(), lr=TRc['lr'])
    wu = TRc['warmup_steps']

    def lr(i):
        if i < wu:
            return TRc['lr'] * (i + 1) / wu
        return TRc['lr_final'] + 0.5 * (TRc['lr'] - TRc['lr_final']) * (1 + math.cos(math.pi * (i - wu) / max(steps - wu, 1)))

    def validate():
        a = {k: 0.0 for k in ('w_n', 'w_d', 's_n', 's_d', 'q_n', 'q_d', 'qp_n', 'qp_d', 'Ls', 'Lq')}
        with torch.no_grad():
            for it in val:
                gs = it.it.gs(); s, t, t2, q = it.tens(dev)
                G = MM.global_term(own, gs, s)
                S_, Bq, _ = net.single(net.factors(gs), gs, s)
                pm = torch.as_tensor(it.it.tab['pmat'] == 1, device=dev).view(1, -1, 1, 1)
                a['w_n'] += float((S_ + G - t).abs().sum()); a['w_d'] += float(t.sum())
                a['s_n'] += float((S_ - t2).abs().sum()); a['s_d'] += float(t2.sum())
                a['q_n'] += float((Bq - q).abs().sum()); a['q_d'] += float(q.sum())
                a['qp_n'] += float(((Bq - q).abs() * pm).sum()); a['qp_d'] += float((q * pm).sum())
                l_sum, _, _, l_q = MM.losses(S_, G, Bq, t, t2, q); a['Ls'] += float(l_sum); a['Lq'] += float(l_q)
        n = max(len(val), 1)
        return {'whole_rel_l1': a['w_n'] / a['w_d'], 'single_vs_L2_rel_l1': a['s_n'] / a['s_d'], 'Bq_vs_trainval_q_rel_l1': a['q_n'] / a['q_d'],
                'Bq_vs_trainval_q_panel_rel_l1': a['qp_n'] / max(a['qp_d'], 1e-30), 'L_sum': a['Ls'] / n, 'L_source_stage': a['Lq'] / n}
    val0 = validate()
    rng = np.random.default_rng(seed + TRc['order_seed_offset'])
    order = []; log = []; run = []; gn = []; nonfinite = 0; seen = []
    t0 = time.time()
    for i in range(steps):
        if not order:
            order = rng.permutation(len(items)).tolist()
        k = order.pop(); it = items[k]; seen.append(k)
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
        if (i + 1) % 100 == 0 or i + 1 == steps:
            m = np.mean(run, 0)
            rec = {'step': i + 1, 'objective': float(m[0]), 'L_sum': float(m[1]), 'L_single': float(m[2]), 'L_global': float(m[3]), 'L_source_stage': float(m[4]),
                   'grad_norm_mean': float(np.mean(gn)), 'grad_norm_max': float(np.max(gn)), 'clipped_fraction': float(np.mean(np.array(gn) > TRc['clip'])), 's': round(time.time() - t0, 1)}
            if (i + 1) % 500 == 0 or i + 1 == steps:
                rec['val'] = validate()
            log.append(rec); run = []; gn = []
            print(json.dumps(rec), flush=True)
    train_s = time.time() - t0
    import hashlib
    res = {'arm': arm, 'mode': A['mode'], 'seed': seed, 'steps': steps, 'trainable_params': net.n_trainable(), 'init_sha256': init_hash, 'final_sha256': TR.sd_hash(net),
           'own_sha256': TR.sd_hash(own), 'train_s': train_s, 'items': len(items), 'order_sha256': hashlib.sha256(np.array(seen, np.int64).tobytes()).hexdigest(),
           'nonfinite_steps': nonfinite, 'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'val_init': val0, 'val': log[-1]['val'], 'log': log}
    torch.save(net.state_dict(), os.path.join(out, f'{tag}.pt'))
    json.dump(res, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print('FINAL', tag, json.dumps({k: res[k] for k in ('train_s', 'nonfinite_steps', 'trainable_params')}), json.dumps({k: round(v, 4) for k, v in res['val'].items()}), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
