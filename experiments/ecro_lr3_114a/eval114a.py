"""WL114A LR3 Stage B / E evaluation with the scene's own area light (a source class never seen in training).

  python eval114a.py DATA_ROOT OUT_DIR step...     steps: oracle, arms
oracle : full-depth oracle tables from ONE deep field construction per replica (2 replicas x 256 spp, probe to K*-1):
         B_D6 = f2 + fh, B_F = f2 + fh + deep buckets <= K*; their converged ECRO readouts (2 x 512 spp, seeds 5001 / 5002).
arms   : learned tables of arms A (D6) and B (full) (source encoding SceneLights(64), generator factors + apply, as eval102),
         converged readouts (2 x 512 spp, seeds 5001 / 5002, paired with the oracle readouts), 4-spp operating-point frames
         (20 frames, seeds 4000 + i; warm median wall time), point-light table controls (test / val, train102.evaluate).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pt114a as P  # noqa: E402

XML = r'C:\Users\dna10\TransGI-Reproduction-data\scenes\living-room-3\scene_transgi.xml'
KSTAR = int(os.environ.get('WL114A_KSTAR', '96'))
DEV = os.environ.get('WL114A_DEV') == '1'
S_OR, S512, NF = (8, 16, 3) if DEV else (256, 512, 20)
mi, dr = P.mi, P.dr


def readout(ctx, L, B, out, tag):
    Bf = P.fl(np.ascontiguousarray(B, np.float32))
    W, H = ctx.scene.sensors()[0].film().crop_size()
    for h, seed in enumerate((5001, 5002)):
        p = out / f'{tag}_spp{S512}_h{h}.npz'
        if p.exists():
            continue
        acc, T = ctx.render(L, S512, seed, 'ecro', B=Bf, b=4)
        np.savez_compressed(p, **P.pack(acc, W, H, S512))


def main(data, out, steps):
    data, out = Path(data), Path(out); out.mkdir(parents=True, exist_ok=True)
    ctx = P.Ctx(P.load_scene(XML)); L = P.SceneLights(64)
    meta = json.loads((out / 'eval_meta.json').read_text()) if (out / 'eval_meta.json').exists() else {}
    if 'oracle' in steps:
        tabs = []
        for r, seed in enumerate((114601, 114602)):
            p = out / f'oracle_field_r{r}.npz'
            if not p.exists():
                t0 = time.perf_counter()
                fd, cnt = ctx.field_deep(L, S_OR, seed, KSTAR - 1, b=16)
                np.savez_compressed(p, cnt=cnt, **fd); meta[f'oracle_field_r{r}_s'] = time.perf_counter() - t0
            z = np.load(p)
            d6 = z['f2'] + z['fh']; F = d6 + sum((z[n] for n in P.deep_through(KSTAR) if n in z.files), np.zeros_like(d6))
            tabs.append((d6, F))
        BD6 = 0.5 * (tabs[0][0] + tabs[1][0]); BF = 0.5 * (tabs[0][1] + tabs[1][1])
        np.savez_compressed(out / 'oracle_tables.npz', B_D6=BD6, B_F=BF)
        t0 = time.perf_counter(); readout(ctx, L, BD6, out, 'oracle_D6'); readout(ctx, L, BF, out, 'oracle_F'); meta['oracle_readout_s'] = time.perf_counter() - t0
        print('ORACLE done', flush=True)
    if 'arms' in steps:
        import torch
        sys.path.insert(0, str(HERE.parent / 'ecro86')); sys.path.insert(0, str(HERE.parent / 'ecro_lr3_102'))
        os.environ.pop('WL102_AREA_FLOOR', None)
        import model86 as M
        import train102 as TRN
        import train114a as T4
        dev = torch.device('cuda')
        gs, rows = TRN.geostate(data)
        s = ctx.source(L)
        for arm, depth in (('A', 'D6'), ('B', 'F')):
            mdir = out.parent / f'model_{arm}'
            model = M.ECRO().to(dev); model.mat_states = rows
            model.load_state_dict(torch.load(mdir / 'model.pt', map_location=dev)); model.eval()
            T_ = {}
            for rep in range(3):
                dr.sync_thread(); torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad():
                    f = model.factors(TRN.input_state(gs)); torch.cuda.synchronize(); T_['factors_s'] = time.perf_counter() - t0
                    t1 = time.perf_counter(); B = model.apply(f, gs, torch.as_tensor(s, device=dev)[None])[0]; torch.cuda.synchronize(); T_['apply_s'] = time.perf_counter() - t1
            Bn = B.detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(out / f'table_{arm}.npz', B=Bn, s=s)
            readout(ctx, L, Bn, out, f'arm_{arm}')
            Bf = P.fl(Bn); W, H = ctx.scene.sensors()[0].film().crop_size(); frames = []
            for i in range(NF):
                acc, Tm = ctx.render(L, 4, 4000 + i, 'ecro', B=Bf, b=4); frames.append(Tm['wall_s'])
                if i == 0:
                    np.savez_compressed(out / f'arm_{arm}_spp4.npz', **P.pack(acc, W, H, 4))
            T_['frame_4spp_wall_s_median_warm'] = float(np.median(frames[2:] if len(frames) > 2 else frames))
            T4.DEPTH = depth; TRN.Item = T4.Item
            T_['controls_test'] = TRN.evaluate(model, gs, [TRN.Item(f) for f in sorted((data / 'test').glob('g*.npz'))])
            T_['validation'] = TRN.evaluate(model, gs, [TRN.Item(f) for f in sorted((data / 'val').glob('g*.npz'))])
            T_['table_bytes'] = int(Bn.nbytes)
            meta[f'arm_{arm}'] = T_
            print('ARM', arm, json.dumps({k: v for k, v in T_.items() if not isinstance(v, dict)}), flush=True)
            del model; torch.cuda.empty_cache()
    json.dump(meta, open(out / 'eval_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ['oracle', 'arms'])
    sys.stdout.flush(); os._exit(0)
