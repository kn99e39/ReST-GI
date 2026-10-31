"""WL67 focused pre-run tests (protocol baseline_sup67_v1). CPU only; run on LabServer69 before any timed stage.

python tests/test_b67.py <out json>
Env: B67_ROOT (server root), B67_W2IN (WL22 input copies with sha256 manifest).
"""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'fnto_sel56')); sys.path.insert(0, str(HERE.parent / '8dna_deformation_replication'))
HOME = Path.home(); WL56 = HOME / 'rest_wl56'; ROOT = Path(os.environ['B67_ROOT']); W2IN = Path(os.environ['B67_W2IN'])
RES = []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def check(name, ok, **kw):
    RES.append(dict(test=name, passed=bool(ok), **kw)); print(('PASS ' if ok else 'FAIL ') + name, kw if not ok else '', flush=True)


def main():
    proto = json.load(open(HERE / 'protocol/bsup67.json'))
    # 1 seed isolation (W1): ladder / bounce seeds disjoint from every WL53-56 reference / oracle seed
    w1 = {670000 + 1000 * s + 10 * li + r for s in range(3) for li in range(6) for r in range(2)} | {679001, 679999}
    used56 = set(range(5501, 5800))
    check('W1 seeds disjoint from WL53-56 seeds', not (w1 & used56), n=len(w1))
    # W2: PT chunk seed ranges disjoint from GT chunk ranges (32768 spp / chunk 32 -> 1024 seeds each)
    gt = set(range(1000000, 1001024)) | set(range(2000000, 2001024)); pt = set()
    for s in range(2):
        for li, spp in enumerate((1, 4, 16, 64, 256, 1024)):
            for r in range(2):
                b = 6700000 + 100000 * s + 1000 * li + 100 * r; pt |= set(range(b, b + max(1, spp // min(spp, 32))))
    pt |= set(range(6790000, 6790128)) | {6799999}
    check('W2 PT seed ranges disjoint from GT and from each other', not (pt & gt) and len(pt) == 2 * 2 * sum(max(1, s // min(s, 32)) for s in (1, 4, 16, 64, 256, 1024)) + 129)
    # 2 W1 source identity: fnto_sel56 / k55 / c54 / p53 equal the WL56 server copy
    diffs = []
    for d in ('fnto_sel56', 'fnto_k55', 'fnto_c54', 'phys_ref53'):
        for p in sorted((REPO / 'experiments' / d).rglob('*.py')):
            q = WL56 / 'repo' / p.relative_to(REPO)
            if not q.exists() or sha(p) != sha(q): diffs.append(str(p.relative_to(REPO)))
    check('W1 oracle/renderer sources identical to the WL56 server snapshot', not diffs, differing=diffs)
    env = json.load(open(WL56 / 'out/environment.json'))
    bad = [k for k, v in env['source_hashes'].items() if (REPO / 'experiments' / k).exists() and sha(REPO / 'experiments' / k) != v]
    check('W1 k55/c54/p53 sources match WL56 environment.json hashes', not bad, differing=bad[:5])
    # 3 W1 reference immutability record + scene signatures
    refs = sorted((WL56 / 'out/cycles').glob('cimg_*.npy')); man = {p.name: sha(p) for p in refs}
    json.dump(man, open(ROOT / 'reference_sha256.json', 'w'), indent=1)
    check('W1 reference images present (60 = 3 states x 5 cams x 4 seeds)', len(refs) == 60, n=len(refs))
    import s56_fixtures as SF
    meta = json.load(open(WL56 / 'out/runs/dent_S_r0.json'))
    sig_ok = all(meta['packages'][str(ci)]['signature'] == SF.signature('dent', ci) for ci in (0, 1))
    check('W1 dent intrinsic signatures equal WL56 packages', sig_ok)
    check('W1 rigid has no dirty domain, dent dirties only the upper', SF.dirty('rigid') == [] and SF.dirty('dent') == [1])
    # 4 W1 camera/emitter identity (fixture construction is deterministic)
    a, b = SF.scene('dent'), SF.scene('dent')
    check('W1 fixture deterministic (emitter + cameras)', all((x[3] == y[3]).all() for x, y in zip(a['comps'], b['comps'])) and a['cams'] == b['cams'])
    # 5 W2 inputs: sha manifest, lighting / states identical to the WL22 locked protocol
    man2 = json.load(open(W2IN / 'sha256.json')); bad2 = [k for k, v in man2.items() if sha(W2IN / k) != v]
    check('W2 WL22 input copies match PC sha256 manifest', not bad2, differing=bad2)
    p22 = json.load(open(REPO / 'experiments/8dna_deformation_replication/protocol/teaset_cross_backbone_locked.json'))
    gts = json.load(open(W2IN / 'gt_design/gt_states.json'))['protocol']
    check('W2 lighting/states/res equal between GT design and the WL22 locked protocol',
          gts['lighting'] == p22['lighting'] and gts['res'] == p22['res'] and all(gts['states'][s] == p22['states'][s] for s in ('T0', 'T3')))
    check('W2 frozen-dir GT equals design GT (byte identity)', all(sha(W2IN / f'frozen/{s}/gt_{x}.exr') == sha(W2IN / f'gt_design/{s}/gt_{x}.exr') for s in ('T0', 'T3') for x in 'AB'))
    ck = HOME / 'NeuralShaderDemo/external/8dna26/checkpoints/teaset.ckpt'
    check('W2 released 8DNA teaset checkpoint sha equals WL22 protocol', sha(ck) == p22['checkpoint_sha256'])
    # 6 protocol self-consistency
    check('protocol ladders as declared', proto['W1_arms']['CY']['spp_ladder'] == [1, 4, 16, 64, 256, 1024] == proto['W2_arms']['PT']['spp_ladder'])
    json.dump(dict(passed=sum(r['passed'] for r in RES), total=len(RES), results=RES), open(sys.argv[1], 'w'), indent=1)
    assert all(r['passed'] for r in RES), 'focused test failure: stop'


if __name__ == '__main__':
    main(); sys.stdout.flush(); os._exit(0)
