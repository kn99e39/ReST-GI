"""WL81 chain decisions (frozen rules of protocol compiled81_v1 + amendment 1). Prints a token and exits 0; the chain reads it.

  python gate81.py a0 <A0_dir>             -> QUALIFIED | REFERENCE_NOT_QUALIFIED
  python gate81.py data <data_dir>         -> QUALIFIED | REFERENCE_NOT_QUALIFIED
  python gate81.py b <B_dir>               -> CONTINUE | STOP
  python gate81.py c <C_eval> <C_contract> -> space-separated qualified arms, primary first | NONE
  python gate81.py d <D_eval> <D_contract> <primary> -> LEARNED_PHYSICAL_PASS | PHYSICAL_ONLY | FAIL
"""
import json, os, sys


def load(*p):
    return json.load(open(os.path.join(*p)))


def a0(d):
    it = load(d, 'iface.json')['_worst']
    ok1 = it['eval_max_rel'] <= 1e-4 and it['eval_nonzero_mismatch'] == 0 and it['sample_valid_mismatch'] == 0 and \
        it['sample_weight_max_rel'] <= 1e-4 and it['sample_pdf_max_rel'] <= 1e-4 and it['pdf_max_rel'] <= 1e-3
    w = load(d, 'walk.json')
    ok2 = all(v['albedo']['max_abs_z'] <= 4 and v['reciprocity']['max_abs_z'] <= 4 for k, v in w.items() if k != 'cost')
    s = load(d, 'slab.json')
    ok3 = all(max(abs(z) for z in r['z']) <= 4 for r in s)
    ok4 = load(d, 'trunc.json')['PASS']
    return 'QUALIFIED' if (ok1 and ok2 and ok3 and ok4) else 'REFERENCE_NOT_QUALIFIED'


def data(d):
    ok = all(load(d, f'noise_{s}.json')['PASS'] for s in ['gate', 'test_iid', 'test_block', 'train_eval'])
    ok &= all(load(d, f'hl_noise_{s}.json')['PASS'] for s in ['gate', 'test_iid'])
    return 'QUALIFIED' if ok else 'REFERENCE_NOT_QUALIFIED'


def b(d):
    return 'CONTINUE' if load(d, 'stageB.json')['continue'] else 'STOP'


def c(ev, ct):
    e = load(ev, 'stageC_eval.json')['representation_gate']
    q = [a for a in ['candidate', 'control'] if e[a]['PASS'] and load(ct, a, 'contract.json')['verdict_wl81']['PHYSICAL_CONTRACT_PASS']]
    return ' '.join(q) if q else 'NONE'


def dd(ev, ct, primary):
    lr = load(ev, 'stageD_eval.json')['learned_rule_gate'][primary]['PASS']
    ph = load(ct, primary, 'contract.json')['verdict_wl81']['PHYSICAL_CONTRACT_PASS']
    return 'LEARNED_PHYSICAL_PASS' if (lr and ph) else ('PHYSICAL_ONLY' if ph else 'FAIL')


if __name__ == '__main__':
    m = sys.argv[1]
    print({'a0': lambda: a0(sys.argv[2]), 'data': lambda: data(sys.argv[2]), 'b': lambda: b(sys.argv[2]),
           'c': lambda: c(sys.argv[2], sys.argv[3]), 'd': lambda: dd(sys.argv[2], sys.argv[3], sys.argv[4])}[m]())
