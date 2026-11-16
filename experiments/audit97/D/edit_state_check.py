"""WL97 audit track D: correctness (NOT timing) of the relation-state cache invalidation on the timing-only edits.
Frozen code imported read-only from ~/rest_x96a/code (commit a1718f7); run with PYTHONDONTWRITEBYTECODE=1; own run root as <R>.

  python edit_state_check.py <own R> <out.json>
For CK2 and the edits used by WL96A Stage 3 (box A +0.15 m y and back) and the WL96 critic (-0.3 m y):
  selective update (ecro86.update_state, moved = ['A']) == full rebuild (build_state) == fresh scene (new Ctx)?
  dirty rays re-traced vs rays whose record actually changed; move-and-return identity.
"""
import copy, json, os, sys
import numpy as np
CODE = os.path.expanduser('~/rest_x96a/code/experiments')
sys.path.insert(0, os.path.join(CODE, 'ecro_util96a'))
import w96a as W  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(R, 'results'), exist_ok=True)
PT, S, A, FX, ctx, lt = W.setup(R)
import ecro86 as EC  # noqa: E402
MT = W.MT
g0 = copy.deepcopy(MT.SC['geom'])


def sh(dy):
    g = copy.deepcopy(g0); g['A']['c'][1] += dy
    return g


eq = lambda a, b: bool(np.array_equal(a.hit, b.hit) and np.array_equal(a.dbin, b.dbin))
out = {'N': int(ctx.N), 'K': int(S.K), 'relation_rays': int(ctx.N * S.K), 'edits': {}}
st0 = EC.build_state(ctx)
out['digest_T0'] = st0.digest()
st = st0
for name, dy in (('plus0.15y', 0.15), ('back_to_T0', 0.0), ('minus0.3y', -0.3), ('back_to_T0_b', 0.0)):
    g = sh(dy)
    ctx.set_geom(g)
    st_sel, nd = EC.update_state(ctx, st, ['A'])
    st_full = EC.build_state(ctx)
    c2 = PT.Ctx(g)
    MT.set_pmat(PT, c2, MT.variant_pmat(c2.G, 'none'))
    st_fresh = EC.build_state(c2)
    changed = int(((st_sel.hit != st.hit) | (st_sel.dbin != st.dbin)).sum())
    out['edits'][name] = {'dirty_rays_retraced': int(nd), 'dirty_fraction': nd / st.hit.shape[0], 'records_changed': changed,
                          'selective_equals_full': eq(st_sel, st_full), 'full_equals_fresh_scene': eq(st_full, st_fresh),
                          'digest': st_sel.digest()}
    del c2
    st = st_sel
out['move_and_return_identical'] = bool(out['edits']['back_to_T0']['digest'] == out['digest_T0'] and out['edits']['back_to_T0_b']['digest'] == out['digest_T0'])
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
os._exit(0)
