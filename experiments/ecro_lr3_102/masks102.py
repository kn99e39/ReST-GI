"""WL102 region masks: object id of the first hit at every pixel centre of the evaluation camera (ECRO env, Mitsuba 3.5.1).

  python masks102.py <scene.xml> <out.npz>
Groups (protocol metrics.regions): rug, cushions, paint (WhitePaint_*: shelves / niches / frames), sofa, floor, walls, plus
material classes diffuse vs glossy (roughplastic / roughconductor surfaces).
"""
import json, sys
import numpy as np
import pt102 as P

mi, dr = P.mi, P.dr
ctx = P.Ctx(P.load_scene(sys.argv[1]))
ray, pix, (W, H) = ctx.gen_rays(1, None)
si = ctx.scene.ray_intersect(ray)
sid = np.where(np.asarray(si.is_valid()), np.asarray(ctx.shape_id(si)).astype(np.int64), -1).reshape(H, W)
names = [m[1] for m in ctx.meshes]
glossy_obj = np.array(['RoughPlastic' in str(m[0].bsdf()) or 'RoughConductor' in str(m[0].bsdf()) for m in ctx.meshes])
groups = {'rug': ['Rug'], 'cushions': ['Cushions_0001', 'Cushions_0002', 'Cushions_0003'],
          'paint': [n for n in names if n.startswith('WhitePaint')], 'sofa': ['Sofa_0001', 'Sofa_0002', 'Sofa_0003'],
          'floor': ['Floor'], 'walls': ['Walls']}
masks = {}
for g, members in groups.items():
    ids = [names.index(m) for m in members if m in names]
    masks[g] = np.isin(sid, ids)
masks['glossy'] = (sid >= 0) & glossy_obj[np.clip(sid, 0, None)]
masks['diffuse'] = (sid >= 0) & ~glossy_obj[np.clip(sid, 0, None)]
np.savez_compressed(sys.argv[2], sid=sid, names=json.dumps(names), **{f'mask_{k}': v for k, v in masks.items()})
print({k: int(v.sum()) for k, v in masks.items()})
