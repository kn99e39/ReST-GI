"""Post-hoc physical visual control only; no training or model evaluation.

Replace the SSS sphere by opaque Lambert with the same RGB parameter, solely
to show nonlocal source response on the shadowed camera-facing hemisphere.
The parameter match is NOT an emergent-reflectance or energy calibration.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
import common100 as C
from pt100 import Ctx,PT


def main(root):
    root=Path(root);cfg=json.loads((root/'configs.json').read_text())['sss']['test'][0]
    ctx=Ctx(cfg['geom'],'sss');ctx.case='near'  # turn off volume/ballistic dispatch
    ctx.mats.states[2]=C.lambert([.92,.80,.65])
    cam=dict(cfg['camera'],res=512)
    acc,timing=ctx.render(PT.Lights(cfg['lights'][0]),'ER',cam,512,4096,100905,'cam',b=32,kmax=1)
    path=root/'sss/opaque_control.npz'
    np.savez_compressed(path,**PT.pack(acc,512,4096))
    C.jd({'posthoc':True,'config':cfg,'seed':100905,'spp':4096,'timing':timing,
          'control':'opaque Lambert [.92,.80,.65]; same RGB parameter, not matched emergent reflectance',
          'scope':'physical-only direct-source visual control, no model/training changes','script_sha256':C.sha(__file__)},path.with_suffix('.json'))
    print('PHYSICAL_CONTROL_COMPLETE',timing,flush=True)


if __name__=='__main__':main(sys.argv[1]);sys.stdout.flush();os._exit(0)
