"""WL79 shared paths. Imports the WL68-WL78 modules unchanged; their server roots are read-only inputs:
~/rest_repr68, ~/rest_cover71, ~/rest_cmp72 (pt0 symmetry-image geometries), ~/rest_src73 (S16-R fit targets, WL73 references, te0),
~/rest_ato74 (G2 basis), ~/rest_rank75 (A75 runs), ~/rest_rank75L (WL74 Stage 4 / WL75 Stage 3 references), ~/rest_srcid78 (Q78 runs),
~/rest_arch77a (WL77A Part 0 references: scored in WL78, development data only in WL79). Evidence root: ~/rest_sym79 (R79_ROOT)."""
import os,sys,json,types
from pathlib import Path
if os.name=='nt':
    for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R79_ROOT',HOME/'rest_sym79'))
os.environ.pop('R78_ROOT',None)          # WL78 modules always resolve their own (read-only) root
p=str(HERE.parent/'fnto_srcid78')
if p not in sys.path:sys.path.insert(0,p)
import common78 as C78  # noqa: E402  (adds fnto_rank75, fnto_ato74, fnto_src73, fnto_cmp72, fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R78=C78.ROOT;R75=C78.R75;R75L=C78.R75L;R77A=C78.R77A;R74=C78.R74;R73=C78.R73;R72=C78.R72;R71=C78.R71;R68=C78.R68;INP68=C78.INP68
PROTO=HERE/'protocol/sym79_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}

def environment():
    return C68.environment(ROOT/'repo')

def dev():
    return ROOT!=HOME/'rest_sym79'

def load(p,mmap=False):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)

# ------------------------------------------------------------------ reference bundles (all archived, already-scored data)
def p71():
    out={int(k):v['port'] for k,v in json.loads((R71/'ports.json').read_text())['slots'].items()}
    out.update({int(k):v['port'] for k,v in json.loads((R72/'ports.json').read_text())['slots'].items()});return out
BUNDLE_DIRS=[('repr68_fit',INP68/'fit'),('repr68_heldout',R68/'heldout/bundles'),('cover71',R71/'refs/bundles'),('cmp72',R72/'refs/bundles'),('src73',R73/'refs/bundles'),
             ('rank75L',R75L/'refs/bundles'),('arch77a',R77A/'refs/bundles')]
def bundles():
    """Every archived reference bundle: (dir tag, geometry, slot, port, path). Ports from the bundle, else from WL71 ports.json."""
    import re,torch
    pm=p71();out=[]
    for tag,d in BUNDLE_DIRS:
        for f in sorted(d.glob('*.pt')):
            m=re.match(r'(\w+?)_s(\d+)\.pt$',f.name)
            if not m:continue
            b=torch.load(f,map_location='cpu',weights_only=False,mmap=True);s=int(m.group(2))
            port=int(b['port']) if 'port' in b else pm[s];out.append(dict(tag=tag,geom=m.group(1),slot=s,port=port,K=int(b['S'].shape[0]),path=str(f)))
    return out
