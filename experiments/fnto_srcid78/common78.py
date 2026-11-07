"""WL78 shared paths. Imports the WL68-WL75 modules unchanged (read-only roots on the server: ~/rest_repr68, ~/rest_cover71, ~/rest_cmp72,
~/rest_src73, ~/rest_ato74, ~/rest_rank75 (A75 runs, hash-verified), ~/rest_rank75L (WL75 Stage 3 references, mirrored from the local WL75 run,
hash-verified), ~/rest_arch77a (WL77A Part 0 references, never read before the WL78 protocol freeze)). Evidence root: ~/rest_srcid78."""
import os,sys,json,types
from pathlib import Path
if os.name=='nt':
    for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R78_ROOT',HOME/'rest_srcid78'))
p=str(HERE.parent/'fnto_rank75')
if p not in sys.path:sys.path.insert(0,p)
import common75 as C75  # noqa: E402  (adds fnto_ato74, fnto_src73, fnto_cmp72, fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R75=C75.ROOT;R74=C75.R74;R73=C75.R73;R72=C75.R72;R71=C75.R71;R68=C75.R68;INP68=C75.INP68
R75L=Path(os.environ.get('R75L_ROOT',HOME/'rest_rank75L'))
R77A=Path(os.environ.get('R77A_ROOT',HOME/'rest_arch77a'))
PROTO=HERE/'protocol/srcid78_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}

def environment():
    return C68.environment(ROOT/'repo')

def dev():
    return ROOT!=HOME/'rest_srcid78'

def load(p,mmap=False):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False,mmap=mmap)
