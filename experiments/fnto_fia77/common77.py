"""WL77 shared paths (WL76 FIA with an azimuth-resolving outgoing response). Reuses the WL68/71/72/73/74/75 modules unchanged; their roots are read-only inputs:
~/rest_repr68, ~/rest_cover71, ~/rest_cmp72, ~/rest_src73 (S16-R targets, WL73 references, AR controls), ~/rest_ato74 (G2 basis),
~/rest_rank75 (A75 controls), ~/rest_fia76 (WL76 FIA runs, Stage 1 record, the unused WL76 Stage 4 population). Evidence root: ~/rest_fia77 (R77_ROOT)."""
import os,sys,json,types
from pathlib import Path
if os.name=='nt':
    for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R77_ROOT',HOME/'rest_fia77'))
os.environ.setdefault('R75_ROOT',str(HOME/'rest_rank75'))
p=str(HERE.parent/'fnto_rank75')
if p not in sys.path:sys.path.insert(0,p)
import common75 as C75  # noqa: E402  (adds fnto_ato74, fnto_src73, fnto_cmp72, fnto_cover71, fnto_repr68)
import common68 as C68  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R75=C75.ROOT;R74=C75.R74;R73=C75.R73;R72=C75.R72;R71=C75.R71;R68=C75.R68;INP68=C75.INP68
PROTO=HERE/'protocol/fia77_v1.json'
R76=Path(os.environ.get('R76_ROOT_RO',HOME/'rest_fia76'))   # WL76 evidence root (read-only)
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}
TRAIN=('rest','pt0')

def environment():
    return C68.environment(ROOT/'repo')

def dev():
    return ROOT!=HOME/'rest_fia77'

def seed(kind,g,slot,a,rep):
    """Seeds: the WL76 quadrature seeds 762xxxxxx are reused unchanged (identical W / SS); confirmation 766/767xxxxxx are the seeds WL76 predeclared for its never-generated Stage 4 population; WL77 synthetic tests 771xxxxxx."""
    base={'phys':761000000,'quad':762000000,'conf':766000000,'synth':771000000}[kind]
    return base+g*1000000+slot*10000+a*100+rep
