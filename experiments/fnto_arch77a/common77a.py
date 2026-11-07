"""WL77A shared paths (multi-agent architecture discovery). Read-only inputs: the WL68-WL76 roots (via common76 -> common75 -> ...).
Evidence root: ~/rest_arch77a (R77A_ROOT). WL77 code and roots are never touched."""
import os,sys,json,types
from pathlib import Path
if os.name=='nt':
    for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))
HOME=Path.home();HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('R77A_ROOT',HOME/'rest_arch77a'))
p=str(HERE.parent/'fnto_fia76')
if p not in sys.path:sys.path.insert(0,p)
import common76 as C76  # noqa: E402
from common68 import sha,tensor_sha,write_new,save_new  # noqa: E402,F401
R76=Path(os.environ.get('R76_ROOT_RO',HOME/'rest_fia76'));R75=C76.R75;R74=C76.R74;R73=C76.R73;R72=C76.R72;R71=C76.R71;R68=C76.R68;INP68=C76.INP68
PROTO=HERE/'protocol/arch77a_v1.json'
CFG=json.loads(PROTO.read_text()) if PROTO.exists() else {}
def environment():return C76.C68.environment(ROOT/'repo')
def dev():return ROOT!=HOME/'rest_arch77a'
def seed(g,slot,a,rep):
    """Confirmation seeds 766/767xxxxxx: the seeds WL76 predeclared for this never-generated population (rest 0, te1 1)."""
    return 766000000+g*1000000+slot*10000+a*100+rep
