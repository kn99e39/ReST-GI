"""Verify extracted public files against downloaded release archives, read-only."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2])
def digest(b): return hashlib.sha256(b).hexdigest()
payload={'upstream':str(root),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_diff':subprocess.check_output(['git','diff','--stat','HEAD'],cwd=root,text=True),'archives':{},'hardware':subprocess.check_output(['nvidia-smi'],text=True)}
for name in ['scenes.zip','weights.zip']:
    archive=root/name; result={'sha256':digest(archive.read_bytes()),'bytes':archive.stat().st_size,'checked':0,'missing':[],'mismatch':[],'first_entries':[]}
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir(): continue
            rel=Path(member.filename)
            if len(result['first_entries'])<4: result['first_entries'].append(member.filename)
            candidates=[root/rel,root/'scenes'/rel,root/'checkpoints'/rel]
            path=next((p for p in candidates if p.is_file()),None)
            if path is None: result['missing'].append(member.filename); continue
            if digest(path.read_bytes())!=digest(z.read(member)): result['mismatch'].append(member.filename)
            result['checked']+=1
    payload['archives'][name]=result
out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
