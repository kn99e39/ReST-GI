"""Hold this job at its next completed epoch; never discard optimizer updates."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
base=Path.home()/'rest_8dna_ck21b'
pid=int(sys.argv[1]); link=base/'training'/'CK2_bx38'/'last.ckpt'
initial=link.resolve()
expected=b'python\x00-u\x00scripts/train_native.py\x00'
while True:
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    if not cmd.endswith(expected): raise RuntimeError('Job PID identity changed')
    try: current=link.resolve(strict=True)
    except FileNotFoundError: time.sleep(.25); continue
    if current!=initial:
        os.kill(pid,signal.SIGSTOP)
        out=base/'resume_anchor'; out.mkdir(exist_ok=True)
        import shutil
        anchor=out/current.name; shutil.copy2(current,anchor)
        record={'status':'paused_at_epoch_boundary','pid':pid,'checkpoint':str(anchor),'source_checkpoint':str(current),'sha256':hashlib.sha256(anchor.read_bytes()).hexdigest(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'reason':'User-authorized execution-pipeline profiling; next epoch reload cannot have performed an optimizer update before this 250ms observer.'}
        (base/'pause_boundary.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True); break
    time.sleep(.25)
