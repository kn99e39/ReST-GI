"""Validate and export the completed fixed-final-model 4096-spp tile render."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image

base = Path.home() / 'rest_8dna_ck21b'
stem = '8dna_ck2_epoch30_4096spp'
output = base / 'results' / stem
source = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else base / 'results' / 'epoch30_4096spp_tiles'
assert source.parent == (base / 'results').resolve()
key = 'ck2_neural_r0_512'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
output.mkdir(exist_ok=True)
if (output / 'matched4096_complete.json').exists():
    print('Final comparison already complete')
    raise SystemExit(0)
record = json.loads((source / (key + '.json')).read_text())
assert record['status'] == 'complete' and record['completed_spp'] == 4096
checkpoint = output / 'checkpoint.ckpt'
assert sha(checkpoint) == 'd107c982ce9546774acf2c0d575c0f151c482a50af1f6f6a1b92e15c7fafd077'
assert record['checkpoint_sha256'] == sha(checkpoint)
hdr = np.load(source / (key + '_s4096.npy'))
assert hdr.shape == (512,512,3) and np.isfinite(hdr).all()
shutil.copyfile(source / (key + '_s4096.npy'), output / (stem + '.npy'))
shutil.copyfile(source / (key + '_s4096.exr'), output / (stem + '.exr'))
positive = np.maximum(hdr, 0)
display = np.clip(positive / (1 + positive), 0, 1) ** (1/2.2)
Image.fromarray(np.rint(display*255).astype(np.uint8)).save(output / (stem + '.png'))
complete = {'status':'complete','epochs_completed':30,'checkpoint_global_step':245760,
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
            'spp_completed':4096,'resolution':[512,512],
            'source_render_directory':str(source), 'source_render_metadata_sha256':sha(source/(key+'.json')),
            'execution':record['execution'],'worker_admission':record.get('worker_admission'),
            'display':'exposure1, Reinhard x/(1+x), gamma2.2',
            'files':{stem+suffix:sha(output/(stem+suffix)) for suffix in ['.png','.npy','.exr']}}
temporary = output / 'complete.tmp'
temporary.write_text(json.dumps(complete,indent=2)+'\n')
temporary.replace(output / 'complete.json')
subprocess.run([sys.executable,str(base/'scripts/figure_highspp_matched.py'),'30'],check=True)
