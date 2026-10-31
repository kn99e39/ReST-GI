"""Descriptive remaining-label uncertainty: original rest r0/r1 single-label noise for all 96 train fields."""
import sys
import torch
from ext65 import ORIG,HOME,EXT,write_new
from moments65 import aggregate,norm

def main():
    torch.set_grad_enabled(False);parent=torch.load(ORIG/'data/geometry/rest.pt')['parent']
    r0=torch.load(ORIG/'data/training_targets/rest_r0.pt')['train'].double();r1=torch.load(HOME/'rest_wl57/data/targets/rest_r1.pt')['train'].double()
    rows=[]
    for coarse,label in ((True,'coarse'),(False,'fine16')):
        a,b=aggregate(r0,parent,coarse),aggregate(r1,parent,coarse);noise=norm((a-b)/2**.5)/norm((a+b)/2).clamp(min=1e-300)
        for f in range(96):
            for c in range(3):rows.append(dict(field=f,channel='RGB'[c],resolution=label,original_single_label_noise_ratio=float(noise[f,c]),replaced_in_B=f>=80))
    write_new(EXT/'label_noise.json',dict(rows=rows,note='Original rest r0 single-label noise (r0-r1)/sqrt2 relative to the r0/r1 midpoint. Deformed-geometry original labels have no independent replicate; their noise is not measured. Fields 0-79 are unchanged in both arms; bank-v2 precision for fields 80-95 is in qualification.json.'))
    print('LABEL_NOISE_DONE',flush=True)
if __name__=='__main__':main();sys.stdout.flush();__import__('os')._exit(0)
