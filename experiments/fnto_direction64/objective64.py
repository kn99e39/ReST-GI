"""Only a training objective; runtime Transport remains the frozen WL59 class."""
from r57_data import aggregate

def contrast(y,t,parent,eta):
    assert y.shape[0]==t.shape[0]==2
    values=[]
    for i,coarse in enumerate((True,False)):
        a,b=aggregate(y,parent,coarse),aggregate(t,parent,coarse)
        d=a[0]-a[1];q=b[0]-b[1];mean=.5*(b[0]+b[1])
        norm=lambda z:z.flatten(0,1).square().sum(0)
        values.append((norm(d-q)/(norm(q)+eta[i]*norm(mean)+1e-20)).mean())
    return .5*(values[0]+values[1])

def total(base,directional,weight):
    # The control must preserve the original objective and gradient graph exactly.
    return base if weight==0 else base+weight*directional
