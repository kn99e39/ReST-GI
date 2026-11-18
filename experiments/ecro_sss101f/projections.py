"""Independent physical estimation and held-out camera readout for WL101F."""
from types import SimpleNamespace

import numpy as np

from query_readout import LUMA


class Reduction:
    """Preserve each 32-spp launch reduction before double-precision accumulation."""
    def __init__(self, reader, width):
        self.R,self.dr,self.mi=reader.R,reader.dr,reader.mi
        self.width=width
        self.sum={}

    def add(self, columns, indices):
        dr,mi=self.dr,self.mi
        dr.eval(columns,indices)
        partial={}
        for name,value in columns.items():
            buf=dr.zeros(self.R.F,self.width)
            dr.scatter_reduce(dr.ReduceOp.Add,buf,value,indices)
            partial[name]=buf
        dr.eval(partial)
        for name,value in partial.items():
            if name not in self.sum:
                self.sum[name]=mi.Float64(value)
            else:
                self.sum[name]+=mi.Float64(value)
        dr.eval(self.sum)

    def arrays(self):
        return {name:np.asarray(value).copy() for name,value in self.sum.items()}


def weight_y(q):
    return sum(LUMA[c]*q.weight[c] for c in range(3))


def stack(values, label, shape):
    return np.stack([values[f"{label}{c}"] for c in range(3)],-1).reshape(*shape,3)


def train(reader, proxy, masks, config, seed, suffix_seed, monitor):
    """Use independent physical suffix samples on the fixed near/flank query measure."""
    R,dr=reader.R,reader.dr
    near,flank=masks["near"].ravel(),masks["flank"].ravel()
    raster=np.flatnonzero(near|flank)
    assert not np.any(near&flank)
    pixels=R.U(raster.astype(np.uint32))
    # Multiplying both region coefficients by a shared constant cancels in every table.
    balance=R.F(np.select([near,flank],[1.,near.sum()/flank.sum()],0.).astype(np.float32))
    spp=config["training_spp"]
    batch=config["training_launch_spp"]
    nested=Reduction(reader,reader.slots*16)
    max_coord,queries=0.,0
    for i in range(spp//batch):
        p,pix,rng,sd=reader.rays(batch,seed,i,pixels=pixels)
        _,_,q=reader.prefix(p,pix,rng,sd)
        slot,s,t,err=reader.addresses(q,config["gates"]["address_coordinate_tolerance"])
        max_coord=max(max_coord,err)
        queries+=dr.width(q.pix)
        suffix=reader.suffix(q,sd,proxy,suffix_seed*1000003+i)
        qweight=dr.gather(R.F,balance,q.pix)
        columns={"count":dr.full(R.F,1.,dr.width(q.pix)),"mass":qweight}
        for c in range(3):
            light=suffix[f"f2{c}"]+suffix[f"fh{c}"]
            w=qweight*q.weight[c]
            columns[f"L{c}"]=qweight*light
            columns[f"W{c}"]=w
            columns[f"WL{c}"]=w*light
        nested.add(columns,slot*16+s*4+t)
        if (i+1)%16==0:
            monitor(f"training {seed} {batch*(i+1)}/{spp}")
    result=nested.arrays()
    shape=(reader.slots,4,4)
    for k in ("count","mass"):
        result[k]=result[k].reshape(shape)
    result["count"]=result["count"].astype(np.uint32)
    for k in ("L","W","WL"):
        result[k]=stack(result,k,shape)
        for c in range(3): del result[f"{k}{c}"]
        assert np.isfinite(result[k]).all() and result[k].min()>=0
    assert int(result["count"].sum())==queries
    assert np.all((result["count"]==0)==(result["mass"]==0))
    return result,{"prefix_seed":seed,"suffix_seed":suffix_seed,"spp_per_pixel":spp,
                   "candidate_rays":int(spp*len(raster)),"positive_queries":queries,
                   "coordinate_excursion":max_coord}


def projected(numerator, denominator, fallback):
    assert numerator.shape==fallback.shape
    valid=np.broadcast_to(denominator>0,numerator.shape)
    assert np.all(numerator[~valid]==0)
    table=np.divide(numerator,denominator,out=fallback.copy(),where=valid).astype(np.float32)
    return table, valid


def make_tables(raw, uniform, names):
    uniform=uniform.reshape(-1,3)
    weighted, weight=raw["WL"],raw["W"]
    pt,ps=projected(weighted.sum((1,2)),weight.sum((1,2)),uniform)
    tables={"P":(pt,ps)}
    if "Q" in names:
        tables["Q"]=projected(raw["L"].sum((1,2)),raw["mass"].sum((1,2))[:,None],uniform)
    for name, axis in (("S",2),("A",1),("SA",None)):
        if name not in names:
            continue
        n,d=(weighted,weight) if axis is None else (weighted.sum(axis),weight.sum(axis))
        parent=np.broadcast_to(pt.reshape(len(pt),*(1,)*(n.ndim-2),3),n.shape)
        tables[name]=projected(n,d,parent)
    return {name:tables[name] for name in names}


def verify_projection():
    # Unequal throughputs, missing cells and parent collapse on a known field.
    field=np.arange(2*4*4*3,dtype=np.float64).reshape(2,4,4,3)/7
    weights=1+np.square(field)
    weights[1,0,:]=0
    truth={"W":weights,"WL":weights*field,"mass":(weights[...,0]>0).astype(np.float64),
           "L":np.where(weights>0,field,0)}
    out=make_tables(truth,np.ones((2,3),np.float32),("P","Q","S","A","SA"))
    p,sp=out["P"]
    assert sp.all() and np.allclose(np.sum(weights,axis=(1,2))*p,np.sum(truth["WL"],axis=(1,2)))
    np.testing.assert_array_equal(out["S"][0][1,0],p[1])
    assert not out["S"][1][1,0].any()
    assert np.allclose(out["SA"][0][out["SA"][1]],field[out["SA"][1]])
    return True


def uniform_geometry(reader, oracle, config, monitor):
    """Read geometric distributions in WL101E's area–solid-angle cells."""
    dr,R=reader.dr,reader.R
    batch=8
    spp=config["uniform_audit_spp"]
    histogram=Reduction(reader,reader.slots*16)
    totals={k:0 for k in ("valid","buried","back","tie","invalid")}
    max_coord=0.
    for i in range(spp//batch):
        slots,rays,hits,_,_,info=oracle.probes(batch,config["uniform_audit_seed"],i)
        for key in totals:
            totals[key]+=int(dr.count(info[key])[0])
        ids=dr.compress(info["valid"])
        requested=dr.gather(R.U,slots,ids)
        q=SimpleNamespace(point=dr.gather(type(hits.p),hits.p,ids),
                          ray=dr.gather(type(rays),rays,ids),pid=requested//64,bn=requested%64)
        slot,s,t,err=reader.addresses(q,config["gates"]["address_coordinate_tolerance"])
        max_coord=max(max_coord,err)
        histogram.add({"count":dr.full(R.F,1.,dr.width(ids))},slot*16+s*4+t)
        monitor()
    assert totals["invalid"]==0 and sum(totals.values())==reader.slots*spp
    assert totals["tie"]<=config["gates"]["uniform_audit_tie_fraction"]*(totals["tie"]+totals["valid"])
    hist=histogram.arrays()["count"].reshape(reader.slots,4,4).astype(np.uint32)
    assert int(hist.sum())==totals["valid"]
    return hist,{"counts":totals,"coordinate_excursion":max_coord,"proposals_per_slot":spp,
                 "seed":config["uniform_audit_seed"]}


def render(reader, proxy, config, seed, tables, monitor, query_control=False, learned=None):
    """Trace each frozen camera stream once per phase, independently of field sampling."""
    R,dr,mi=reader.R,reader.dr,reader.mi
    projected={}
    for name,(table,native) in tables.items():
        projected[name]=(R.PT.fl(table),tuple(mi.Bool(native.reshape(-1,3)[:,j].copy()) for j in range(3)))
    if learned is not None:
        learned=R.PT.fl(learned)
    collect=Reduction(reader,reader.n)
    total_queries=0
    coord_error=0.
    spp=config["camera_spp"]
    batch=config["camera_launch_spp"]
    for i in range(spp//batch):
        ray,pix,rng,sd=reader.rays(batch,seed,i)
        prefix,_,q=reader.prefix(ray,pix,rng,sd,learned)
        if query_control:
            collect.add({**{k:v for k,v in prefix.items() if k[:2] in ("e1","e2")},
                         **{f"learned{c}":prefix[f"hl{c}"] for c in range(3)}},pix)
        slot,s,t,err=reader.addresses(q,config["gates"]["address_coordinate_tolerance"])
        coord_error=max(coord_error,err)
        total_queries+=dr.width(q.pix)
        queries={"total_weight":weight_y(q)}
        if query_control:
            field=reader.suffix(q,sd,proxy)
            for c in range(3):
                queries[f"query{c}"]=q.weight[c]*(field[f"f2{c}"]+field[f"fh{c}"])
        collect.add(queries,q.pix)
        keys={"P":slot,"Q":slot,"U":slot,"S":slot*4+s,"A":slot*4+t,"SA":slot*16+s*4+t}
        for name,(table,native) in projected.items():
            index=keys[name.split("_")[0]]
            value=reader.gather(table,index,q)
            weight=sum(LUMA[c]*dr.select(dr.gather(mi.Bool,native[c],index),q.weight[c],0.)
                       for c in range(3))
            collect.add({**{f"{name}:{c}":v for c,v in enumerate(value)},f"{name}:weight":weight},q.pix)
        if (i+1)%16==0:
            monitor(f"camera {seed} {batch*(i+1)}/{spp}")
    arrays=collect.arrays()
    shape=(reader.res,reader.res)
    output={"total_weight":(arrays["total_weight"]/spp).reshape(shape).astype(np.float32)}
    for name in tables:
        output[name]=(stack(arrays,name+":",shape)/spp).astype(np.float32)
        output[name+":weight"]=(arrays[name+":weight"]/spp).reshape(shape).astype(np.float32)
    if query_control:
        for key in ("e1","e2","learned","query"):
            output[key]=(stack(arrays,key,shape)/spp).astype(np.float32)
    return output,{"seed":seed,"spp":spp,"positive_queries":total_queries,
                   "coordinate_excursion":coord_error}
