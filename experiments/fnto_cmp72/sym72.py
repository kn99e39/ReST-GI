"""Exact domain symmetries of the slab family (Candidate A): in-plane D4 for every pose (dent centre transformed), plus z-flip for the
undented rest slab. Port permutations are taken from rest material coordinates and must be exact bijections."""
import numpy as np
import torch

D4={'id':((1,0),(0,1)),'r90':((0,-1),(1,0)),'r180':((-1,0),(0,-1)),'r270':((0,1),(-1,0)),
    'mx':((-1,0),(0,1)),'my':((1,0),(0,-1)),'sw':((0,1),(1,0)),'asw':((0,-1),(-1,0))}

def ops(rest):
    """rest: D4 x {no flip, z-flip} (16); dented poses: D4 (8)."""
    out=[(k,False) for k in D4]
    return out+[(k,True) for k in D4] if rest else out

def matrix(op):
    k,flip=op;m=np.eye(3);m[:2,:2]=np.array(D4[k],dtype=float)
    if flip:m[2,2]=-1
    return m

def permutation(rest_xyz,op,tol=1e-9):
    """perm[p] = port at op(rest_xyz[p]); exact bijection or error."""
    from scipy.spatial import cKDTree
    x=np.asarray(rest_xyz,dtype=np.float64);y=x@matrix(op).T;d,i=cKDTree(x).query(y)
    assert d.max()<=tol and len(set(i.tolist()))==len(x),('symmetry not exact',op,float(d.max()))
    return i

def pose_image(pose,op):
    """(depth, radius, cx, cy) -> pose with centre op(c) (z-flip only allowed for depth 0)."""
    dep,rad,cx,cy=pose;assert not (op[1] and dep),'z-flip maps a dented slab outside the family'
    c=np.array(D4[op[0]],dtype=float)@np.array([cx,cy]);return (dep,rad,float(np.round(c[0],12)),float(np.round(c[1],12)))

def integrated(S):
    """Angle-integrated collided response F_i(j,c) = sum_a sum_o S[i,a,j,o,c]; S (n,16,P,64,3) -> (n,P,3)."""
    return S.sum((1,3))

def model_integrated(T,A,d):
    """sum_a d[a] sum_r A[r,a,c] T[j,r,c] (sum_o O = 1); T (n,P,M,C), d (n,16) -> (n,P,C)."""
    return torch.einsum('na,rac,njrc->njc',d,A,T)
