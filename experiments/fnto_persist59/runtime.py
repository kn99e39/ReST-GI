"""One passive local-graph response rule; real resident F/S coefficient cache.

No targets, reference operators, pose identifiers or incident fields enter state
construction. B is a geometry-only first-exit zero-collision operator. Because
its inherited quadrature has no segment certificates, NONRIGID S conservatively
rebuilds B in full; this shared cost is never hidden from the comparison.
"""
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'fnto_rule57'))
import r57_geometry as RG
from r57_model import Transport as HistoricalTransport


class Transport(HistoricalTransport):
    def __init__(self, arm='P'):
        super().__init__('R')
        self.arm = 'P'
        self.encoder = nn.Linear(16, 32)
        del self.global_enc

    def layers(self, g):
        z = F.silu(self.encoder(g['node'][:, :16]))
        zs = [z]
        for layer in self.messages:
            z = z + .25*layer(torch.cat([z, z[g['graph']].mean(1)], 1))
            zs.append(z)
        return zs

    def state(self, g):
        z = self.layers(g)[-1]
        return dict(z=z, s=self.src_layer(z), d=self.dst_layer(z))


def versions(model):
    return tuple(p._version for p in model.parameters())


def edge_values(model, g, st, indices):
    pieces = []
    chunk = int(__import__('os').environ.get('WL57_EDGE_CHUNK', '65536'))
    for ix in indices.split(chunk):
        h = st['s'][g['src'][ix]]+st['d'][g['dst'][ix]]+model.pair_layer(g['pair'][ix])
        pieces.append(F.softplus(model.edge(h)).view(-1, RG.MIX, 3)*g['prior'][ix, None])
    return torch.cat(pieces) if pieces else torch.empty(0, RG.MIX, 3, device=g['src'].device)


def finish(model, cache, edges=None, nodes=None):
    """Global reductions are charged, even when local raw coefficients persist."""
    g, raw, head = cache['g'], cache['raw'], cache['head']
    p, t, m = g['P'], RG.RANK, RG.MIX
    U = torch.softmax(head[:, :t*m*3].view(p, t, m, 3), dim=0)
    V = F.softplus(head[:, t*m*3:2*t*m*3].view(p, t, m, 3))
    mass = torch.sigmoid(head[:, 2*t*m*3:].view(p, m, 3))
    norm = torch.zeros(p, m, 3, device=raw.device).index_add_(0, g['src'], raw)+V.sum(1)
    scale = mass/(norm+1e-12)
    # Explicitly materialised global normalisation; count its work and bytes.
    cache['co'] = dict(W=raw*scale[g['src']], U=U, V=V*scale[:, None],
                       A=torch.softmax(model.in_logits, dim=0), O=torch.softmax(model.out_logits, dim=1))
    return cache


@torch.no_grad()
def full(model, g):
    zs = model.layers(g)
    st = dict(z=zs[-1], s=model.src_layer(zs[-1]), d=model.dst_layer(zs[-1]))
    cache = dict(g=g, zs=zs, st=st, raw=edge_values(model, g, st, torch.arange(len(g['src']), device=g['src'].device)),
                 head=model.node_head(zs[-1]), versions=versions(model), rule_identity=id(model))
    finish(model, cache)
    return cache


@torch.no_grad()
def selective(model, cache, g):
    if cache['rule_identity'] != id(model):
        raise ValueError('Different rule instance/checkpoint: explicitly rebuild resident state')
    if versions(model) != cache['versions']:
        raise ValueError('Rule weights changed: invalidate resident state explicitly')
    old = cache['g']
    for key in ('src', 'dst', 'graph', 'parent'):
        if old[key].shape != g[key].shape or not torch.equal(old[key], g[key]):
            raise ValueError('Topology/correspondence changed: unsupported selective edit')
    if old.get('cache_contract') != g.get('cache_contract'):
        raise ValueError('Physical/cache identity changed: explicitly rebuild state')
    if 'signature' in old and old['signature'] == g.get('signature'):
        return cache, dict(node_updates=[0, 0, 0], node_total=g['P'], edge_updates=0,
                           edge_total=len(g['src']), raw_retained_fraction=1.,
                           normalisation='reused', ballistic='reused material-frame intrinsic geometry',
                           masks=dict(nodes=torch.zeros(g['P'], dtype=torch.bool, device=g['src'].device),
                                      edges=torch.zeros(len(g['src']), dtype=torch.bool, device=g['src'].device)))
    changed = (old['node'][:, :16] != g['node'][:, :16]).any(1)
    layer_counts = []
    ids = changed.nonzero().flatten()
    cache['zs'][0][ids] = F.silu(model.encoder(g['node'][ids, :16]))
    layer_counts.append(int(changed.sum()))
    for k, layer in enumerate(model.messages):
        changed = changed | changed[g['graph']].any(1)
        ids = changed.nonzero().flatten()
        prev = cache['zs'][k]
        cache['zs'][k+1][ids] = prev[ids]+.25*layer(torch.cat([prev[ids], prev[g['graph'][ids]].mean(1)], 1))
        layer_counts.append(int(changed.sum()))
    st = cache['st']
    st['z'] = cache['zs'][-1]
    st['s'][ids] = model.src_layer(st['z'][ids])
    st['d'][ids] = model.dst_layer(st['z'][ids])
    edge_changed = changed[g['src']] | changed[g['dst']]
    edge_changed |= (old['pair'] != g['pair']).any(1) | (old['prior'] != g['prior']).any(1)
    ei = edge_changed.nonzero().flatten()
    cache['raw'][ei] = edge_values(model, g, st, ei)
    cache['head'][ids] = model.node_head(st['z'][ids])
    cache['g'] = g
    finish(model, cache)
    return cache, dict(node_updates=layer_counts, node_total=g['P'], edge_updates=len(ei), edge_total=len(g['src']),
                       raw_retained_fraction=1-len(ei)/len(g['src']), normalisation='global full reduction and derived-coefficient rebuild',
                       ballistic='full current-geometry rebuild on every nonrigid update',
                       masks=dict(nodes=changed, edges=edge_changed))


def apply(model, cache, x):
    return model.apply(cache['g'], x, cache['co'])
