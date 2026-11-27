"""WL111A predictor variants (Stage B H3 control). The WL111 CompactNet is unchanged; a variant only replaces modules.

  fullres  CompactNet with the full-resolution 1 x 1 fusion layer replaced by two 3 x 3 convolutions
           (c1 + c0 -> 2 c0 -> 2 c0, ReLU between; the forward's ReLU follows), so the predictor has three full-resolution
           3 x 3 layers (stem + 2) instead of one. Head, trunk, inputs, outputs and the spatial operator are unchanged.
"""
import torch.nn as nn

import common_r111a  # noqa: F401  (sys.path to the WL111 code)
import recon_r111 as RC


class FullResNet(RC.CompactNet):
    def __init__(self, m=1.0):
        super().__init__(m)
        c0, c1, _, _ = RC.widths(m)
        self.fuse = nn.Sequential(RC.conv(c1 + c0, 2 * c0, k=3), nn.ReLU(), RC.conv(2 * c0, 2 * c0, k=3))


NETS = {"ctak": RC.CompactNet, "fullres": FullResNet}


def load_candidate(path, net="ctak", m=1.0, device="cuda"):
    import torch
    import common_r111 as C
    n = NETS[net](m)
    n.load_state_dict(torch.load(path, map_location=device))
    return RC.Candidate(n, C.P["candidate"]["temporal_settings"], device)
