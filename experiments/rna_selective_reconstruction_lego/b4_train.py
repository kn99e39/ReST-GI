"""Architecture Batch 4 (worklog 37): train owned-unit RNA on a T0/T1 state with the UNCHANGED worklog-32 trainer
(rso_train.py), swapping only the isolated edit adapter (b4_model): the owner over the state's combined mesh, canonical
positions, and the 17 unit specs (L16 worklog-36 specs + B).

    B4_STATE=T0|T1 python b4_train.py --config <yml> --partition <partition_L16.npz> --out <dir> [rso_train args]

--init <state.pt> --train-units <D*...> is the worklog-36 REINIT mechanism (re-initialise the listed units, freeze all
others with requires_grad False, train only the listed ones): arm B_selective.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b4_model as E  # noqa: E402
import rso_model as M  # noqa: E402
import rso_partition  # noqa: E402
import rso_train  # noqa: E402

B_MESH = E.ROOT / "results/rna_selective_reconstruction_lego/scene/B_box_T0.npz"


def main() -> int:
    import numpy as np

    state = os.environ["B4_STATE"]
    owner = E.make_owner(state)
    canon = {}
    _unit_specs = M.unit_specs

    def specs(partition, base, aabb, matched=True):
        canon["aabb"] = aabb  # the training set's AABB = the canonical AABB (T0 and T1 sets are checked equal before training)
        M.unit_specs = _unit_specs
        try:
            return E.edit_unit_specs(partition, base, aabb, B_MESH, matched=matched)
        finally:
            M.unit_specs = specs

    M.unit_specs = specs
    rso_partition.Owner = lambda partition: owner
    M.OwnedSurfaceDataModule = lambda conf, owner=None, **kw: E.EditOwnedDataModule(conf, owner=owner, canonical_aabb=canon["aabb"], **kw)
    return rso_train.main()


if __name__ == "__main__":
    sys.exit(main())
