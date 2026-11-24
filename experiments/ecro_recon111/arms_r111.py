"""WL111 arms behind one per-frame interface. Every arm receives the SAME current-frame component means (paired), reads
only current / past information and returns TOTAL (3,H,W) linear HDR on the GPU (the returned tensor may be an arm-owned
static buffer: copy it before the arm's next call if it must be kept).

  R0    native ECRO, no reconstruction (TOTAL = E1 + E2 + HL)
  R3    frozen WL105 R3 on its 8 x 8 sub-pixel guides (recomputed whenever camera or geometry changed), executed as one
        CUDA graph (common execution contract with the candidate; output equality with the eager forward is a Stage A
        check). R3e: the same model executed eagerly as in WL105 / WL108 (timing report only).
  SVGF  frozen WL101B joint-RGB SVGF (ecro_sss101b/svgf.py, its protocol settings, centre-ray guide, previous-eye
        visibility mask on camera moves, global history reset on any transport revision change), eager as in WL101B
  S     candidate, history disabled (ESS = spp), CUDA graph
  ST    candidate with the declared temporal history (same weights as S), CUDA graphs
"""
import json
import sys

import torch

import common_r111 as C
import recon_r111 as RC
import r3_r111

sys.path.insert(0, str(C.EXP / "ecro_sss101b"))
import svgf as SV  # noqa: E402

SVGF_SETTINGS = json.loads((C.EXP / "ecro_sss101b" / "protocol" / "reconstruction.json").read_text())["reconstruction"]
W = C.RES
N = W * W


def _ck(cam):
    return None if cam is None else (tuple(cam["origin"]), tuple(cam["target"]), cam["fov_deg"])


class Frame:
    """Per-frame context built by the runner: comp (9,H,W) component means; cam; spp; revision (transport state id);
    prev_cam (camera of the previous frame of this stream, None at the start); sub (candidate guide sub-sampling)."""

    def __init__(self, eng, comp, cam, spp, revision, prev_cam, geometry_changed, sub):
        self.eng, self.comp, self.cam, self.spp = eng, comp, cam, spp
        self.revision, self.prev_cam, self.geometry_changed, self.sub = revision, prev_cam, geometry_changed, sub
        self.moved = prev_cam is not None and prev_cam != cam

    def pack(self, sub, with_centre, cache):
        prev = self.prev_cam if (with_centre and self.moved) else None
        key = (sub, bool(with_centre), _ck(self.cam), _ck(prev), self.revision)
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[key] = self.eng.pack(self.cam, sub, prev, with_centre)
        return cache[key]


class R0Arm:
    name = "R0"

    def reset(self):
        pass

    def __call__(self, fr, cache):
        c = fr.comp
        return c[0:3] + c[3:6] + c[6:9]


class R3Graph:
    """Frozen R3 forward captured once as a CUDA graph on static input buffers (same modules and fused kernels)."""

    def __init__(self, model):
        self.model = model
        self.comp = torch.zeros(1, 9, W, W, device="cuda")
        self.aux = torch.zeros(1, 14, W, W, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                self.out = r3_r111.forward(model, self.comp, self.aux)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g), torch.no_grad():
            self.out = r3_r111.forward(model, self.comp, self.aux)

    @torch.no_grad()
    def __call__(self, comp, aux):
        self.comp.copy_(comp)
        self.aux.copy_(aux)
        self.g.replay()
        return self.out


class R3Arm:
    def __init__(self, model, graph=True):
        self.model = model
        self.graph = R3Graph(model) if graph else None
        self.name = "R3" if graph else "R3e"

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, fr, cache):
        g = fr.pack(8, False, cache)
        aux = r3_r111.aux14(g, fr.spp)
        if self.graph is not None:
            return self.graph(fr.comp[None], aux)[0]
        return r3_r111.forward(self.model, fr.comp[None], aux)[0]


class SVGFArm:
    name = "SVGF"

    def __init__(self, basis, device="cuda"):
        self.basis = basis
        self.den = SV.SVGF(W, device, basis, SVGF_SETTINGS)
        self.last_cam = None

    def reset(self):
        self.den = SV.SVGF(W, self.den.history.device, self.basis, SVGF_SETTINGS)
        self.last_cam = None

    @torch.no_grad()
    def __call__(self, fr, cache):
        p = fr.pack(1, True, cache)                     # centre guide (+ previous-eye visibility when the camera moved)
        guide = p["centre"]
        raw = (fr.comp[0:3] + fr.comp[3:6] + fr.comp[6:9]).reshape(3, N).contiguous()
        moved = self.last_cam is not None and self.last_cam != fr.cam
        blocked = None
        if moved:
            assert fr.prev_cam == self.last_cam
            x, y, z, _, _ = SV.projection(guide, self.last_cam, W, self.basis)
            blocked = (guide[6] >= 0) & (p["hidden"] | (z <= 0) | (x < -0.5) | (x >= W - 0.5) | (y < -0.5) | (y >= W - 0.5))
        out, count, _ = self.den.apply(raw, guide, fr.cam, fr.revision, blocked=blocked)
        self.last_cam = fr.cam
        self.count = count
        return out.reshape(3, W, W)


class CandidateArm:
    """S (history=False) or ST (history=True) around one frozen candidate network, executed by RC.GraphCandidate."""

    def __init__(self, cand, basis, history, label):
        self.cand, self.basis, self.history = cand, basis, history
        self.name = label
        self.g = RC.GraphCandidate(cand, N, history)
        self.last = {}

    def reset(self):
        self.g.reset()

    @torch.no_grad()
    def __call__(self, fr, cache):
        p = fr.pack(fr.sub, self.history, cache)
        low, high = RC.raw_low_high(fr.comp)
        out = self.g.run(low, high, fr.spp, p, fr.cam, fr.revision, self.basis)
        if self.history:
            self.last = {"age": self.g.age, "accept": self.g.map >= 0, "ess": 1.0 / self.g.rinv, "moving": self.g.last_mode == "moving",
                         "mode": self.g.last_mode}
        return out


def load_candidate(path, m, device="cuda"):
    net = RC.CompactNet(m)
    net.load_state_dict(torch.load(path, map_location=device))
    return RC.Candidate(net, C.P["candidate"]["temporal_settings"], device)
