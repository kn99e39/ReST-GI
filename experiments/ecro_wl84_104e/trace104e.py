"""Stage-local continuation and terminal evaluation with the frozen WL104 camera tracer."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104d"))
from common104d import H

C, RT, mi, dr, R81 = H.C, H.RT, H.mi, H.dr, H.R81
PT = RT.P
TERMINALS = {"A": ("A", "C"), "C": ("A", "C", "L"), "L": ("C", "L")}


def luma(v):
    return v.x * H.LUMA[0] + v.y * H.LUMA[1] + v.z * H.LUMA[2]


def angle(cosine):
    return dr.acos(dr.clamp(cosine, -1.0, 1.0)) * (180.0 / np.pi)


def mirror_dot(wi, wo):
    return -wi.x * wo.x - wi.y * wo.y + wi.z * wo.z


class Counterfactual(RT.Ctx):
    def __init__(self, tag, desc, a, c):
        H.RR.REG[f"wl104e_{tag}_C"] = ("ER", c)
        H.RR.REG[f"wl104e_{tag}_A"] = ("ER", a)
        R81.REG[f"wl104e_{tag}_L"] = desc
        base = {"type": "cmp83", "key": f"wl104e_{tag}_C"}
        super().__init__(mi.load_dict(C.scene_dict(mi, RT.T, base, with_sensor=True)))
        self.alt = {
            "A": mi.load_dict({"type": "cmp83", "key": f"wl104e_{tag}_A"}),
            "L": mi.load_dict({"type": "lay104b", "key": f"wl104e_{tag}_L", "max_depth": 1024}),
        }
        self.names = [m[1] for m in self.meshes]
        self.sphere_index, self.cube_index = (self.names.index(x) for x in ("sphere", "cube"))

    def uses_er(self, sid):
        return dr.eq(sid, self.sphere_index) | dr.eq(sid, self.cube_index)

    def bsdf(self, si, sid, variant):
        native = si.bsdf()
        return native if variant == "C" else dr.select(self.uses_er(sid), mi.BSDFPtr(self.alt[variant]), native)

    def terminal(self, si, sid, light, active, variants):
        if light.n != 1:
            raise ValueError("The frozen experiment requires one point light")
        dl = light.pos[0] - si.p
        d2 = dr.squared_norm(dl)
        wl = dl * dr.rsqrt(d2)
        wo = si.to_local(wl)
        on = active & (wo.z > 1e-6) & (dr.dot(-wl, light.axis[0]) > light.cc[0])
        vis = on & ~self.scene.ray_test(si.spawn_ray_to(light.pos[0]), on)
        wo_safe = dr.select(vis, wo, mi.Vector3f(0, 0, 1))
        values = {}
        for variant in variants:
            f = self.bsdf(si, sid, variant).eval(self.bctx, si, wo_safe, vis)
            values[variant] = dr.select(vis, f * light.rgb[0] * dr.rcp(d2), 0.0)
        return values, wo_safe, vis

    def trace_stages(self, ray, rng, sampler, light, diagnostic=False):
        active, thr = mi.Bool(True), mi.Color3f(1.0)
        tail = sampler != "C" and not diagnostic
        depth = PT.D if tail else 2
        outputs, diagnostic_lanes = {}, None
        for hit in range(1, depth + 1):
            si, sid, _, ok = self.surface(ray, active)
            active &= ok & (si.wi.z > 1e-6)
            if hit == 2:
                variants = ("A", "C") if diagnostic else TERMINALS[sampler]
                fields, wo, vis = self.terminal(si, sid, light, active, variants)
                outputs.update({sampler + k: thr * x for k, x in fields.items()})
                if diagnostic:
                    first_sid, first_wi, first_mirror, first_pdf, weight_resid, f_bound, first_valid = first
                    diagnostic_lanes = {
                        "first_sid": first_sid, "first_theta": first_wi, "first_mirror": first_mirror,
                        "first_pdf": first_pdf, "sample_weight_error": weight_resid, "sample_eval_peak": f_bound,
                        "first_valid": first_valid, "second_sid": sid, "second_theta": angle(si.wi.z),
                        "second_mirror": angle(mirror_dot(si.wi, wo)),
                        "second_visible": vis, "throughput": luma(thr),
                        "e2_a": luma(outputs[sampler + "A"]), "e2_c": luma(outputs[sampler + "C"]),
                    }
            elif hit > 2:
                contribution = thr * self.nee(si, None, light, rng, active)
                outputs["tail"] = outputs["tail"] + contribution if "tail" in outputs else contribution
            if hit == depth:
                break
            b = self.bsdf(si, sid, sampler) if hit == 1 else si.bsdf()
            bs, w = b.sample(self.bctx, si, rng.next_float32(),
                             mi.Point2f(rng.next_float32(), rng.next_float32()), active)
            valid = active & (bs.pdf > 0) & (dr.max(w) > 0)
            if diagnostic and hit == 1:
                denom = b.eval(self.bctx, si, dr.select(valid, bs.wo, mi.Vector3f(0, 0, 1)), valid)
                residual = dr.select(valid, dr.max(dr.abs(w * bs.pdf - denom)), 0.0)
                first = (sid, angle(si.wi.z), angle(mirror_dot(si.wi, bs.wo)), bs.pdf,
                         residual, dr.select(valid, dr.max(denom), 0.0), mi.Bool(valid))
            thr = thr * dr.select(valid, w, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(bs.wo))
        return outputs, diagnostic_lanes

    def packet(self, seed, launch, spp, sampler, light, diagnostic=False):
        R81.PASS_SEED["v"] = mi.UInt64(seed * 100003 + launch + 1)
        dr.make_opaque(R81.PASS_SEED["v"])
        sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
        n = C.RES * C.RES * spp
        rng = mi.PCG32(size=n, initstate=sd, initseq=mi.UInt64(0))
        ray, pix, _ = self.gen_rays(spp, rng)
        out, lanes = self.trace_stages(ray, rng, sampler, light, diagnostic)
        if diagnostic:
            dr.eval(*lanes.values())
            dr.sync_thread()
            return {key: np.asarray(v).copy() for key, v in lanes.items()}
        # The old accumulator keys use triplets with a common string prefix.
        fields = {f"{key}{c}": v[c] for key, v in out.items() for c in range(3)}
        return self.accum(fields, pix, C.RES * C.RES)

    def evaluate(self, seed, sampler, light, spp, batch):
        acc = {}
        for j, start in enumerate(range(0, spp, batch)):
            packet = self.packet(seed, j, min(batch, spp - start), sampler, light)
            for key, value in packet.items():
                acc[key] = acc[key] + value if key in acc else value
        prefixes = tuple(dict.fromkeys(key[:-1] for key in acc))
        return {
            key: PT.pack({f"e1{c}": acc[f"{key}{c}"] for c in range(3)}, C.RES, C.RES, spp)["e1"]
            for key in prefixes
        }
