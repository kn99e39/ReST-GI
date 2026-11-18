"""WL101H candidate transport semantics: contact-valid native origins (isolated path).

The historical native semantics (pt100.Ctx.trace / direct, Mitsuba spawn_ray / spawn_ray_to and the
SSS exit continuation ep + 1e-4 ew) are never modified. `install(R, rt, h_min)` binds rewritten copies
of `trace` and `direct` to ONE runtime instance. Three source lines change:

  NEE shadow ray       si.spawn_ray_to(light)   -> contact-valid origin, singular -> no NEE
  BSDF continuation    si.spawn_ray(dir)        -> contact-valid origin, singular -> path ends
  SSS exit             ep + ew * 1e-4           -> contact-valid step,   singular -> path ends

Contact-valid origin. Native spawn offsets the hit point p by m along the unit side normal u. Register
entry 1 (sphere/floor tangent contact) gives the analytic free distance g from p along u to the other
solid: from a non-sphere point, the sphere entry distance; from a sphere point, the floor-plane distance.
If m < g/2 the native ray is kept unchanged. Otherwise the origin moves to p + u g/2, the midpoint of the
free gap, keeping the native direction (spawn_ray) or re-aiming at the same target with the same relative
shadow epsilon (spawn_ray_to). If g/2 < h_min no representable free-space origin exists: the event is
singular and contributes nothing. No other register entry has a finite-area native-origin degeneracy.
"""
import inspect
import textwrap
import types

INF = 1e30


class Contact:
    def __init__(self, R, rt, h_min):
        self.R, self.rt, self.mi, self.dr = R, rt, R.mi, R.dr
        self.c = R.mi.Point3f(*[float(v) for v in rt.SC])
        self.r = float(rt.SR)
        self.h_min = float(h_min)

    def gap(self, p, u, on_sphere):
        dr, F = self.dr, self.R.F
        q = p - self.c
        b = dr.dot(q, u)
        cc = dr.squared_norm(q) - self.r * self.r
        disc = b * b - cc
        # Stable entry distance for an outside point approaching the sphere: cc / (-b + sqrt(disc)).
        approach = (cc > 0) & (b < 0) & (disc > 0)
        t_s = dr.select(approach, cc / (-b + dr.sqrt(dr.maximum(disc, 0.))), F(INF))
        t_f = dr.select(u.z < 0, p.z / dr.select(u.z < 0, -u.z, F(1.)), F(INF))
        return dr.select(on_sphere, t_f, t_s)

    def _fix(self, si, ray, target):
        dr, mi, F = self.dr, self.mi, self.R.F
        off = ray.o - si.p
        m = dr.norm(off)
        u = off / dr.maximum(m, 1e-30)
        on_sphere = dr.eq(self.rt.shape_id(si), 6)
        g = self.gap(si.p, u, on_sphere)
        need = m >= 0.5 * g
        singular = need & (0.5 * g < self.h_min)
        o = mi.Point3f(dr.select(need, si.p + u * (0.5 * g), ray.o))
        if target is None:
            d, maxt = ray.d, ray.maxt
        else:
            v = target - o
            lv = dr.norm(v)
            d = mi.Vector3f(dr.select(need, v / dr.maximum(lv, 1e-30), ray.d))
            maxt = dr.select(need, lv * (ray.maxt / dr.maximum(dr.norm(target - ray.o), 1e-30)), ray.maxt)
        out = mi.Ray3f(o, d)
        out.maxt = maxt
        return out, singular, need

    def spawn(self, si, d):
        return self._fix(si, si.spawn_ray(d), None)

    def spawn_to(self, si, target):
        return self._fix(si, si.spawn_ray_to(target), target)

    def exit(self, ep, ew):
        """SSS exit continuation origin: native ep + 1e-4 ew unless the floor plane is closer."""
        dr, mi, F = self.dr, self.mi, self.R.F
        g = dr.select(ew.z < 0, ep.z / dr.select(ew.z < 0, -ew.z, F(1.)), F(INF))
        need = 1e-4 >= 0.5 * g
        singular = need & (0.5 * g < self.h_min)
        step = dr.select(need, 0.5 * g, F(1e-4))
        return mi.Point3f(ep + ew * step), singular, need


REPLACEMENTS = {
    "direct": [
        ("            sh=self.scene.ray_intersect(si.spawn_ray_to(lights.pos[l]),on)\n",
         "            _hr,_hs,_hn=self._h101.spawn_to(si,lights.pos[l]);on=on&~_hs\n"
         "            sh=self.scene.ray_intersect(_hr,on)\n"),
    ],
    "trace": [
        ("        newray=si.spawn_ray(si.to_world(wo))\n",
         "        newray,_hs,_hn=self._h101.spawn(si,si.to_world(wo))\n"),
        ("            newray.o=dr.select(is_sss,ep+ew*1e-4,newray.o)\n",
         "            _eo,_es,_en=self._h101.exit(ep,ew)\n"
         "            newray.o=dr.select(is_sss,_eo,newray.o)\n"),
        ("        thr*=dr.select(valid,weight,0.);active &= valid;ray=newray\n",
         "        valid=valid&~(_hs&~is_sss)\n"
         "        if self.case=='sss':valid=valid&~(_es&is_sss)\n"
         "        thr*=dr.select(valid,weight,0.);active &= valid;ray=newray\n"),
    ],
}


def rewritten_source(R, name):
    src = textwrap.dedent(inspect.getsource(getattr(R.pt100.Ctx, name)))
    for old, new in REPLACEMENTS[name]:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    return src


def compile_method(R, name, src, extra=None):
    scope = {}
    exec(compile(src, f"wl101h_candidate_{name}", "exec"), getattr(R.pt100.Ctx, name).__globals__, scope)
    return scope[name]


def install(R, rt, h_min):
    """Bind the candidate semantics to one runtime instance. Returns the Contact object."""
    assert rt.case == "sss"
    rt._h101 = Contact(R, rt, h_min)
    for name in ("direct", "trace"):
        setattr(rt, name, types.MethodType(compile_method(R, name, rewritten_source(R, name)), rt))
    return rt._h101


def uninstall(rt):
    for name in ("direct", "trace", "_h101"):
        if name in rt.__dict__:
            delattr(rt, name)


def candidate_spawn_to(R, rt):
    return lambda si, target: rt._h101.spawn_to(si, target)[0]


def candidate_spawn(R, rt):
    return lambda si, d: rt._h101.spawn(si, d)[0]


def observed_trace(R, h_min_unused=None):
    """Candidate trace source with the WL101F event-2 observation line (camera readout instrument)."""
    src = rewritten_source(R, "trace")
    before = "        if mode=='ecro' and k==2:\n"
    assert src.count(before) == 1
    src = src.replace(before, before + "            self.observe_high(si,ray,active,thr,pid,bn,rng)\n")
    return compile_method(R, "trace", src)
