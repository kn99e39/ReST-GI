"""Diagnostic (not evidence): which way of passing a small per-call parameter array into a Dr.Jit frozen function is
replayed with fresh values on Mitsuba 3.8.0 / Dr.Jit 1.3.1."""
import numpy as np
import engine_r111 as E
mi, dr, torch, R = E.mi, E.dr, E.torch, E.R
F, U = mi.Float, mi.UInt32
n = 8


def body(vals):
    x = dr.arange(F, n)
    return x * 0 + vals[0] + vals[1]


def A(ca):
    return body([dr.gather(F, ca, U(k)) for k in range(2)])


def B(ca):
    return body([dr.gather(F, ca, dr.opaque(U, k)) for k in range(2)])


def Cf(a, b):
    return body([a, b])


fa, fb, fc, fd = (R.maybe_freeze(A), R.maybe_freeze(B), R.maybe_freeze(Cf), R.maybe_freeze(B))
buf = torch.zeros(16, device="cuda")
for i in range(5):
    v = [float(i), 10.0 * i]
    ca = F(np.array(v + [0.0] * 14, np.float32)); dr.make_opaque(ca)
    ra = fa(ca); rb = fb(ca); rc = fc(dr.opaque(F, v[0]), dr.opaque(F, v[1]))
    buf.copy_(torch.tensor(v + [0.0] * 14))
    rd = fd(F(buf.__dlpack__()))
    dr.eval(ra, rb, rc, rd)
    print(i, "expected", v[0] + v[1], "A", ra[0], "B", rb[0], "C", rc[0], "D", rd[0], flush=True)
