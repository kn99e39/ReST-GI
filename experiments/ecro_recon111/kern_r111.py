"""WL111 fused Triton kernels (inference): the a-trous softmax-kernel pass and the temporal history step. Reference
implementations of the same statements are in recon_r111 (atrous_ref / temporal_ref). Imported lazily by recon_r111."""
import triton
import triton.language as tl


@triton.jit
def _atrous(img, logits, out, H, W, HW, D: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(1)
    pix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pix < HW
    y = pix // W
    x = pix % W
    lb = logits + c * 25 * HW
    m = tl.full((BLOCK,), -1e30, tl.float32)
    for k in range(25):
        m = tl.maximum(m, tl.load(lb + k * HW + pix, mask=mask, other=-1e30).to(tl.float32))
    s = tl.zeros((BLOCK,), tl.float32)
    a0 = tl.zeros((BLOCK,), tl.float32)
    a1 = tl.zeros((BLOCK,), tl.float32)
    a2 = tl.zeros((BLOCK,), tl.float32)
    ib = img + c * 3 * HW
    for k in range(25):
        yy = y + (k // 5 - 2) * D
        xx = x + (k % 5 - 2) * D
        yy = tl.where(yy < 0, -yy, yy)
        yy = tl.where(yy >= H, 2 * H - 2 - yy, yy)
        xx = tl.where(xx < 0, -xx, xx)
        xx = tl.where(xx >= W, 2 * W - 2 - xx, xx)
        q = yy * W + xx
        e = tl.exp(tl.load(lb + k * HW + pix, mask=mask, other=-1e30).to(tl.float32) - m)
        s += e
        a0 += e * tl.load(ib + q, mask=mask, other=0.0)
        a1 += e * tl.load(ib + HW + q, mask=mask, other=0.0)
        a2 += e * tl.load(ib + 2 * HW + q, mask=mask, other=0.0)
    ob = out + c * 3 * HW
    tl.store(ob + pix, a0 / s, mask=mask)
    tl.store(ob + HW + pix, a1 / s, mask=mask)
    tl.store(ob + 2 * HW + pix, a2 / s, mask=mask)

@triton.jit
def _temporal(Lo, Hi, Cen, Pure, PCen, PPure, Hid, Acc, Cnt, Rinv, Age,
              NAcc, NCnt, NRinv, NAge, Map, Par, N, W,
              VALID: tl.constexpr, MOVING: tl.constexpr, BZ: tl.constexpr):
    # Par (float32 device tensor, read on the device so the step can be replayed inside a CUDA graph):
    # 0-2 previous eye, 3-5 previous right, 6-8 previous up, 9-11 previous forward, 12 previous tan(fov / 2),
    # 13-15 current eye, 16 spp, 17 sample cap, 18 normal cos, 19 footprint pixels, 20 SSS view cos
    spp = tl.load(Par + 16)
    smax = tl.load(Par + 17)
    i = tl.program_id(0) * BZ + tl.arange(0, BZ)
    m = i < N
    j = i
    ok = m & (i < 0)          # all false
    if VALID:
        ok = m
        if MOVING:
            cox = tl.load(Par + 0)
            coy = tl.load(Par + 1)
            coz = tl.load(Par + 2)
            rx = tl.load(Par + 3)
            ry = tl.load(Par + 4)
            rz = tl.load(Par + 5)
            ux = tl.load(Par + 6)
            uy = tl.load(Par + 7)
            uz = tl.load(Par + 8)
            fx = tl.load(Par + 9)
            fy = tl.load(Par + 10)
            fz = tl.load(Par + 11)
            tanf = tl.load(Par + 12)
            ox = tl.load(Par + 13)
            oy = tl.load(Par + 14)
            oz = tl.load(Par + 15)
            nrm_min = tl.load(Par + 18)
            foot_px = tl.load(Par + 19)
            sss_cos = tl.load(Par + 20)
            px = tl.load(Cen + i, m, 0.0)
            py = tl.load(Cen + N + i, m, 0.0)
            pz = tl.load(Cen + 2 * N + i, m, 0.0)
            nx = tl.load(Cen + 3 * N + i, m, 0.0)
            ny = tl.load(Cen + 4 * N + i, m, 0.0)
            nz = tl.load(Cen + 5 * N + i, m, 0.0)
            sid = tl.load(Cen + 6 * N + i, m, -1.0)
            dx = px - cox
            dy = py - coy
            dz = pz - coz
            xr = rx * dx + ry * dy + rz * dz
            yu = ux * dx + uy * dy + uz * dz
            zf = fx * dx + fy * dy + fz * dz
            den = tl.maximum(zf, 1e-6) * tanf
            xx = (xr / den + 1.0) * W / 2 - 0.5
            yy = (1.0 - yu / den) * W / 2 - 0.5
            fxx = xx + 0.5
            fyy = yy + 0.5
            ix = fxx.to(tl.int32)
            iy = fyy.to(tl.int32)
            ix = tl.where(fxx < ix.to(tl.float32), ix - 1, ix)      # floor (truncation toward zero corrected)
            iy = tl.where(fyy < iy.to(tl.float32), iy - 1, iy)
            inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < W) & (zf > 0)
            j = tl.minimum(tl.maximum(iy, 0), W - 1) * W + tl.minimum(tl.maximum(ix, 0), W - 1)
            qx = tl.load(PCen + j, m, 0.0)
            qy = tl.load(PCen + N + j, m, 0.0)
            qz = tl.load(PCen + 2 * N + j, m, 0.0)
            qnx = tl.load(PCen + 3 * N + j, m, 0.0)
            qny = tl.load(PCen + 4 * N + j, m, 0.0)
            qnz = tl.load(PCen + 5 * N + j, m, 0.0)
            qsid = tl.load(PCen + 6 * N + j, m, -2.0)
            dist = tl.sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy) + (pz - qz) * (pz - qz))
            foot = foot_px * zf * (2.0 * tanf / W)
            ndot = nx * qnx + ny * qny + nz * qnz
            ax = cox - px
            ay = coy - py
            az = coz - pz
            bx = ox - px
            by = oy - py
            bz = oz - pz
            la = tl.maximum(tl.sqrt(ax * ax + ay * ay + az * az), 1e-12)
            lb = tl.maximum(tl.sqrt(bx * bx + by * by + bz * bz), 1e-12)
            vcos = (ax * bx + ay * by + az * bz) / (la * lb)
            view_ok = (sid != 6.0) | (vcos >= sss_cos)
            pure = tl.load(Pure + i, m, 0) != 0
            ppure = tl.load(PPure + j, m, 0) != 0
            visible = tl.load(Hid + i, m, 1) == 0
            ok = (m & inside & (sid >= 0.0) & (sid == qsid) & (ndot >= nrm_min) & (dist <= foot) & view_ok
                  & pure & ppure & visible)
    n_prev = tl.load(Cnt + j, ok, 0.0)
    r_prev = tl.load(Rinv + j, ok, 0.0)
    a_prev = tl.load(Age + j, ok, 0.0)
    n_new = tl.where(ok, tl.minimum(n_prev + spp, smax), spp)
    w = spp / n_new
    r_new = tl.where(ok, (1.0 - w) * (1.0 - w) * r_prev + w * w / spp, 1.0 / spp)
    for c in tl.static_range(0, 3):
        xl = tl.load(Lo + c * N + i, m, 0.0)
        xh = tl.load(Hi + c * N + i, m, 0.0)
        hl = tl.load(Acc + c * N + j, ok, 0.0)
        hh = tl.load(Acc + (3 + c) * N + j, ok, 0.0)
        tl.store(NAcc + c * N + i, tl.where(ok, hl + w * (xl - hl), xl), m)
        tl.store(NAcc + (3 + c) * N + i, tl.where(ok, hh + w * (xh - hh), xh), m)
    tl.store(NCnt + i, n_new, m)
    tl.store(NRinv + i, r_new, m)
    tl.store(NAge + i, tl.where(ok, a_prev + 1.0, 1.0), m)
    tl.store(Map + i, tl.where(ok, j, -1), m)
