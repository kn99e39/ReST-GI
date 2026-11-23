"""WL110 screen recorder: samples the X server's own framebuffer (Xvfb -fbdir, XWD format, memory-mapped) — i.e. what the
display shows, including the Tk overlay — independently of the viewer process.

  python record110.py <Xvfb_screen0> <out_dir> --region x,y,w,h --seconds 12 [--probe-hz 120 --store-hz 30]

* probe loop (--probe-hz): hashes a subsample of the image region every tick and counts distinct screen states -> an
  independent screen-side estimate of presented frames per second (valid while probe-hz >= 2 x presented FPS);
* store loop (every probe-hz / store-hz ticks): keeps the full window region for the recording;
* outputs: recording.webp (animated, lossy, --quality, stored rate), recording_preview.gif (half size, 15 fps), screenshot.png (a
  stored frame from the middle), capture.json (timestamps, distinct-state counts, overhead of the recorder loop itself).
The recorder is a separate process; its CPU cost and any effect on the viewer are measured by comparing runs with / without it.
"""
import argparse, hashlib, json, mmap, struct, time
from pathlib import Path

import numpy as np


def xwd_view(path):
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    h = struct.unpack('>25I', mm[:100])
    header_size, version, fmt, depth, width, height = h[0], h[1], h[2], h[3], h[4], h[5]
    byte_order, bpp, bpl, ncolors = h[7], h[11], h[12], h[19]
    assert version == 7 and bpp == 32, (version, bpp)
    off = header_size + ncolors * 12
    arr = np.frombuffer(mm, dtype=np.uint8, count=bpl * height, offset=off).reshape(height, bpl // 4, 4)
    return f, mm, arr, {'width': width, 'height': height, 'depth': depth, 'byte_order': byte_order, 'bytes_per_line': bpl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fb')
    ap.add_argument('out')
    ap.add_argument('--region', required=True, help='x,y,w,h of the window on the screen')
    ap.add_argument('--image-region', default=None, help='x,y,w,h of the rendered image inside the screen (distinct-state probe)')
    ap.add_argument('--seconds', type=float, default=12.0)
    ap.add_argument('--probe-hz', type=float, default=120.0)
    ap.add_argument('--store-hz', type=float, default=30.0)
    ap.add_argument('--delay', type=float, default=0.0, help='wait before recording (viewer warm-up)')
    ap.add_argument('--quality', type=int, default=80, help='WebP quality of recording.webp')
    ap.add_argument('--wait-change', type=float, default=0.0,
                    help='WL110 execution fix: first wait (up to this many s) until the image region has changed 10 times, '
                         'i.e. the viewer is presenting, then apply --delay (a fixed delay missed the slower-starting SVGF viewer)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    x, y, w, h = map(int, a.region.split(','))
    ix, iy, iw, ih = map(int, (a.image_region or a.region).split(','))
    f, mm, fb, info = xwd_view(a.fb)
    every = max(1, int(round(a.probe_hz / a.store_hz)))
    nstore = int(a.seconds * a.store_hz) + 2
    store = np.empty((nstore, h, w, 3), np.uint8)
    st_t = []
    waited = None
    if a.wait_change > 0:
        tw = time.perf_counter(); last, nch = None, 0
        while nch < 10 and time.perf_counter() - tw < a.wait_change:
            hsh = hashlib.blake2b(np.ascontiguousarray(fb[iy:iy + ih:3, ix:ix + iw:3, :3]).tobytes(), digest_size=8).digest()
            nch += int(last is not None and hsh != last); last = hsh
            time.sleep(0.02)
        waited = {'seconds': time.perf_counter() - tw, 'changes_seen': nch}
    time.sleep(a.delay)
    probe_t, probe_h = [], []
    busy = 0.0
    t0 = time.perf_counter()
    tick = 0
    while True:
        due = t0 + tick / a.probe_hz
        now = time.perf_counter()
        if now - t0 > a.seconds:
            break
        if now < due:
            time.sleep(max(0.0, due - now - 2e-4))
            continue
        tb = time.perf_counter()
        sub = fb[iy:iy + ih:3, ix:ix + iw:3, :3]
        probe_h.append(hashlib.blake2b(np.ascontiguousarray(sub).tobytes(), digest_size=8).digest())
        probe_t.append(tb - t0)
        if tick % every == 0 and len(st_t) < nstore:
            store[len(st_t)] = fb[y:y + h, x:x + w, 2::-1]       # BGRX -> RGB
            st_t.append(tb - t0)
        busy += time.perf_counter() - tb
        tick += 1
    wall = time.perf_counter() - t0
    # distinct screen states (consecutive probes that differ)
    changes = [probe_t[i] for i in range(1, len(probe_h)) if probe_h[i] != probe_h[i - 1]]
    gaps = np.diff(probe_t) * 1e3 if len(probe_t) > 1 else np.array([])
    ns = len(st_t)
    frames = store[:ns]
    from PIL import Image
    Image.fromarray(frames[ns // 2]).save(out / 'screenshot.png')
    ims = [Image.fromarray(fr) for fr in frames]
    ims[0].save(out / 'recording.webp', save_all=True, append_images=ims[1:], duration=int(round(1000 / a.store_hz)), loop=0, quality=a.quality, method=4)
    half = [Image.fromarray(fr[::2, ::2]) for fr in frames[::2]]
    half[0].save(out / 'recording_preview.gif', save_all=True, append_images=half[1:], duration=int(round(2000 / a.store_hz)), loop=0, optimize=True)
    cap = {'framebuffer': info, 'region': [x, y, w, h], 'image_region': [ix, iy, iw, ih], 'seconds': wall,
           'probe_hz_target': a.probe_hz, 'probes': len(probe_t), 'probe_interval_ms': {'median': float(np.median(gaps)) if gaps.size else None,
                                                                                         'p95': float(np.percentile(gaps, 95)) if gaps.size else None,
                                                                                         'max': float(gaps.max()) if gaps.size else None},
           'distinct_screen_changes': len(changes),
           'screen_change_rate_hz': (len(changes) - 1) / (changes[-1] - changes[0]) if len(changes) > 1 else None,
           'stored_frames': ns, 'store_hz_target': a.store_hz,
           'recorder_busy_fraction': busy / wall,
           'wait_change': waited, 'delay_s': a.delay,
           'note': 'screen_change_rate_hz counts distinct image-region states seen in the X framebuffer; it undercounts if the '
                   'viewer presents faster than probe_hz / 2.'}
    (out / 'capture.json').write_text(json.dumps(cap, indent=1))
    print('RECORDED', json.dumps({k: cap[k] for k in ('seconds', 'probes', 'distinct_screen_changes', 'screen_change_rate_hz', 'stored_frames', 'recorder_busy_fraction')}), flush=True)


if __name__ == '__main__':
    main()
