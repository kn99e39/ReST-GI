"""Spawned GPU tile workers for the native released 8DNA integrator on Linux."""
import faulthandler
import fcntl
import importlib
import json
import math
import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from render import atomic_json, sha


def select_worker_count(args, output):
    """Measure a representative tile before admitting GPU-resident processes."""
    profile = output / 'worker_memory_probe'
    size = min(args.res, args.tile_size + 2)
    offset = (args.res-size)//2
    command = [sys.executable, str(Path(__file__).with_name('diagnose_fabric.py')),
               '--upstream', args.upstream, '--out', str(profile), '--vcall', str(args.vcall),
               '--res', str(args.res), '--spp', str(args.chunk),
               '--crop', str(offset), str(offset), str(size), str(size)]
    profile.mkdir(exist_ok=True)
    with open(profile/'run.log', 'w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.task_timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code != 0:
        raise RuntimeError('Worker memory calibration failed; see worker_memory_probe/run.log')
    result = json.loads((profile/'phase.json').read_text())
    if result['phase'] != 'complete' or result['gpu_process_mib'] <= 0:
        raise RuntimeError('Worker memory calibration did not complete')
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    # Chosen headroom covers allocator differences between tiles.
    estimate = math.ceil(result['gpu_process_mib'] * 1.2)
    admitted = min(args.workers, max(0, (free-512)//estimate))
    if admitted < 1:
        raise RuntimeError('Insufficient free VRAM for one calibrated worker')
    record = {'requested_cpu_workers': args.workers, 'worker_gpu_mib': result['gpu_process_mib'],
              'free_gpu_mib': free, 'worker_budget_mib': estimate, 'reserve_mib':512,
              'selected_workers': admitted}
    atomic_json(output/'worker_admission.json', record)
    print(json.dumps({'event':'worker_admission', **record}),flush=True)
    return admitted


def read_worker_cpu(processes, measured):
    ticks = os.sysconf('SC_CLK_TCK')
    for rank, process in enumerate(processes):
        if process.pid is None:
            continue
        try:
            fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
            measured[rank] = max(measured[rank], (int(fields[11])+int(fields[12]))/ticks)
        except FileNotFoundError:
            pass


def worker_main(args, tasks, events, rank, identity):
    import ctypes
    os.setsid()
    # End the GPU context if its coordinator disappears unexpectedly.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() == 1:
        return
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.cpu_threads)
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.task_timeout, repeat=True)
    try:
        import numpy as np
        import torch
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        import drjit as dr
        root = Path(args.upstream)
        os.chdir(root / 'demo')
        sys.path.insert(0, str(root))
        import models.integrator
        torch.set_num_threads(args.cpu_threads)
        dr.set_flag(dr.JitFlag.LoopRecord, False)
        dr.set_flag(dr.JitFlag.VCallRecord, bool(args.vcall))
        definition = importlib.import_module('scenes.' + args.asset).scene1(args.res)
        scene = mi.load_dict(definition)
        integrator = scene.integrator() if args.method == 'pt' else mi.load_dict({
            'type': 'neuralpath', 'max_depth': 10, 'rr_depth': 5, 'device': 0,
            'model_type': '8dna', 'asset_instance0': str(root / 'checkpoints' / (args.asset + '.ckpt'))})
        border = int(math.ceil(scene.sensors()[0].film().rfilter().radius()))
        events.put({'event': 'ready', 'worker': rank, 'pid': os.getpid()})
        while True:
            task = tasks.get()
            if task is None:
                break
            index, x, y, width, height = task
            events.put({'event': 'started', 'worker': rank, 'tile': index})
            left, top = max(0, x - border), max(0, y - border)
            right, bottom = min(args.res, x + width + border), min(args.res, y + height + border)
            sensor_definition = definition['camera'].copy()
            sensor_definition['film'] = sensor_definition['film'].copy()
            sensor_definition['film'].update(crop_width=right-left, crop_height=bottom-top,
                crop_offset_x=left, crop_offset_y=top)
            sensor = mi.load_dict(sensor_definition)
            path = Path(args.tile_directory) / f'tile_{index:05d}.npz'
            total = np.zeros((height, width, 3), dtype=np.float64)
            first128 = np.zeros_like(total)
            completed = 0
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved['identity']) != identity:
                        raise ValueError('Tile cache belongs to another render configuration')
                    total, first128 = saved['sum'], saved['first128']
                    completed = int(saved['completed_spp'])
                if total.shape != (height,width,3) or first128.shape != total.shape:
                    raise ValueError('Invalid cached tile dimensions')
                if not np.isfinite(total).all() or not np.isfinite(first128).all():
                    raise ValueError('Nonfinite cached tile')
            if completed > args.spp:
                raise ValueError('Use the previous target or a higher spp when resuming partial tiles')
            tile_start = time.perf_counter()
            while completed < args.spp:
                count = min(args.chunk, args.spp - completed)
                if completed < 128:
                    count = min(count, 128 - completed)
                seedbase = (210000 if args.method == 'neural' else 410000) + 100000 * args.rep
                seed = seedbase + index * 1000000 + completed
                image = mi.render(scene, sensor=sensor, integrator=integrator, spp=count, seed=seed)
                dr.eval(image)
                torch.cuda.synchronize()
                pixels = np.array(image, dtype=np.float64)[y-top:y-top+height, x-left:x-left+width]
                if pixels.shape != total.shape or not np.isfinite(pixels).all():
                    raise ValueError('Invalid tile image')
                total += pixels * count
                if completed < 128:
                    first128 += pixels * count
                completed += count
                if completed % 32 == 0 or completed == args.spp:
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez(temporary, sum=total, first128=first128,
                             completed_spp=completed, identity=identity)
                    temporary.replace(path)
                events.put({'event': 'progress', 'worker': rank, 'tile': index,
                            'spp': completed, 'pixels': width * height,
                            'cpu_seconds': os.times().user + os.times().system})
                faulthandler.dump_traceback_later(args.task_timeout, repeat=True)
            events.put({'event': 'tile_complete', 'worker': rank, 'tile': index,
                        'spp': completed, 'wall_s': time.perf_counter()-tile_start,
                        'cpu_seconds': os.times().user + os.times().system,
                        'peak_torch_bytes': torch.cuda.max_memory_allocated()})
        faulthandler.cancel_dump_traceback_later()
    except BaseException:
        events.put({'event': 'error', 'worker': rank, 'traceback': traceback.format_exc()})
        raise


def stop_workers(processes):
    for process in processes:
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                process.terminate()
    for process in processes:
        if process.pid is None:
            continue
        process.join(timeout=3)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.join(timeout=3)


def run(args):
    import hashlib
    import numpy as np
    entry_started = time.monotonic()
    if args.workers < 1 or args.res < 1 or args.chunk < 1 or args.spp < 1:
        raise ValueError('Workers, resolution, chunk and spp must be positive')
    if args.task_timeout <= 0 or args.cpu_threads < 1 or args.rep not in (0, 1):
        raise ValueError('Use a positive timeout/thread count and released replicate 0 or 1')
    if args.spp >= 100000:
        raise ValueError('The tile seed schedule supports fewer than 100000 spp')
    args.upstream = str(Path(args.upstream).resolve())
    args.out = str(Path(args.out).resolve())
    args.tile_size = args.tile_size or 128
    if args.tile_size < 1:
        raise ValueError('Tile size must be positive')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    key = f'{args.asset}_{args.method}_r{args.rep}_{args.res}'
    lock = open(output / (key + '.lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tiles = []
    for y in range(0, args.res, args.tile_size):
        for x in range(0, args.res, args.tile_size):
            tiles.append((len(tiles), x, y, min(args.tile_size, args.res-x), min(args.tile_size, args.res-y)))
    if 510000 + (len(tiles)-1) * 1000000 + args.spp >= 2**32:
        raise ValueError('This tile count exceeds the uint32 seed schedule')
    root = Path(args.upstream)
    config = {'mode': 'parallel_tiles_v1', 'asset': args.asset, 'method': args.method,
              'rep': args.rep, 'resolution': args.res, 'tile_size': args.tile_size,
              'chunk': args.chunk, 'vcall': bool(args.vcall),
              'checkpoint_sha256': sha(root/'checkpoints'/f'{args.asset}.ckpt'),
              'scene_source_sha256': sha(root/'scenes'/args.asset/'__init__.py'),
              'seed_schedule': 'method base + 100000*rep + 1000000*tile_index + completed_spp',
              'reconstruction': 'Filter-radius overlap rendered around each tile, then interior crop'}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.tile_directory = str(output / (key + '_tiles'))
    Path(args.tile_directory).mkdir(exist_ok=True)
    metadata_path = output / (key + '.json')
    prior = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if prior and prior.get('execution_identity') != identity:
        raise ValueError('Output has another execution configuration. Select a separate --out directory.')
    if prior and prior['completed_spp'] >= args.spp:
        print(json.dumps({'event': 'already_complete', 'key': key, 'spp': prior['completed_spp']}), flush=True)
        return
    requested_workers = args.workers
    if args.auto_workers:
        if args.asset != 'fabric' or args.method != 'neural':
            raise ValueError('Automatic memory calibration currently targets Fabric neural rendering')
        args.workers = select_worker_count(args, output)
    metadata = {'key': key, 'arguments': vars(args).copy(), 'execution': config,
                'execution_identity': identity, 'completed_spp': 0, 'chunks': [],
                'checkpoint_sha256': config['checkpoint_sha256'],
                'scene_source_sha256': config['scene_source_sha256'],
                'requested_workers': requested_workers,
                'jit_flags': {'LoopRecord': False, 'VCallRecord': bool(args.vcall)}}
    if args.auto_workers:
        metadata['worker_admission'] = json.loads((output/'worker_admission.json').read_text())
    context = multiprocessing.get_context('spawn')
    tasks, events = context.Queue(), context.Queue()
    processes = [context.Process(target=worker_main, args=(args, tasks, events, rank, identity))
                 for rank in range(args.workers)]
    for task in tiles:
        tasks.put(task)
    for _ in processes:
        tasks.put(None)
    started = time.monotonic()
    last_event = [started] * args.workers
    completed_tiles = set()
    tile_progress = {task[0]: 0 for task in tiles}
    for task in tiles:
        path = Path(args.tile_directory) / f'tile_{task[0]:05d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['identity']) != identity:
                    raise ValueError('Existing tile configuration differs')
                tile_progress[task[0]] = int(saved['completed_spp'])
    initial_samples = sum(tile_progress[t[0]] * t[3] * t[4] for t in tiles)
    completed_tiles = {index for index, spp in tile_progress.items() if spp == args.spp}
    if len(completed_tiles) == len(tiles):
        processes = []
        metadata['cache_only_completion'] = True
    cpu_seconds = [0.0] * args.workers
    next_report = started
    status_path = output / (key + '.progress.json')
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        for process in processes:
            process.start()
        while len(completed_tiles) < len(tiles):
            try:
                event = events.get(timeout=1)
                rank = event['worker']
                last_event[rank] = time.monotonic()
                cpu_seconds[rank] = event.get('cpu_seconds', cpu_seconds[rank])
                if event['event'] == 'error':
                    raise RuntimeError(event['traceback'])
                if event['event'] == 'progress':
                    tile_progress[event['tile']] = event['spp']
                if event['event'] == 'tile_complete':
                    completed_tiles.add(event['tile'])
                    tile_progress[event['tile']] = event['spp']
                    metadata['chunks'].append(event)
                if event['event'] in ('ready', 'started', 'tile_complete'):
                    print(json.dumps(event), flush=True)
            except queue.Empty:
                pass
            now = time.monotonic()
            for rank, process in enumerate(processes):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f'Worker {rank} exited with code {process.exitcode}')
                if process.is_alive() and now-last_event[rank] > args.task_timeout:
                    raise TimeoutError(f'Worker {rank} made no progress for {args.task_timeout} seconds')
            if now >= next_report:
                read_worker_cpu(processes, cpu_seconds)
                samples = sum(tile_progress[t[0]] * t[3] * t[4] for t in tiles)
                progress = {'status': 'rendering', 'key': key, 'workers': args.workers,
                            'completed_tiles': len(completed_tiles), 'total_tiles': len(tiles),
                            'pixel_samples': samples, 'target_pixel_samples': args.res**2 * args.spp,
                            'elapsed_seconds': now-started,
                            'worker_cpu_seconds': sum(cpu_seconds),
                            'average_cpu_cores': sum(cpu_seconds)/(now-started),
                            'estimated_remaining_seconds': (now-started) * (args.res**2 * args.spp-samples) / (samples-initial_samples) if samples > initial_samples else None,
                            'worker_pids': [p.pid for p in processes]}
                atomic_json(status_path, progress)
                print(json.dumps(progress), flush=True)
                next_report = now + 5
        read_worker_cpu(processes, cpu_seconds)
        # Every tile is already committed. Workers still initializing or exiting
        # can be released without delaying the completed image.
        stop_workers(processes)
        total = np.zeros((args.res, args.res, 3), dtype=np.float64)
        first128 = np.zeros_like(total)
        for index, x, y, width, height in tiles:
            with np.load(Path(args.tile_directory)/f'tile_{index:05d}.npz', allow_pickle=False) as saved:
                assert int(saved['completed_spp']) == args.spp
                total[y:y+height, x:x+width] = saved['sum']
                first128[y:y+height, x:x+width] = saved['first128']
        temporary = output / (key + '.tmp.npz')
        np.savez_compressed(temporary, sum=total, first128=first128)
        temporary.replace(output / (key + '.npz'))
        average = (total/args.spp).astype(np.float32)
        if not np.isfinite(average).all():
            raise ValueError('Nonfinite assembled image')
        np.save(output / (key + f'_s{args.spp}.npy'), average)
        # Export in the parent only after workers release their GPU contexts.
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        exr = output / (key + f'_s{args.spp}.exr')
        mi.Bitmap(average).write(str(exr))
        metadata.update(completed_spp=args.spp, timed_wall_s=time.monotonic()-started,
                        end_to_end_wall_s=time.monotonic()-entry_started,
                        worker_cpu_seconds=sum(cpu_seconds),
                        average_cpu_cores=sum(cpu_seconds)/(time.monotonic()-started),
                        last_checkpoint_sha256=sha(exr), status='complete')
        previous_wall = prior.get('cumulative_worker_wall_s', prior['timed_wall_s']) if prior else 0.0
        history_known = previous_wall is not None and not (initial_samples and prior is None)
        metadata['cumulative_worker_wall_s'] = previous_wall + metadata['timed_wall_s'] if history_known else None
        metadata['timing_scope'] = 'tiled worker startup, rendering and export; calibration separate'
        metadata['render_wall_s_at128'] = prior.get('render_wall_s_at128') if prior else None
        if args.spp == 128 and history_known:
            metadata['render_wall_s_at128'] = metadata['cumulative_worker_wall_s']
        atomic_json(metadata_path, metadata)
        atomic_json(status_path, {'status': 'complete', 'key': key, 'spp': args.spp,
                                 'elapsed_seconds': metadata['timed_wall_s']})
        print(json.dumps({'event': 'complete', 'key': key, 'wall_s': metadata['timed_wall_s']}), flush=True)
    except BaseException as error:
        atomic_json(status_path, {'status': 'failed', 'error': str(error),
                                 'completed_tiles': len(completed_tiles)})
        raise
    finally:
        stop_workers(processes)
        signal.signal(signal.SIGTERM, previous_handler)
        tasks.cancel_join_thread()
        tasks.close()
        events.close()
        lock.close()
