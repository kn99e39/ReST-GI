"""Scene/checkpoint inputs shared by native and bridged tile rendering."""
import importlib
import importlib.util
import sys
from pathlib import Path


def checkpoint_path(args):
    if args.checkpoint:
        return Path(args.checkpoint).resolve()
    return Path(args.upstream) / 'checkpoints' / (args.asset + '.ckpt')


def scene_source_path(args):
    if args.scene_bridge:
        return Path(args.scene_bridge).resolve()
    return Path(args.upstream) / 'scenes' / args.asset / '__init__.py'


def scene_definition(args):
    if args.scene_bridge:
        path = scene_source_path(args)
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('render_scene_bridge', path)
        bridge = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bridge
        spec.loader.exec_module(bridge)
        return bridge.make_scene(args.res, max_depth=7)[0]
    return importlib.import_module('scenes.' + args.asset).scene1(args.res)
