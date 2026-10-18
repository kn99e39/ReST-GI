"""Read-only inspection of the official RNA Lego blend: objects, materials (node graphs, Principled values), world, AOVs."""
import json, sys
import bpy

bpy.ops.wm.open_mainfile(filepath=sys.argv[1])
sc = bpy.context.scene
out = {"objects": [], "materials": {}, "world": {}, "view_layer_aovs": [], "cycles": {}}
for o in bpy.data.objects:
    out["objects"].append({"name": o.name, "type": o.type, "loc": list(o.location), "rot": list(o.rotation_euler), "scale": list(o.scale),
                           "materials": [s.material.name if s.material else None for s in getattr(o, "material_slots", [])],
                           "dims": list(o.dimensions), "hide_render": o.hide_render})
for m in bpy.data.materials:
    d = {"use_nodes": m.use_nodes, "nodes": []}
    if m.use_nodes:
        for n in m.node_tree.nodes:
            e = {"type": n.bl_idname, "name": n.name}
            if n.bl_idname == "ShaderNodeOutputAOV":
                e["aov"] = n.name if not hasattr(n, "aov_name") else n.aov_name
            vals = {}
            for i in n.inputs:
                if not i.is_linked and hasattr(i, "default_value"):
                    try:
                        v = i.default_value; vals[i.name] = list(v) if hasattr(v, "__len__") else float(v)
                    except Exception:
                        pass
            e["inputs"] = vals
            e["linked_inputs"] = [i.name for i in n.inputs if i.is_linked]
            d["nodes"].append(e)
        d["links"] = [f"{l.from_node.name}.{l.from_socket.name}->{l.to_node.name}.{l.to_socket.name}" for l in m.node_tree.links]
    out["materials"][m.name] = d
w = sc.world
out["world"] = {"name": w.name if w else None, "use_nodes": bool(w and w.use_nodes)}
if w and w.use_nodes:
    out["world"]["nodes"] = [{"type": n.bl_idname, "inputs": {i.name: (list(i.default_value) if hasattr(i.default_value, "__len__") else i.default_value) for i in n.inputs if hasattr(i, "default_value") and not i.is_linked}} for n in w.node_tree.nodes]
for vl in sc.view_layers:
    out["view_layer_aovs"] += [{"layer": vl.name, "aov": a.name, "type": a.type} for a in vl.aovs]
out["use_nodes_compositor"] = sc.use_nodes
if sc.use_nodes:
    out["compositor"] = [{"type": n.bl_idname, "name": n.name, "extra": {k: getattr(n, k) for k in ("base_path",) if hasattr(n, k)},
                          "slots": [s.path for s in getattr(n, "file_slots", [])]} for n in sc.node_tree.nodes]
out["cycles"] = {"seed": sc.cycles.seed, "use_animated_seed": sc.cycles.use_animated_seed, "samples": sc.cycles.samples,
                 "film_transparent": sc.render.film_transparent, "unit_scale": sc.unit_settings.scale_length}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
print("done")
