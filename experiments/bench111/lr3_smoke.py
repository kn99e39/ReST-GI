import os, sys, json, numpy as np
import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); import drjit as dr
sys.path.insert(0, os.environ['TRANSGI_REPO'])
from transgi import config
from transgi.session import Session
cfg = config.load(os.path.join(os.environ['TRANSGI_REPO'], 'configs', 'living_room_3.yaml'))
sess = Session(cfg, res=[640, 360])
print('manifest', sess.sd.manifest['modified_sha256'], mi.__version__)
sc, se = sess.sd.mi_scene, sess.sensor
full = mi.render(sc, sensor=se, integrator=mi.load_dict({'type':'path','max_depth':65}), spp=1, seed=3)
di = mi.render(sc, sensor=se, integrator=mi.load_dict({'type':'path','max_depth':2}), spp=1, seed=3)
a = mi.render(sc, sensor=se, integrator=mi.load_dict({'type':'aov','aovs':'albedo:albedo,nn:sh_normal'}), spp=1, seed=4)
print('shapes', full.shape, di.shape, a.shape)
gi = full - di
den = mi.OptixDenoiser(input_size=[640, 360], albedo=True, normals=True, temporal=False)
y = den(mi.TensorXf(gi), albedo=a[:, :, 0:3], normals=a[:, :, 3:6], to_sensor=se.world_transform().inverse())
print('den', y.shape, float(np.asarray(gi).mean()), float(np.asarray(y).mean()), 'neg_gi_frac', float((np.asarray(gi) < -1e-6).mean()))
