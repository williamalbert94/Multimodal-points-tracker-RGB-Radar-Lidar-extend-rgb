"""HOTA para o método proposto e para RaTrack sob o MESMO protocolo.

RaTrack exporta objetos como CONJUNTOS DE PONTOS, não caixas. Para comparar,
mede-se tudo no espaço de pontos (que é o protocolo do próprio RaTrack): cada
objeto -- anotado ou predito -- vira o conjunto de índices da nuvem de radar que
lhe pertencem, e a similaridade é a IoU sobre esses conjuntos.

  GT      : pontos dentro das caixas móveis anotadas
  proposto: pontos dentro das caixas preditas (lidas dos .txt de track_inference)
  RaTrack : os pontos do próprio agrupamento

Uso:  run_hota.py proposto <dir_de_resultados> | run_hota.py ratrack <dir_base>
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/project"); sys.path.insert(0, "/project/examples/ratrack")
import paths
from point_iou import load_radar_cloud, frame_transforms, _gt_obb, _cluster_indices
from ratrack_io import parse_prediction_frame
from run_eval import _load_gt, _prediction_dir
from tracker.tracking.hota import hota

MODO, BASE = sys.argv[1], sys.argv[2]


def gt_sets(frame_id):
    import open3d as o3d
    radar = load_radar_cloud(frame_id)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(radar)
    globals()["_CLOUD"] = cloud
    t_rc, t_rl = frame_transforms(frame_id)
    ids, sets = [], []
    for g in _load_gt(frame_id):
        idx = set(_gt_obb(g, t_rc, t_rl).get_point_indices_within_bounding_box(cloud.points))
        if idx:
            ids.append(int(g.track_id)); sets.append(idx)
    return ids, sets, radar, cloud


def pred_sets_proposto(clip, frame_id, cloud):
    """Caixas preditas -> índices de pontos dentro, com EXATAMENTE a mesma
    rotina de contenção usada para o GT (OBB 3D do open3d, sem margem). Usar um
    teste BEV com margem aqui daria IoU < 1 mesmo para caixas idênticas e
    deprimiria DetA e AssA artificialmente."""
    import open3d as o3d
    from scipy.spatial.transform import Rotation as R3
    p = os.path.join(BASE, "data", clip, f"{frame_id}.txt")
    ids, sets = [], []
    if not os.path.exists(p):
        return ids, sets
    for ln in open(p):
        q = ln.split()
        if len(q) < 9:
            continue
        tid = int(float(q[1])); b = np.array(list(map(float, q[2:9])))
        rot = R3.from_euler("XYZ", [0, 0, b[6]]).as_matrix()
        obb = o3d.geometry.OrientedBoundingBox(b[:3].reshape(3, 1), rot,
                                               np.array(b[3:6]).reshape(3, 1))
        idx = set(obb.get_point_indices_within_bounding_box(cloud.points))
        if idx:
            ids.append(tid); sets.append(idx)
    return ids, sets


frames = []
for clip in paths.VAL_CLIPS:
    d = _prediction_dir(BASE, clip) if MODO == "ratrack" else os.path.join(BASE, "data", clip)
    if not d or not os.path.isdir(d):
        continue
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".txt")):
        fid = fn[:-4]
        gi, gs, radar, cloud = gt_sets(fid)
        if MODO == "ratrack":
            objs = parse_prediction_frame(os.path.join(d, fn))
            pi = [int(o.track_id) for o in objs]
            ps = [_cluster_indices(o.points, radar) for o in objs]
            pi, ps = zip(*[(a, b) for a, b in zip(pi, ps) if b]) if any(ps) else ([], [])
            pi, ps = list(pi), list(ps)
        else:
            pi, ps = pred_sets_proposto(clip, fid, cloud)
        frames.append((gi, pi, gs, ps))
    print(f"  {clip}: acumulado {len(frames)} quadros", flush=True)


def sim(gs, ps):
    M = np.zeros((len(gs), len(ps)))
    for a, A in enumerate(gs):
        for b, B in enumerate(ps):
            u = len(A | B)
            M[a, b] = len(A & B) / u if u else 0.0
    return M


r = hota(frames, sim)
print(f"\n=== HOTA ({MODO}) — IoU por pontos, {len(frames)} quadros ===")
print(f"  HOTA : {100*r['HOTA_mean']:.2f}")
print(f"  DetA : {100*r['DetA_mean']:.2f}")
print(f"  AssA : {100*r['AssA_mean']:.2f}")
