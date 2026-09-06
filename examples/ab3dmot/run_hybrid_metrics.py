"""Full metric suite for the "RaTrack detector + AB3DMOT association" ablation.

Produces the same columns as the results table (sAMOTA / AMOTA / AMOTP / MOTA /
MODA / MT / ML / IDSW) for two arms that share RaTrack's detections:

    arm A = RaTrack detector + RaTrack association   (its shipped ids)
    arm B = RaTrack detector + AB3DMOT association   (Kalman + Hungarian)

Both are scored with the SAME evaluator (`examples/ratrack/mot_metrics.py`), so
the A-vs-B delta is meaningful even where absolute values differ from RaTrack's
published table — its evaluator was never released, so arm A is also printed
next to the published row as a calibration check.

mIoU is deliberately not recomputed: segmentation comes from the detector, which
is identical in both arms, so the hybrid inherits RaTrack's mIoU by construction.

Association parameters default to AB3DMOT's own suggested `Pedestrian` values
(`giou_3d, -0.4, min_hits 1, max_age 4`) — the right analogue for VoD, whose
moving objects are almost all VRUs, and the only official config that establishes
the same number of tracks as RaTrack (so coverage is matched).
"""
import os
import sys
import numpy as np
from easydict import EasyDict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ratrack"))
from mot_metrics import compute_clearmot, compute_integral      # noqa: E402
from idsw_eval import idsw_report                               # noqa: E402

import run_hybrid as H                                          # noqa: E402
import run_ab3dmot as R                                         # noqa: E402
from AB3DMOT_libs.model import AB3DMOT                          # noqa: E402

# AB3DMOT official Pedestrian parameters (KITTI + pointrcnn branch of get_param)
ALGM, METRIC, THRES, MIN_HITS, MAX_AGE = "greedy", "giou_3d", -0.4, 1, 4

PUBLISHED = {"sAMOTA": 74.16, "AMOTA": 31.50, "AMOTP": 60.17,
             "MOTA": 67.27, "MODA": 77.83, "MT": 42.65, "ML": 14.71}


def make_official(max_age=MAX_AGE, min_hits=MIN_HITS):
    cfg = EasyDict({"dataset": "KITTI", "det_name": "pointrcnn",
                    "vis": False, "affi_pro": False, "ego_com": False})
    t = AB3DMOT(cfg, cat="Car", log=open(os.devnull, "w"))
    t.algm, t.metric, t.thres = ALGM, METRIC, THRES
    t.min_hits, t.max_age = min_hits, max_age
    t.max_sim, t.min_sim = (0.0, -100.) if METRIC.startswith("dist") else (1.0, -1.0)
    return t


def evaluate(frames, key):
    ev = H.to_eval(frames, key)

    # AB3DMOT reports MOTA/MODA/MT/ML at the confidence threshold that MAXIMISES
    # MOTA, not at zero. Keeping every cluster (conf_thr=0) floods the count with
    # false positives and drives MOTA negative, so sweep and take the best.
    confs = sorted({c for _, _, cs, _ in ev for c in cs})
    cands = list(np.quantile(confs, np.linspace(0, 1, 40))) if confs else [0.0]
    single, best = None, -1e9
    for thr in cands:
        m = compute_clearmot(ev, conf_thr=float(thr), iou_thr=H.IOU_THR)
        if m["mota"] > best:
            best, single = m["mota"], m

    integ = compute_integral(ev, iou_thr=H.IOU_THR, num_sample_pts=41)
    idsw, per_track = idsw_report(ev, iou_thr=H.IOU_THR)
    return {
        "sAMOTA": 100 * integ.get("sAMOTA", float("nan")),
        "AMOTA": 100 * integ.get("AMOTA", float("nan")),
        "AMOTP": 100 * integ.get("AMOTP", float("nan")),
        "MOTA": 100 * single["mota"],
        "MODA": 100 * single["moda"],
        "MT": 100 * single["mt"],
        "ML": 100 * single["ml"],
        "IDSW": idsw,
        "tracks": len(per_track),
    }


def main():
    R.make_tracker = make_official
    H.make_tracker = make_official

    frames = []
    for clip in H.VAL_CLIPS:
        fr = H.run_clip(clip, max_age=MAX_AGE)
        print(f"  {clip}: {len(fr)} frames")
        frames.extend(fr)

    a = evaluate(frames, "ratrack_ids")
    b = evaluate(frames, "ab3dmot_ids")

    cols = ["sAMOTA", "AMOTA", "AMOTP", "MOTA", "MODA", "MT", "ML", "IDSW", "tracks"]
    print("\n=== Same detections (RaTrack clusters), different association ===\n")
    print(f"{'':44s}" + "".join(f"{c:>9s}" for c in cols))
    for name, m in (("A: RaTrack det + RaTrack assoc (shipped)", a),
                    ("B: RaTrack det + AB3DMOT assoc (official)", b)):
        row = "".join(f"{m[c]:9.2f}" if isinstance(m[c], float) else f"{m[c]:9d}"
                      for c in cols)
        print(f"{name:44s}{row}")

    print(f"\n{'RaTrack PUBLISHED (their own evaluator)':44s}" +
          "".join(f"{PUBLISHED.get(c, float('nan')):9.2f}" if c in PUBLISHED
                  else f"{'-':>9s}" for c in cols))
    print("\nArm A vs the published row calibrates our evaluator against theirs;\n"
          "the A-vs-B delta is the actual result and is evaluator-independent.")
    print("mIoU is identical for both arms (same detector) -> inherit RaTrack's.")


if __name__ == "__main__":
    main()
