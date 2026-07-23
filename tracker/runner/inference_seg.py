"""Inferencia y evaluación del modelo de segmentación de puntos.

Carga un checkpoint entrenado, corre sobre el split de validación y por cada
frame produce:

  results/<exp>/vis/<clip>/<frame>.png       figura de 3 paneles (GT | RGB | pred)
  results/<exp>/vis_bev/<clip>/<frame>.png   figura de 2 paneles (sin la imagen)
  results/<exp>/data/<clip>/<frame>.txt      objetos móviles, formato tipo RaTrack

Las carpetas se van creando solas a medida que se necesitan.

Sobre el muestreo
-----------------
En entrenamiento la nube se muestrea al azar a `num_points`. Para inferencia eso
no sirve porque queremos una predicción por CADA punto real del radar, así que
repetimos la nube de forma determinística hasta llenar `num_points` (todos los
puntos entran al menos una vez) y después promediamos la probabilidad de las
copias de cada punto. Las métricas quedan calculadas sobre los puntos originales.

Uso:
    python -m tracker.runner.inference_seg \
        --config tracker/config/seg_exp_A_vrcomp.yaml \
        --checkpoint tracker/checkpoints/seg_exp_A_vrcomp/best_miou_model.pth
"""
import argparse
import os
import warnings

import numpy as np
import torch

from external.vod.configuration import VodTrackLocations
from external.vod.frame import FrameDataLoader
from tracker.config import load_config
from tracker.dataset.datagen_vod import TrackingDataVOD
from tracker.dataset.gt_labels import moving_point_labels
from tracker.metrics import metricas_de_conteos
from tracker.model import build_model
from tracker.runner.eval_utils.postproc import postprocesar
from tracker.visualization.seg_viz import figura_bev, figura_comparacion

EPS = 1e-20


# ─────────────────────────────────────────────────────────────────────────────
# Métricas sobre los puntos originales
# ─────────────────────────────────────────────────────────────────────────────

def contar(pred, gt):
    """Cuenta aciertos y errores de un frame: (TP, FP, FN, TN).

    Versión en numpy de `tracker.metrics.contar_confusion` (que trabaja con
    tensores). Las métricas se arman con la misma función compartida, así el
    número de la inferencia es directamente comparable con el del entrenamiento.
    """
    return (float(np.sum(pred & gt)), float(np.sum(pred & ~gt)),
            float(np.sum(~pred & gt)), float(np.sum(~pred & ~gt)))


def barrer_umbral(probs, gts, umbrales=None):
    """Busca el umbral que maximiza el IoU de la clase móvil.

    Con tanto desbalance, el 0.5 por defecto casi nunca es el mejor corte: la red
    sale poco confiada en la clase móvil y a 0.5 se queda callada. Acá probamos
    varios cortes sobre TODO el split y nos quedamos con el mejor.

    Args:
        probs: lista de arrays (N,) con las probabilidades por frame.
        gts:   lista de arrays (N,) bool con el GT por frame.

    Returns:
        (mejor_umbral, mejores_metricas, tabla) donde tabla es la lista de
        (umbral, iou_moving) recorrida.
    """
    if umbrales is None:
        umbrales = np.arange(0.05, 0.96, 0.05)

    tabla, mejor, mejor_u, mejor_m = [], -1.0, 0.5, {}
    for u in umbrales:
        tp = fp = fn = tn = 0.0
        for p, g in zip(probs, gts):
            a, b, c, d = contar(p > u, g)
            tp += a; fp += b; fn += c; tn += d
        m = metricas_de_conteos(tp, fp, fn, tn)
        tabla.append((float(u), m["iou_moving"]))
        if m["iou_moving"] > mejor:
            mejor, mejor_u, mejor_m = m["iou_moving"], float(u), m
    return mejor_u, mejor_m, tabla


# ─────────────────────────────────────────────────────────────────────────────
# Predicción de un frame
# ─────────────────────────────────────────────────────────────────────────────

def _a_tensor(arr, idx, cols, device):
    """Toma las filas `idx` y las primeras `cols` columnas -> tensor [1, cols, P]."""
    return torch.from_numpy(arr[idx, :cols]).float().T.unsqueeze(0).to(device)


def predecir_frame(net, puntos, feats, num_points, in_channels, device="cuda",
                   puntos_prev=None, feats_prev=None, puntos_comp=None):
    """Devuelve la probabilidad de "móvil" para cada punto ORIGINAL de la nube.

    Repite los puntos de forma determinística hasta `num_points`, corre la red y
    luego promedia las probabilidades de las copias de cada punto original.

    Args:
        net: la red ya en modo eval.
        puntos: (N, 3) coordenadas del radar del frame actual.
        feats:  (N, C) atributos del radar.
        num_points: cuántos puntos espera la red.
        in_channels: cuántos canales de atributos usa la red.
        puntos_prev / feats_prev: frame anterior (solo si la red usa rama temporal).
        puntos_comp: frame actual compensado por ego-movimiento.

    Returns:
        (N,) probabilidades por punto original.
    """
    n = len(puntos)
    # Índices que cubren todos los puntos al menos una vez.
    repes = int(np.ceil(num_points / n))
    idx = np.tile(np.arange(n), repes)[:num_points]

    pc = _a_tensor(puntos, idx, 3, device)                       # [1,3,P]
    ft = _a_tensor(feats, idx, in_channels, device)              # [1,C,P]

    pc2 = ft2 = pc_comp = None
    if getattr(net, "use_temporal", False):
        # El frame anterior se muestrea igual (tiene otra cantidad de puntos).
        m = len(puntos_prev)
        idx2 = np.tile(np.arange(m), int(np.ceil(num_points / m)))[:num_points]
        pc2 = _a_tensor(puntos_prev, idx2, 3, device)
        ft2 = _a_tensor(feats_prev, idx2, in_channels, device)
        pc_comp = _a_tensor(puntos_comp, idx, 3, device)

    with torch.no_grad():
        seg, _ = net(pc, ft, pc2, ft2, pc_comp)          # [1, 1, P]
    prob_muestreada = seg.squeeze().cpu().numpy()        # (P,)

    # Promediamos las copias: cada punto original recibe la media de sus repeticiones.
    suma = np.zeros(n, dtype=np.float64)
    cuenta = np.zeros(n, dtype=np.float64)
    np.add.at(suma, idx, prob_muestreada)
    np.add.at(cuenta, idx, 1.0)
    return (suma / np.maximum(cuenta, 1.0)).astype(np.float32)


def escribir_txt(ruta, clusters, puntos, prob):
    """Guarda los objetos móviles con el mismo formato que usa RaTrack.

    Un renglón por objeto:
        NA 1 -1 -1 <score> <id> <x y z> <x y z> ...

    donde `score` es la probabilidad media del grupo y luego vienen las
    coordenadas de todos sus puntos.
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as f:
        for i, miembros in enumerate(clusters):
            score = float(prob[miembros].mean())
            coords = " ".join(f"{v}" for v in puntos[miembros, :3].reshape(-1))
            f.write(f"NA 1 -1 -1 {score} {i} {coords}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inferencia de segmentación de puntos.")
    parser.add_argument("--config", required=True, help="YAML de configuración.")
    parser.add_argument("--checkpoint", required=True, help="Ruta al .pth entrenado.")
    parser.add_argument("--output", default="tracker/results", help="Carpeta raíz de salida.")
    parser.add_argument("--umbral", type=float, default=0.5, help="Corte para decidir móvil.")
    parser.add_argument("--limit", type=int, default=0, help="Máximo de frames (0 = todos).")
    parser.add_argument("--cada", type=int, default=1, help="Guardar figura cada N frames.")
    parser.add_argument("--sin-postproc", action="store_true", help="Apaga el post-procesamiento.")
    parser.add_argument("--fijar-umbral", action="store_true",
                        help="Usa el --umbral dado en vez del mejor del barrido.")
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)
    cfg.num_workers = 0
    cfg.eval = True                                      # split de validación
    cfg.aug = False                                      # nunca aumentar en inferencia

    exp = getattr(cfg, "exp_name", "seg")
    dir_vis = os.path.join(args_cli.output, exp, "vis")
    dir_vis_bev = os.path.join(args_cli.output, exp, "vis_bev")
    dir_data = os.path.join(args_cli.output, exp, "data")

    # ── Modelo ───────────────────────────────────────────────────────────────
    net = build_model(cfg)
    ckpt = torch.load(args_cli.checkpoint, map_location="cuda")
    estado = ckpt.get("model", ckpt)
    net.load_state_dict(estado)
    net.eval()
    print(f"[inferencia] checkpoint cargado: {args_cli.checkpoint} "
          f"(época {ckpt.get('epoch', '?')}, mIoU {ckpt.get('miou', float('nan')):.4f})")

    # ── Datos ────────────────────────────────────────────────────────────────
    ds = TrackingDataVOD(cfg, cfg.dataset_path)
    total = len(ds) if args_cli.limit <= 0 else min(args_cli.limit, len(ds))
    print(f"[inferencia] frames a procesar: {total}")

    kitti = VodTrackLocations(root_dir=cfg.dataset_path, output_dir=cfg.dataset_path,
                              frame_set_path="", pred_dir="")

    # ── Pasada 1: inferencia de todos los frames ─────────────────────────────
    # Guardamos probabilidades y GT para poder barrer el umbral después sobre
    # todo el split (es poca memoria: ~300 puntos por frame).
    frames = []
    for i in range(total):
        try:
            muestra = ds[i]
        except Exception as e:                           # frame dañado: seguimos
            print(f"  [aviso] frame {i} no se pudo cargar: {type(e).__name__}")
            continue

        # Se segmenta raw_pc0, que va con lbl1 y transforms1 (ver datagen_vod).
        puntos, feats = muestra[0], muestra[2]
        num_frame, clip = muestra[5], muestra[6]
        lbl1, transforms = muestra[12], muestra[14]
        if len(puntos) == 0:
            continue

        pc_t = torch.from_numpy(puntos[:, :3]).float().T.unsqueeze(0)
        gt = moving_point_labels(lbl1, pc_t, transforms,
                                 margen=float(getattr(cfg, "gt_box_margin", 0.0))).numpy().astype(bool)
        prob = predecir_frame(net, puntos, feats, cfg.num_points,
                              int(getattr(cfg, "in_channels", 2)),
                              puntos_prev=muestra[1], feats_prev=muestra[3],
                              puntos_comp=muestra[4])

        frames.append({"pts": puntos[:, :3], "prob": prob, "gt": gt,
                       "clip": clip, "nombre": f"{int(num_frame):05d}",
                       "tf": transforms})
        if len(frames) % 100 == 0:
            print(f"  inferidos {len(frames)}/{total}")

    if not frames:
        print("No se pudo procesar ningún frame.")
        return

    # ── Umbral: el 0.5 de referencia y el mejor del barrido ──────────────────
    probs = [f["prob"] for f in frames]
    gts = [f["gt"] for f in frames]

    def globales(umbral, post=False):
        tp = fp = fn = tn = 0.0
        for f in frames:
            if post:
                pred, _, _ = postprocesar(f["pts"], f["prob"], umbral=umbral)
            else:
                pred = f["prob"] > umbral
            a, b, c, d = contar(pred, f["gt"])
            tp += a; fp += b; fn += c; tn += d
        return metricas_de_conteos(tp, fp, fn, tn)

    m_base = globales(args_cli.umbral)
    mejor_u, m_mejor, tabla = barrer_umbral(probs, gts)
    umbral_final = args_cli.umbral if args_cli.fijar_umbral else mejor_u
    m_post = globales(umbral_final, post=not args_cli.sin_postproc)

    # ── Pasada 2: guardar salidas con el umbral elegido ──────────────────────
    guardados = 0
    for n, f in enumerate(frames):
        if args_cli.sin_postproc:
            pred, clusters, prob = f["prob"] > umbral_final, [], f["prob"]
        else:
            pred, clusters, prob = postprocesar(f["pts"], f["prob"], umbral=umbral_final)

        escribir_txt(os.path.join(dir_data, f["clip"], f"{f['nombre']}.txt"),
                     clusters, f["pts"], prob)

        if n % max(args_cli.cada, 1) == 0:
            m_frame = metricas_de_conteos(*contar(pred, f["gt"]))
            try:
                imagen = FrameDataLoader(kitti_locations=kitti,
                                         frame_number=f["nombre"]).image
            except Exception:
                imagen = None
            figura_comparacion(os.path.join(dir_vis, f["clip"], f"{f['nombre']}.png"),
                               f["pts"], f["gt"], pred, imagen, f["tf"],
                               f["clip"], f["nombre"], m_frame)
            figura_bev(os.path.join(dir_vis_bev, f["clip"], f"{f['nombre']}.png"),
                       f["pts"], f["gt"], pred, f["clip"], f["nombre"], m_frame)
            guardados += 1

    # ── Resumen ──────────────────────────────────────────────────────────────
    resumen = [
        "=" * 74,
        f"RESULTADOS DE INFERENCIA — {exp}   ({len(frames)} frames)",
        "Métricas micro-promediadas: se acumulan TP/FP/FN/TN de todo el split.",
        "=" * 74,
        f"{'métrica':<14}{f'umbral {args_cli.umbral:.2f}':>18}"
        f"{f'umbral {umbral_final:.2f}':>18}{'+ post-proc':>18}",
    ]
    for k in ("miou", "iou_moving", "iou_static", "f1", "sen", "acc"):
        resumen.append(f"{k:<14}{m_base[k]:>18.4f}{m_mejor[k]:>18.4f}{m_post[k]:>18.4f}")
    resumen += [
        "=" * 74,
        "Barrido de umbral (IoU móvil):",
        "  " + "  ".join(f"{u:.2f}:{v:.3f}" for u, v in tabla),
        "=" * 74,
        f"umbral elegido      : {umbral_final:.2f}",
        f"figuras (3 paneles) : {dir_vis}  ({guardados} figuras)",
        f"figuras (2 paneles) : {dir_vis_bev}",
        f"txt tipo RaTrack    : {dir_data}",
    ]
    texto = "\n".join(resumen)
    print("\n" + texto)

    os.makedirs(os.path.join(args_cli.output, exp), exist_ok=True)
    with open(os.path.join(args_cli.output, exp, "metrics.txt"), "w") as f:
        f.write(texto + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
