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

Sobre `--clases`
----------------
Por defecto se evalúa contra TODOS los objetos móviles. Con `--clases Car` el GT
positivo queda solo con los autos y los puntos que caen en objetos móviles de
otras clases (peatón, ciclista, ...) pasan a una máscara de "ignorar": no cuentan
ni como acierto ni como error. Es lo correcto, porque el modelo es binario
móvil/estático y no tiene forma de saber la clase: castigarlo por acertar un
peatón móvil sería medir otra cosa. Con `--otras-como-estatico` se puede forzar
el criterio estricto (todo lo que no es auto cuenta como estático).

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

def contar(pred, gt, ignorar=None):
    """Cuenta aciertos y errores de un frame: (TP, FP, FN, TN).

    Versión en numpy de `tracker.metrics.contar_confusion` (que trabaja con
    tensores). Las métricas se arman con la misma función compartida, así el
    número de la inferencia es directamente comparable con el del entrenamiento.

    Args:
        pred: (N,) bool con la predicción binarizada.
        gt:   (N,) bool con la etiqueta.
        ignorar: (N,) bool opcional; esos puntos no se cuentan en ningún casillero.
    """
    if ignorar is not None:
        valido = ~ignorar
        pred, gt = pred[valido], gt[valido]
    return (float(np.sum(pred & gt)), float(np.sum(pred & ~gt)),
            float(np.sum(~pred & gt)), float(np.sum(~pred & ~gt)))


def barrer_umbral(probs, gts, ignorados=None, umbrales=None):
    """Busca el umbral que maximiza el IoU de la clase móvil.

    Con tanto desbalance, el 0.5 por defecto casi nunca es el mejor corte: la red
    sale poco confiada en la clase móvil y a 0.5 se queda callada. Acá probamos
    varios cortes sobre TODO el split y nos quedamos con el mejor.

    Args:
        probs: lista de arrays (N,) con las probabilidades por frame.
        gts:   lista de arrays (N,) bool con el GT por frame.
        ignorados: lista de arrays (N,) bool con los puntos a no contar, o None.

    Returns:
        (mejor_umbral, mejores_metricas, tabla) donde tabla es la lista de
        (umbral, iou_moving) recorrida.
    """
    if umbrales is None:
        umbrales = np.arange(0.05, 0.96, 0.05)
    if ignorados is None:
        ignorados = [None] * len(probs)

    tabla, mejor, mejor_u, mejor_m = [], -1.0, 0.5, {}
    for u in umbrales:
        tp = fp = fn = tn = 0.0
        for p, g, ign in zip(probs, gts, ignorados):
            a, b, c, d = contar(p > u, g, ign)
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
                   puntos_prev=None, feats_prev=None, puntos_comp=None,
                   puntos_ref2=None, feats_ref2=None, puntos_comp2=None,
                   devolver_feats=False):
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
        devolver_feats: si True, devuelve también las features por-punto del
            backbone (F,) promediadas por punto original -> para embeddings Re-ID.

    Returns:
        (N,) probabilidades por punto original; o (prob (N,), feats (N, F)) si
        devolver_feats=True.
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

    # Segunda referencia (multi-frame / flujo LiDAR denso): pc3/ft3 + comp a t-1.
    pc3 = ft3 = pc_comp2 = None
    if getattr(net, "use_multiframe", False):
        r = len(puntos_ref2)
        idx3 = np.tile(np.arange(r), int(np.ceil(num_points / r)))[:num_points]
        pc3 = _a_tensor(puntos_ref2, idx3, 3, device)
        ft3 = _a_tensor(feats_ref2, idx3, in_channels, device)
        pc_comp2 = _a_tensor(puntos_comp2, idx, 3, device)   # mismos idx que el frame actual

    with torch.no_grad():
        if getattr(net, "use_multiframe", False):
            seg, feat_pp = net(pc, ft, pc2, ft2, pc_comp,
                               pc3=pc3, feature3=ft3, pc1_comp2=pc_comp2)   # [1,1,P],[1,F,P]
        else:
            seg, feat_pp = net(pc, ft, pc2, ft2, pc_comp)    # [1,1,P], [1,F,P]
    prob_muestreada = seg.squeeze().cpu().numpy()        # (P,)

    # Promediamos las copias: cada punto original recibe la media de sus repeticiones.
    suma = np.zeros(n, dtype=np.float64)
    cuenta = np.zeros(n, dtype=np.float64)
    np.add.at(suma, idx, prob_muestreada)
    np.add.at(cuenta, idx, 1.0)
    prob = (suma / np.maximum(cuenta, 1.0)).astype(np.float32)

    if not devolver_feats:
        return prob

    # Features por-punto promediadas por punto original -> (N, F).
    fpp = feat_pp.squeeze(0).permute(1, 0).cpu().numpy()     # (P, F)
    F = fpp.shape[1]
    suma_f = np.zeros((n, F), dtype=np.float64)
    np.add.at(suma_f, idx, fpp)
    feats_orig = (suma_f / np.maximum(cuenta[:, None], 1.0)).astype(np.float32)
    return prob, feats_orig


def separar_por_clase(labels, clases):
    """Parte las cajas móviles del frame en (las de interés, las demás).

    Args:
        labels: dict {id: Object_3D} con los objetos móviles del frame.
        clases: conjunto de tipos a conservar (p. ej. {"Car"}), o None.

    Returns:
        (labels_interes, labels_otras); si `clases` es None, (labels, {}).
    """
    if not clases:
        return labels, {}
    interes = {k: v for k, v in labels.items() if v.type in clases}
    otras = {k: v for k, v in labels.items() if v.type not in clases}
    return interes, otras


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
    parser.add_argument("--clases", nargs="+", default=None, metavar="TIPO",
                        help="Tipos de objeto a evaluar (p. ej. --clases Car). "
                             "Por defecto: todos los móviles.")
    parser.add_argument("--otras-como-estatico", action="store_true",
                        help="Con --clases: cuenta los móviles de otras clases como "
                             "estáticos en vez de ignorarlos.")
    parser.add_argument("--sufijo", default="", help="Sufijo para la carpeta de salida.")
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)
    cfg.num_workers = 0
    cfg.eval = True                                      # split de validación
    cfg.aug = False                                      # nunca aumentar en inferencia

    exp = getattr(cfg, "exp_name", "seg") + args_cli.sufijo
    dir_vis = os.path.join(args_cli.output, exp, "vis")
    dir_vis_bev = os.path.join(args_cli.output, exp, "vis_bev")
    dir_data = os.path.join(args_cli.output, exp, "data")

    clases = set(args_cli.clases) if args_cli.clases else None
    if clases:
        modo = "estáticos" if args_cli.otras_como_estatico else "ignorados"
        print(f"[inferencia] evaluando SOLO las clases {sorted(clases)}; "
              f"los móviles de otras clases quedan {modo}.")

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
        margen = float(getattr(cfg, "gt_box_margin", 0.0))

        lbl_interes, lbl_otras = separar_por_clase(lbl1, clases)
        gt = moving_point_labels(lbl_interes, pc_t, transforms,
                                 margen=margen).numpy().astype(bool)
        # Los móviles de otras clases no se pueden premiar ni castigar: el modelo
        # es binario y no distingue clases (salvo que se pida el criterio estricto).
        if lbl_otras and not args_cli.otras_como_estatico:
            ign = moving_point_labels(lbl_otras, pc_t, transforms,
                                      margen=margen).numpy().astype(bool)
            ign &= ~gt                       # si además es auto, manda el auto
        else:
            ign = np.zeros(len(puntos), dtype=bool)

        prob = predecir_frame(net, puntos, feats, cfg.num_points,
                              int(getattr(cfg, "in_channels", 2)),
                              puntos_prev=muestra[1], feats_prev=muestra[3],
                              puntos_comp=muestra[4],
                              puntos_ref2=muestra[17], feats_ref2=muestra[18],
                              puntos_comp2=muestra[19])

        frames.append({"pts": puntos[:, :3], "prob": prob, "gt": gt, "ign": ign,
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
    igns = [f["ign"] for f in frames]

    def globales(umbral, post=False):
        tp = fp = fn = tn = 0.0
        for f in frames:
            if post:
                pred, _, _ = postprocesar(f["pts"], f["prob"], umbral=umbral)
            else:
                pred = f["prob"] > umbral
            a, b, c, d = contar(pred, f["gt"], f["ign"])
            tp += a; fp += b; fn += c; tn += d
        return metricas_de_conteos(tp, fp, fn, tn)

    def miou_ratrack(umbral, post=False):
        """mIoU con la convención de RaTrack (`main_utils.eval_motion_seg`): se
        calcula la mIoU binaria de CADA frame y se promedia sobre los frames.
        No es intercambiable con la micro-promediada de `globales`, que acumula
        TP/FP/FN/TN de todo el split; sobre este split difieren en más de 10
        puntos. Solo esta es comparable contra el 57.0 publicado por RaTrack."""
        acum = 0.0
        for f in frames:
            if post:
                pred, _, _ = postprocesar(f["pts"], f["prob"], umbral=umbral)
            else:
                pred = f["prob"] > umbral
            tp, fp, fn, tn = (v + 1e-20 for v in contar(pred, f["gt"], f["ign"]))
            acum += 0.5 * (tp / (tp + fp + fn + 1e-4) + tn / (tn + fp + fn + 1e-4))
        return acum / max(len(frames), 1)

    m_base = globales(args_cli.umbral)
    mejor_u, m_mejor, tabla = barrer_umbral(probs, gts, igns)
    umbral_final = args_cli.umbral if args_cli.fijar_umbral else mejor_u
    m_post = globales(umbral_final, post=not args_cli.sin_postproc)
    miou_rt = miou_ratrack(umbral_final, post=not args_cli.sin_postproc)

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
            m_frame = metricas_de_conteos(*contar(pred, f["gt"], f["ign"]))
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
    n_pts = sum(len(f["gt"]) for f in frames)
    n_gt = sum(int(f["gt"].sum()) for f in frames)
    n_ign = sum(int(f["ign"].sum()) for f in frames)

    resumen = [
        "=" * 74,
        f"RESULTADOS DE INFERENCIA — {exp}   ({len(frames)} frames)",
        "Métricas micro-promediadas: se acumulan TP/FP/FN/TN de todo el split.",
    ]
    if clases:
        modo = "como estáticos" if args_cli.otras_como_estatico else "ignorados"
        resumen += [
            f"Clases evaluadas    : {', '.join(sorted(clases))}   "
            f"(móviles de otras clases: {modo})",
            f"puntos: {n_pts} totales | {n_gt} móviles de la clase "
            f"({100.0 * n_gt / max(n_pts, 1):.2f}%) | {n_ign} ignorados "
            f"({100.0 * n_ign / max(n_pts, 1):.2f}%)",
        ]
    resumen += [
        "=" * 74,
        f"{'métrica':<14}{f'umbral {args_cli.umbral:.2f}':>18}"
        f"{f'umbral {umbral_final:.2f}':>18}{'+ post-proc':>18}",
    ]
    for k in ("miou", "iou_moving", "iou_static", "f1", "sen", "acc"):
        resumen.append(f"{k:<14}{m_base[k]:>18.4f}{m_mejor[k]:>18.4f}{m_post[k]:>18.4f}")
    resumen += [
        "=" * 74,
        f"mIoU con la convención de RaTrack (promedio POR FRAME): {miou_rt:.4f}",
        "Es la única comparable contra el 57.0 que publica RaTrack. La mIoU de la",
        "tabla de arriba es micro-promediada y NO es intercambiable con aquella.",
    ]
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
