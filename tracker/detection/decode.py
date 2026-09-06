"""Decodifica las predicciones por punto de la cabecera en cajas 3D + NMS."""
import numpy as np
import torch

from .metrics_3d import compute_rotated_iou_2d


def _nms_bev(boxes, scores, iou_thr=0.3):
    """NMS greedy con IoU-BEV rotado. boxes: [K,7], scores: [K]. -> índices que quedan."""
    if len(boxes) == 0:
        return []
    orden = np.argsort(-scores)
    keep = []
    while len(orden) > 0:
        i = orden[0]
        keep.append(i)
        resto = orden[1:]
        if len(resto) == 0:
            break
        ious = np.array([compute_rotated_iou_2d(boxes[i], boxes[j]) for j in resto])
        orden = resto[ious < iou_thr]
    return keep


def decode_boxes(pred, pc1_xyz, umbral=0.3, nms_iou=0.3, max_cajas=100):
    """Convierte las predicciones por punto en cajas por muestra.

    Args:
        pred: dict con conf [B,1,N] (logit), offset [B,3,N], size [B,3,N] (log),
              yaw [B,2,N] (sin,cos).
        pc1_xyz: [B, 3, N] coordenadas de los puntos.
        umbral: corte de confianza. nms_iou: IoU para NMS. max_cajas: tope por frame.

    Returns:
        lista de B tuplas (boxes [K,7], scores [K]).
    """
    conf = torch.sigmoid(pred["conf"]).squeeze(1)        # [B, N]
    offset = pred["offset"]                              # [B, 3, N]
    size = torch.exp(pred["size"]).clamp(0.2, 20.0)     # [B, 3, N]
    sin = pred["yaw"][:, 0, :]
    cos = pred["yaw"][:, 1, :]
    yaw = torch.atan2(sin, cos)                          # [B, N]
    centros = pc1_xyz + offset                           # [B, 3, N]

    B, _, N = pc1_xyz.shape
    salida = []
    for b in range(B):
        sc = conf[b].detach().cpu().numpy()
        cand = np.where(sc > umbral)[0]
        if len(cand) == 0:
            salida.append((np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)))
            continue
        cen = centros[b].permute(1, 0).detach().cpu().numpy()[cand]     # [K,3]
        sz = size[b].permute(1, 0).detach().cpu().numpy()[cand]         # [K,3]
        yw = yaw[b].detach().cpu().numpy()[cand]                        # [K]
        boxes = np.concatenate([cen, sz, yw[:, None]], axis=1).astype(np.float32)
        scores = sc[cand].astype(np.float32)
        keep = _nms_bev(boxes, scores, iou_thr=nms_iou)[:max_cajas]
        salida.append((boxes[keep], scores[keep]))
    return salida
