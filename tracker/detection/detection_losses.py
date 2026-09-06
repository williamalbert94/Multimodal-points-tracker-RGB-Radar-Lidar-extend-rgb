"""Asignación de targets y pérdida para la cabecera de detección por punto.

Asignación (por muestra):
  - Un punto de radar es POSITIVO si cae dentro de alguna caja GT móvil (con una
    pequeña inflación para tolerar el error de medición del radar). Si cae en
    varias, se le asigna la de centro más cercano.
  - Target del positivo: offset = centro_caja - punto ; tamaño en log ; sin/cos yaw.

Pérdida:
  L = BCE(conf) + w_off·SmoothL1(offset) + w_sz·SmoothL1(log_size) + w_yaw·SmoothL1(sincos)
  (los términos de caja solo sobre los positivos).
"""
import torch
import torch.nn.functional as F

INFLADO = 0.3          # metros de inflación para el test de "dentro de la caja"


@torch.no_grad()
def asignar_targets(pc1_xyz, gt_boxes_list):
    """Construye los targets por punto a partir de las cajas GT.

    Args:
        pc1_xyz: [B, 3, N] coordenadas de radar (muestreadas).
        gt_boxes_list: lista de B tensores [M_b, 7] = (x,y,z,l,w,h,yaw) en radar.

    Returns:
        dict con:
          pos    [B, N]     máscara de positivos (float 0/1)
          offset [B, 3, N]  centro - punto
          size   [B, 3, N]  log(l,w,h)
          yaw    [B, 2, N]  (sin, cos)
    """
    B, _, N = pc1_xyz.shape
    dev = pc1_xyz.device
    pos = torch.zeros(B, N, device=dev)
    offset = torch.zeros(B, 3, N, device=dev)
    size = torch.zeros(B, 3, N, device=dev)
    yaw = torch.zeros(B, 2, N, device=dev)

    for b in range(B):
        boxes = gt_boxes_list[b]
        if boxes is None or len(boxes) == 0:
            continue
        boxes = boxes.to(dev).float()                    # [M, 7]
        M = boxes.shape[0]
        pts = pc1_xyz[b].permute(1, 0)                   # [N, 3]

        centros = boxes[:, :3]                           # [M, 3]
        lwh = boxes[:, 3:6]                              # [M, 3]
        yaws = boxes[:, 6]                              # [M]

        rel = pts.unsqueeze(0) - centros.unsqueeze(1)    # [M, N, 3]
        cos = torch.cos(-yaws).view(M, 1)
        sin = torch.sin(-yaws).view(M, 1)
        rx = rel[:, :, 0] * cos - rel[:, :, 1] * sin     # [M, N] al frame de la caja
        ry = rel[:, :, 0] * sin + rel[:, :, 1] * cos
        rz = rel[:, :, 2]
        half = (lwh * 0.5 + INFLADO)                     # [M, 3]
        dentro = ((rx.abs() < half[:, 0:1]) &
                  (ry.abs() < half[:, 1:2]) &
                  (rz.abs() < half[:, 2:3]))             # [M, N]

        dist = torch.norm(rel, dim=2)                    # [M, N] al centro
        dist_masked = torch.where(dentro, dist, torch.full_like(dist, 1e9))
        best = dist_masked.argmin(dim=0)                 # [N] caja más cercana
        es_pos = dentro.any(dim=0)                       # [N]

        if es_pos.any():
            idxb = best[es_pos]                          # caja asignada por punto positivo
            pos[b, es_pos] = 1.0
            offset[b, :, es_pos] = (centros[idxb] - pts[es_pos]).permute(1, 0)
            size[b, :, es_pos] = torch.log(lwh[idxb].clamp(min=0.1)).permute(1, 0)
            yaw[b, 0, es_pos] = torch.sin(yaws[idxb])
            yaw[b, 1, es_pos] = torch.cos(yaws[idxb])

    return {"pos": pos, "offset": offset, "size": size, "yaw": yaw}


def detection_loss(pred, tgt, w_off=1.0, w_sz=1.0, w_yaw=1.0):
    """Pérdida de detección. pred y tgt: dicts de asignar_targets / la cabecera."""
    pos = tgt["pos"]                                     # [B, N]
    conf_logit = pred["conf"].squeeze(1)                # [B, N]
    # BCE de confianza (todos los puntos); positivos pesan por el desbalance.
    n_pos = pos.sum().clamp(min=1.0)
    peso_pos = ((pos.numel() - n_pos) / n_pos).clamp(1.0, 50.0)
    l_conf = F.binary_cross_entropy_with_logits(
        conf_logit, pos, pos_weight=peso_pos)

    m = pos.unsqueeze(1)                                 # [B, 1, N]
    def sl1(p, t):
        return (F.smooth_l1_loss(p * m, t * m, reduction="sum") / n_pos)

    l_off = sl1(pred["offset"], tgt["offset"])
    l_sz = sl1(pred["size"], tgt["size"])
    l_yaw = sl1(pred["yaw"], tgt["yaw"])

    total = l_conf + w_off * l_off + w_sz * l_sz + w_yaw * l_yaw
    return total, {"conf": l_conf.item(), "off": l_off.item(),
                   "size": l_sz.item(), "yaw": l_yaw.item()}
