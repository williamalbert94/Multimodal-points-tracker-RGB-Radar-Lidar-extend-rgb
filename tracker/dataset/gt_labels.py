"""Etiquetas GT por punto para la segmentación móvil/estático.

El dataset entrega las cajas de los objetos móviles en coordenadas de cámara
(formato KITTI: h, w, l, x, y, z, ry). Para saber qué puntos del radar están
adentro de cada caja necesitamos dos pasos:

  1. Llevar la caja al frame del radar (que es donde viven los puntos), usando
     las transformaciones del frame. Esto es lo que hacía `get_bbx_param` en el
     repo anterior y reproduce el mismo resultado que dio buen mIoU.
  2. Marcar como "móvil" (etiqueta 1) todo punto que caiga dentro de alguna caja
     orientada; el resto queda "estático" (etiqueta 0).

El punto-dentro-de-caja lo resuelve Open3D con `OrientedBoundingBox`, igual que
en el trabajo original.
"""
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R


def get_bbx_param(obj_info, transforms, sensor="radar", margen=0.0):
    """Construye la caja orientada de un objeto en el frame del sensor.

    Réplica de la función que se usó en el repo anterior. Toma los parámetros
    KITTI en cámara y devuelve una `OrientedBoundingBox` de Open3D ya rotada y
    trasladada al frame pedido.

    Sobre el margen
    ---------------
    Las cajas vienen anotadas sobre LiDAR, que es preciso, así que quedan muy
    pegadas al objeto (un peatón mide ~0.6 m de ancho). El radar, en cambio, tiene
    error de medición de varios centímetros, así que sus returns se salen de la
    caja y quedan mal etiquetados como estáticos. Medido sobre 200 frames, con
    margen 0 el 22% de los objetos móviles NO captura ni un punto de radar.

    `margen` agrega esos centímetros a cada lado de la caja. Valores medidos:

        margen 0.00 m -> 2260 puntos móviles, 78.1% de objetos capturados
        margen 0.15 m -> 2884 puntos (1.28x), 84.2% de objetos
        margen 0.25 m -> 3184 puntos (1.41x), 85.4% de objetos
        margen 0.60 m -> 3963 puntos (1.75x), pero ya se contamina con estáticos

    Por defecto queda en 0.0 para no cambiar el comportamiento de RaTrack (y que
    los números sigan siendo comparables con los suyos).

    Args:
        obj_info: lista [h, w, l, x, y, z, ry] de la caja (frame cámara).
        transforms: `FrameTransformMatrix` del frame (trae las matrices radar<-cámara).
        sensor: 'radar' o 'lidar'.
        margen: metros que se le suman a cada lado de la caja.

    Returns:
        open3d.geometry.OrientedBoundingBox en el frame del sensor.
    """
    h, w, l, x, y, z, ry = obj_info

    # Centro: se lleva el punto [x, y, z] de cámara al frame del sensor.
    if sensor == "lidar":
        center = (transforms.t_lidar_camera @ np.array([x, y, z, 1.0]))[:3]
    else:  # radar
        center = (transforms.t_radar_camera @ np.array([x, y, z, 1.0]))[:3]

    # Tamaño en el orden que espera Open3D: largo, ancho, alto.
    # El margen se suma dos veces porque se reparte a lado y lado.
    extent = np.array([l, w, h]) + 2.0 * float(margen)

    # Rotación: yaw KITTI -> matriz, y luego al frame del sensor.
    rot = R.from_euler("XYZ", [0, 0, -(ry + np.pi / 2)]).as_matrix()
    if sensor == "radar":
        rot = transforms.t_radar_lidar[:3, :3] @ rot
    # (para lidar la rotación queda tal cual)

    return o3d.geometry.OrientedBoundingBox(center.reshape(3, 1), rot, extent.reshape(3, 1))


def moving_point_labels(labels, pc, transforms, margen=0.0, min_puntos=0,
                        devolver_ignorar=False):
    """Marca qué puntos del radar caen dentro de las cajas móviles.

    Sobre `min_puntos`
    ------------------
    Hay objetos móviles que dejan uno o dos returns sueltos de radar. Medido
    sobre 1009 objetos, los de 1-2 puntos son el 22.3% de los objetos pero
    aportan apenas el 6.9% de los puntos móviles: son casi imposibles de
    segmentar y meten mucho ruido en la pérdida.

    Cuando `min_puntos > 0`, esos objetos NO se marcan como estáticos (eso sería
    enseñarle al modelo algo falso), sino que se devuelven aparte en una máscara
    de "ignorar", para poder sacarlos tanto de la pérdida como de la métrica.

    Args:
        labels: dict {id_objeto: Object_3D} con las cajas móviles del frame
            (el dataset ya filtró para dejar solo objetos en movimiento).
        pc: tensor [1, 3, N] con los puntos del frame (radar frame, en GPU o CPU).
        transforms: `FrameTransformMatrix` del mismo frame.
        margen: metros extra a cada lado de la caja (ver `get_bbx_param`).
        min_puntos: objetos con menos de estos puntos van a la máscara de ignorar.
        devolver_ignorar: si es True devuelve también esa máscara.

    Returns:
        cls: tensor bool [N]. True = punto móvil.
        (y si `devolver_ignorar`) ignorar: tensor bool [N] con los puntos que no
        se deben ni premiar ni castigar.
    """
    num_points = pc.shape[2]
    device = pc.device
    cls = torch.zeros(num_points, dtype=torch.bool, device=device)
    ignorar = torch.zeros(num_points, dtype=torch.bool, device=device)

    if not labels:
        return (cls, ignorar) if devolver_ignorar else cls

    # Open3D necesita los puntos en CPU como [N, 3].
    pts_np = pc[0].detach().cpu().numpy().T
    cloud_pts = o3d.utility.Vector3dVector(pts_np)

    for label in labels.values():
        box = get_bbx_param(
            [label.h, label.w, label.l, label.x, label.y, label.z, label.ry],
            transforms, "radar", margen=margen,
        )
        inside = box.get_point_indices_within_bounding_box(cloud_pts)
        if len(inside) == 0:
            continue
        if 0 < len(inside) < min_puntos:
            ignorar[inside] = True          # muy pocos puntos: ni sí ni no
        else:
            cls[inside] = True

    # Un punto que también pertenece a un objeto bien poblado sí cuenta como móvil.
    ignorar &= ~cls

    return (cls, ignorar) if devolver_ignorar else cls


def moving_point_labels_batch(labels_batch, pc_batch, transforms_batch, margen=0.0,
                              min_puntos=0, devolver_ignorar=False):
    """Versión por batch de `moving_point_labels`.

    Args:
        labels_batch: lista de dicts de cajas, uno por elemento del batch.
        pc_batch: tensor [B, 3, N] con los puntos de cada frame.
        transforms_batch: lista de `FrameTransformMatrix`, uno por elemento.
        margen: metros extra a cada lado de la caja (ver `get_bbx_param`).
        min_puntos: umbral para la máscara de ignorar (ver `moving_point_labels`).
        devolver_ignorar: si es True devuelve también la máscara de ignorar.

    Returns:
        cls_batch: tensor [B, N] (bool) con la etiqueta por punto de cada frame,
        y opcionalmente ignorar_batch [B, N].
    """
    batch_size = pc_batch.shape[0]
    cls_out, ign_out = [], []
    for b in range(batch_size):
        r = moving_point_labels(labels_batch[b], pc_batch[b:b + 1],
                                transforms_batch[b], margen=margen,
                                min_puntos=min_puntos, devolver_ignorar=True)
        cls_out.append(r[0])
        ign_out.append(r[1])
    cls_batch = torch.stack(cls_out, dim=0)                 # [B, N]
    if devolver_ignorar:
        return cls_batch, torch.stack(ign_out, dim=0)
    return cls_batch
