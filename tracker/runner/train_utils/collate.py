"""Utilidades para armar el batch de segmentación.

Las nubes de puntos tienen tamaño variable (cada frame trae distinta cantidad de
puntos radar), así que no se pueden apilar directamente en un tensor. Por eso:

* `custom_collate_fn` deja cada muestra como está (en listas), sin apilar, y
* `sample_points` muestrea/rellena cada nube a un número fijo de puntos para
  poder apilarlas en un tensor y meterlas al modelo.

Recordatorio del emparejamiento que entrega el dataset (ver datagen_vod):
la nube `raw_pc0` va con las cajas `lbl1` y las transformaciones `transforms1`.
Ese es el frame que segmentamos.
"""
import numpy as np
import torch


def custom_collate_fn(batch):
    """Junta las muestras del batch sin apilar (todo queda como listas/tuplas).

    Cada muestra es la tupla larga que devuelve `TrackingDataVOD.__getitem__`.
    Trasponemos para agrupar por campo y dejamos cada campo como una tupla de B
    elementos. El apilado real (a tensor) se hace después con `sample_points`.
    """
    transposed = list(zip(*batch))
    return tuple(transposed)


def sample_points(pc_list, num_points):
    """Lleva cada nube a exactamente `num_points` puntos.

    Si la nube tiene más puntos, se muestrea sin reemplazo; si tiene menos, se
    repite con reemplazo (oversampling). Así todas las nubes del batch quedan del
    mismo tamaño y se pueden apilar.

    Args:
        pc_list: lista/tupla de arrays numpy [N_i, C] (C puede variar por campo).
        num_points: número de puntos objetivo.

    Returns:
        tensor [B, num_points, C] float32.
    """
    batch_size = len(pc_list)
    num_channels = pc_list[0].shape[1]
    out = np.zeros((batch_size, num_points, num_channels), dtype=np.float32)

    for i, pc in enumerate(pc_list):
        n = pc.shape[0]
        if n >= num_points:
            idx = np.random.choice(n, num_points, replace=False)
        else:
            idx = np.random.choice(n, num_points, replace=True)
        out[i] = pc[idx]

    return torch.from_numpy(out).float()
