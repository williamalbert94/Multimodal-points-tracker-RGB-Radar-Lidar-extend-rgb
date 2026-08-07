"""Cabecera de detección 3D por punto (anchor-free, estilo CenterPoint).

Se engancha a las features por punto del backbone de segmentación ([B, F, N]) y
predice, para cada punto de radar:

    conf    [B, 1, N]  probabilidad de pertenecer a un objeto móvil (sigmoid)
    offset  [B, 3, N]  vector del punto al CENTRO de su caja (dx, dy, dz)
    size    [B, 3, N]  tamaño en LOG (log_l, log_w, log_h); exp() da (l, w, h)
    yaw     [B, 2, N]  (sin(yaw), cos(yaw))  -> evita la discontinuidad angular

Enfoque por OFFSET (no heatmap de centro): en radar esparso los puntos caen sobre
la superficie del objeto, casi nunca sobre el centro. Cualquier punto DENTRO de la
caja aprende a votar por el centro con su offset. Una sola clase ("móvil").
"""
import math

import torch
import torch.nn as nn

# Tamaño log medio de un objeto móvil (VoD ~ auto/ciclista/peatón mezclados),
# para inicializar la regresión de tamaño en un valor sensato.
MEAN_LOG_SIZE = [math.log(2.0), math.log(1.0), math.log(1.6)]


class PointDetectionHead(nn.Module):
    """Cabecera de detección por punto sobre features del backbone [B, F, N]."""

    def __init__(self, feature_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(feature_dim, hidden, 1)

        def cabeza(out_ch):
            return nn.Sequential(
                nn.Conv1d(hidden, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, out_ch, 1),
            )

        self.conf_head = cabeza(1)
        self.offset_head = cabeza(3)
        self.size_head = cabeza(3)
        self.yaw_head = cabeza(2)

        # Sesgo de confianza a -2 (sigmoid ~ 0.12): evita una avalancha inicial
        # de falsos positivos (práctica de FCOS/CenterNet).
        nn.init.constant_(self.conf_head[-1].bias, -2.0)
        # Tamaño inicializado en el log medio.
        with torch.no_grad():
            self.size_head[-1].bias.copy_(torch.tensor(MEAN_LOG_SIZE))

    def forward(self, feats):
        """feats: [B, F, N] -> dict con conf/offset/size/yaw."""
        x = self.trunk(feats) + self.skip(feats)
        return {
            "conf":   self.conf_head(x),     # [B, 1, N]  (logit)
            "offset": self.offset_head(x),   # [B, 3, N]
            "size":   self.size_head(x),     # [B, 3, N]  (log)
            "yaw":    self.yaw_head(x),      # [B, 2, N]  (sin, cos)
        }
