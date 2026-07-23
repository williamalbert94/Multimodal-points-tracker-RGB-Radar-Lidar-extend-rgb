"""Correlación temporal entre dos frames (cost volume KNN).

Por qué existe este módulo
--------------------------
El radar mide velocidad SOLO a lo largo del rayo sensor→punto. Un peatón que
cruza perpendicular tiene `v_r ≈ 0` y es numéricamente idéntico a un poste.
Medido con el modelo actual, el recall por nivel de Doppler es:

    |v_r_comp| > 2.00  ->  recall 0.95   (lo ve casi siempre)
    |v_r_comp| < 0.10  ->  recall 0.13   (está ciego)

O sea que el modelo, con un solo frame, es en la práctica un detector de Doppler.
La información para resolver el resto NO está en el frame: hay que comparar dos.

Cómo se compara
---------------
El dataset entrega la nube del frame t+1 ya compensada por el ego-movimiento
(`raw_pc0_comp`), es decir, expresada en el sistema de referencia del frame t.
Con eso, si un punto es ESTÁTICO debería caer justo encima de donde estaba en t;
si se MUEVE, queda desplazado. Ese desplazamiento es la señal que falta, y no
depende del Doppler.

El correlador no calcula ese desplazamiento a mano: para cada punto del frame
actual busca sus K vecinos más cercanos en el frame anterior y aprende a resumir
la relación entre ambos (posiciones relativas + features de cada lado). Es el
mismo patrón de RaTrack.
"""
import torch
import torch.nn as nn


def knn_gather(pc2_xyz, pc1_xyz, feature2, K):
    """Para cada punto de `pc1_xyz`, junta sus K vecinos más cercanos de `pc2_xyz`.

    Args:
        pc2_xyz:  [B, N2, 3] puntos del frame anterior.
        pc1_xyz:  [B, N1, 3] puntos del frame actual (compensados).
        feature2: [B, N2, C] features del frame anterior.
        K: cuántos vecinos traer.

    Returns:
        neighbor_xyz:  [B, N1, K, 3] posiciones de los vecinos.
        neighbor_feat: [B, N1, K, C] sus features.
    """
    B, N1, _ = pc1_xyz.shape
    C = feature2.shape[-1]

    dists = torch.cdist(pc1_xyz, pc2_xyz)                  # [B, N1, N2]
    _, idx = dists.topk(K, dim=-1, largest=False)          # [B, N1, K]

    idx_xyz = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    idx_feat = idx.unsqueeze(-1).expand(-1, -1, -1, C)

    neighbor_xyz = pc2_xyz.unsqueeze(1).expand(-1, N1, -1, -1).gather(2, idx_xyz)
    neighbor_feat = feature2.unsqueeze(1).expand(-1, N1, -1, -1).gather(2, idx_feat)
    return neighbor_xyz, neighbor_feat


class WeightNet(nn.Module):
    """Convierte desplazamientos relativos (dx, dy, dz) en pesos de agregación.

    En vez de promediar a los K vecinos por igual, la red aprende cuánto pesa
    cada uno según DÓNDE está respecto al punto. Así puede aprender, por ejemplo,
    que un vecino a 5 cm significa algo muy distinto que uno a 3 m.
    """

    def __init__(self, in_dim=3, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 8, 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_dim, 1),
        )

    def forward(self, xyz):
        """xyz: [B, 3, K, N] -> [B, out_dim, K, N]"""
        return self.net(xyz)


class FeatureCorrelator(nn.Module):
    """Cost volume KNN en dos etapas, entre el frame actual y el anterior.

    Etapa 1 (punto -> vecindario del otro frame):
        Para cada punto del frame actual junta sus K vecinos del frame anterior y
        concatena [feature_actual | feature_vecino | desplazamiento]. Un MLP lo
        procesa y se suma con los pesos que aprende WeightNet.

    Etapa 2 (vecindario -> vecindario, dentro del frame actual):
        Repite la agregación pero entre puntos del mismo frame, para que la
        respuesta de un punto tenga en cuenta la de sus vecinos. Esto suaviza el
        ruido: los puntos de un mismo objeto terminan con respuestas parecidas.

    Args:
        nsample: cuántos vecinos (K).
        in_channel: C1 + C2 + 3 (features de cada frame más el desplazamiento).
        mlp: canales de salida de cada capa del MLP.
    """

    def __init__(self, nsample, in_channel, mlp):
        super().__init__()
        self.nsample = nsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.mlp_convs = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            last_ch = out_ch

        self.weightnet1 = WeightNet(3, last_ch)            # pesos entre frames
        self.weightnet2 = WeightNet(3, last_ch)            # pesos dentro del frame

    def forward(self, pc1, pc2, feature1, feature2):
        """
        Args:
            pc1: [B, 3, N1] puntos del frame actual (idealmente COMPENSADOS por
                 ego-movimiento, para que "moverse" signifique movimiento real).
            pc2: [B, 3, N2] puntos del frame anterior.
            feature1: [B, C1, N1] features del frame actual.
            feature2: [B, C2, N2] features del frame anterior.

        Returns:
            [B, mlp[-1], N1] una feature de correlación por punto actual.
        """
        K = self.nsample
        pc1_t = pc1.permute(0, 2, 1)                       # [B, N1, 3]
        pc2_t = pc2.permute(0, 2, 1)                       # [B, N2, 3]
        feat1_t = feature1.permute(0, 2, 1)                # [B, N1, C1]
        feat2_t = feature2.permute(0, 2, 1)                # [B, N2, C2]

        # ── Etapa 1: cada punto mira el frame anterior ───────────────────────
        nbr_xyz, nbr_feat2 = knn_gather(pc2_t, pc1_t, feat2_t, K)
        # `direction` es el desplazamiento respecto al frame anterior: es
        # justamente lo que distingue a un punto que se movió de uno quieto.
        direction = nbr_xyz - pc1_t.unsqueeze(2)           # [B, N1, K, 3]
        feat1_exp = feat1_t.unsqueeze(2).expand(-1, -1, K, -1)

        x = torch.cat([feat1_exp, nbr_feat2, direction], dim=-1)
        x = x.permute(0, 3, 2, 1)                          # [B, C1+C2+3, K, N1]
        for conv in self.mlp_convs:
            x = self.relu(conv(x))

        w1 = self.weightnet1(direction.permute(0, 3, 2, 1))
        x = (w1 * x).sum(dim=2)                            # [B, last_ch, N1]

        # ── Etapa 2: suavizado entre vecinos del propio frame ────────────────
        nbr_xyz2, nbr_feat_self = knn_gather(pc1_t, pc1_t, x.permute(0, 2, 1), K)
        direction2 = nbr_xyz2 - pc1_t.unsqueeze(2)
        w2 = self.weightnet2(direction2.permute(0, 3, 2, 1))
        x2 = nbr_feat_self.permute(0, 3, 2, 1)
        return (w2 * x2).sum(dim=2)                        # [B, last_ch, N1]
