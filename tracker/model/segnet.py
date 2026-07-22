"""Red de segmentación de puntos móvil/estático.

Para esta primera etapa nos concentramos SOLO en segmentación (no en flow ni en
tracking), así que la red es la ruta mínima que necesita la cabeza de
segmentación supervisada, que es justamente la parte que dio el mejor mIoU:

    pc1 + features1
        └─► extractor (PointNet++ local-global) ─► [B, 128, N]
                └─► fusión local-global ─────────► [B, 256, N]
                        └─► cabeza de segmentación ─► [B, 1, N]  (prob. de "móvil")

No usamos el segundo frame (pc2), ni el cost-volume, ni el decoder de flow: la
cabeza de segmentación solo depende de las características del frame actual, y
mantenerlo así hace el entrenamiento más simple y rápido.
"""
import torch
import torch.nn as nn

from .feature_extractor import LocalGlobalFusionSimple, PNHead


class SupervisedSegmentationHead(nn.Module):
    """Cabeza de segmentación con conexión residual (la que mejor segmentó).

    Toma las características fusionadas de cada punto y devuelve, por punto, la
    probabilidad de que sea "móvil" (pertenece a un objeto en movimiento). La
    conexión residual (skip) ayuda a que la señal no se pierda al profundizar.

    Entrada:  pc1_features [B, 256, N]
    Salida:   seg          [B, 1, N]   con Sigmoid, valores en [0, 1]
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(feature_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2))
        self.skip_proj = nn.Conv1d(feature_dim, 256, 1)     # proyección para el residual
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
        self.final = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 1, 1), nn.Sigmoid())

    def forward(self, pc1_features):
        x = self.conv1(pc1_features)
        skip = self.skip_proj(pc1_features)
        x = self.conv2(x) + skip                            # residual
        x = self.conv3(x)
        return self.final(x)                                # [B, 1, N]


class SegmentationNet(nn.Module):
    """Red completa de segmentación de puntos.

    Junta el extractor con la cabeza de segmentación. La fusión local-global toma
    las 128 características locales de cada punto y les concatena su máximo global
    (otras 128), quedando en 256 antes de la cabeza.

    Args:
        args: configuración; se leen `num_points`, `in_channels` y `extractor`.
    """

    def __init__(self, args):
        super().__init__()
        self.npoints = args.num_points
        in_channels = int(getattr(args, "in_channels", 2))  # RCS y velocidad radial

        extractor = getattr(args, "extractor", "LocalGlobalFusionSimple")
        if extractor == "LocalGlobalFusionSimple":
            self.pn_head = LocalGlobalFusionSimple(args.num_points, in_channels)
        elif extractor == "PNHead":
            self.pn_head = PNHead(args.num_points, in_channels)
        else:
            raise ValueError(f"Extractor desconocido: {extractor!r}")
        print(f"[modelo] extractor = {extractor}  (in_channels={in_channels})")

        # Local(128) + global(128) = 256 canales de entrada a la cabeza.
        self.seg_head = SupervisedSegmentationHead(feature_dim=256)

    def forward(self, pc1, feature1):
        """
        Args:
            pc1:      [B, 3, N]  coordenadas del punto (radar frame)
            feature1: [B, 2, N]  atributos radar (RCS, velocidad radial)

        Returns:
            seg:   [B, 1, N]   probabilidad de "móvil" por punto
            fused: [B, 256, N] features fusionadas por punto (las usa la
                   feature-contrast loss para separar móvil/estático)
        """
        # El extractor espera [B, N, 3] y [B, N, C].
        _, local = self.pn_head(
            pc1.permute(0, 2, 1).contiguous(),
            feature1.permute(0, 2, 1).contiguous(),
        )                                                   # local: [B, 128, N]

        # Fusión local-global: a cada punto le pegamos el máximo global de la nube.
        glob = local.max(dim=-1, keepdim=True)[0].expand_as(local)
        fused = torch.cat([local, glob], dim=1)             # [B, 256, N]

        return self.seg_head(fused), fused                  # [B, 1, N], [B, 256, N]


def build_model(args, logger=None):
    """Crea la red y la manda a GPU. Reporta cuántos parámetros tiene."""
    net = SegmentationNet(args).cuda()
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    msg = f"Modelo de segmentación cargado. Parámetros: {n_params:.2f}M"
    (logger.info if logger else print)(msg)
    return net
