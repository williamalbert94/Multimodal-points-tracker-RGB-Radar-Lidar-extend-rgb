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

from .correlator import FeatureCorrelator
from .feature_extractor import LocalGlobalFusionSimple, LocalGlobalFusionWide, PNHead


class SupervisedSegmentationHead(nn.Module):
    """Cabeza de segmentación con conexión residual (la que mejor segmentó).

    Toma las características fusionadas de cada punto y devuelve, por punto, la
    probabilidad de que sea "móvil" (pertenece a un objeto en movimiento). La
    conexión residual (skip) ayuda a que la señal no se pierda al profundizar.

    Entrada:  pc1_features [B, 256, N]
    Salida:   seg          [B, 1, N]   con Sigmoid, valores en [0, 1]
    """

    def __init__(self, feature_dim: int = 256, hidden: int = 256):
        super().__init__()
        # `hidden` es el ancho interno de la cabeza. Con el extractor simple vale
        # 256 (comportamiento original); con el extractor ancho se sube para que
        # la mayor dimensión de entrada no quede embudada de entrada.
        self.conv1 = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.skip_proj = nn.Conv1d(feature_dim, hidden, 1)  # proyección para el residual
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
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
        elif extractor == "LocalGlobalFusionWide":
            self.pn_head = LocalGlobalFusionWide(args.num_points, in_channels)
        elif extractor == "PNHead":
            self.pn_head = PNHead(args.num_points, in_channels)
        else:
            raise ValueError(f"Extractor desconocido: {extractor!r}")
        print(f"[modelo] extractor = {extractor}  (in_channels={in_channels})")

        # ── Normalizacion de los canales de entrada ──────────────────────────
        # Los canales vienen en escalas muy distintas: medido sobre el split de
        # train, el RCS tiene std 14.0 mientras que v_r_comp tiene 2.07. Al
        # entrar crudos a la primera convolucion, el RCS domina el gradiente y
        # v_r_comp —que es el canal que mejor separa movil de estatico— queda
        # ahogado. Acá se estandariza cada canal (z-score) con las estadisticas
        # del dataset.
        #
        # Van como buffers (no son parametros entrenables) para que se guarden
        # en el checkpoint y la inferencia use exactamente las mismas cifras.
        # Las coordenadas NO se tocan: PointNet++ usa radios en metros.
        media = getattr(args, "feat_mean", None)
        desv = getattr(args, "feat_std", None)
        if media and desv:
            m = torch.tensor(media[:in_channels], dtype=torch.float32).view(1, -1, 1)
            s = torch.tensor(desv[:in_channels], dtype=torch.float32).view(1, -1, 1)
            s = s.clamp(min=1e-6)                       # por si algun canal es constante
            self.normaliza = True
            print(f"[modelo] normalizando features: media={media[:in_channels]} "
                  f"std={desv[:in_channels]}")
        else:
            m = torch.zeros(1, in_channels, 1)
            s = torch.ones(1, in_channels, 1)
            self.normaliza = False
            print("[modelo] AVISO: sin feat_mean/feat_std en el config -> features crudas")
        self.register_buffer("feat_mean", m)
        self.register_buffer("feat_std", s)

        # ── Rama temporal (opcional) ─────────────────────────────────────────
        # Con un solo frame la red termina siendo un detector de Doppler: recall
        # 0.95 cuando |v_r_comp| > 2, pero 0.13 cuando < 0.1 (objetos que cruzan
        # perpendicular al sensor). Esa información no está en el frame actual.
        # El correlador la saca comparando contra el frame anterior.
        # Dimensiones derivadas del extractor. `_extraer` concatena las features
        # locales con su máximo global, así que produce 2*out_dim canales por
        # punto. Todo lo de abajo se escala a partir de ahí para que el extractor
        # simple (out_dim=128 -> feat 256) quede idéntico al original y el ancho
        # (out_dim=256 -> feat 512) escale correlador y cabeza en consecuencia.
        out_dim = int(getattr(self.pn_head, "out_dim", 128))
        feat_dim = 2 * out_dim                              # salida de _extraer

        self.use_temporal = bool(getattr(args, "use_temporal", False))
        # Multi-frame: además de t-1, correlacionar también contra t-2 (segunda
        # rama, línea base más ancha para objetos lentos). Solo aplica si la rama
        # temporal está activa.
        self.use_multiframe = self.use_temporal and (
            bool(getattr(args, "multiframe", False)) or bool(getattr(args, "lidar_temporal", False)))
        if self.use_temporal:
            # in_channel = features del frame actual + del anterior + 3 del
            # desplazamiento = feat_dim + feat_dim + 3.
            K = int(getattr(args, "temporal_knn", 16))
            self.correlator = FeatureCorrelator(
                nsample=K, in_channel=feat_dim * 2 + 3,
                mlp=[feat_dim, feat_dim, feat_dim],
            )
            if self.use_multiframe:
                # Segundo correlador, propio, para el frame t-2.
                self.correlator2 = FeatureCorrelator(
                    nsample=K, in_channel=feat_dim * 2 + 3,
                    mlp=[feat_dim, feat_dim, feat_dim],
                )
                dim_cabeza = 3 * feat_dim                   # propias + corr(t-1) + corr(t-2)
                print(f"[modelo] rama temporal MULTI-FRAME ACTIVA (t-1 y t-2, K={K})")
            else:
                dim_cabeza = 2 * feat_dim                   # propias + correlación
                print(f"[modelo] rama temporal ACTIVA (K={K})")
        else:
            dim_cabeza = feat_dim
            print("[modelo] rama temporal apagada (solo un frame)")

        # Ancho interno de la cabeza: 256 para el extractor simple (original),
        # más grande cuando la entrada es más ancha, sin bajar de 256.
        head_hidden = max(256, feat_dim)
        self.seg_head = SupervisedSegmentationHead(feature_dim=dim_cabeza, hidden=head_hidden)

    def _extraer(self, pc, feature):
        """Saca las features fusionadas local-global de una nube. [B, 256, N]"""
        feature = (feature - self.feat_mean) / self.feat_std
        _, local = self.pn_head(
            pc.permute(0, 2, 1).contiguous(),
            feature.permute(0, 2, 1).contiguous(),
        )                                                   # [B, 128, N]
        # A cada punto le pegamos el máximo global de su nube.
        glob = local.max(dim=-1, keepdim=True)[0].expand_as(local)
        return torch.cat([local, glob], dim=1)              # [B, 256, N]

    def forward(self, pc1, feature1, pc2=None, feature2=None, pc1_comp=None,
                pc3=None, feature3=None, pc1_comp2=None):
        """
        Args:
            pc1:      [B, 3, N]  puntos del frame actual (radar frame).
            feature1: [B, C, N]  atributos radar del frame actual.
            pc2:      [B, 3, M]  puntos del frame anterior t-1 (solo si hay rama temporal).
            feature2: [B, C, M]  sus atributos.
            pc1_comp: [B, 3, N]  el frame actual compensado por ego-movimiento al
                      sistema de referencia de t-1. Es lo que hace que "estar
                      desplazado" signifique movimiento real y no que el carro avanzó.
            pc3, feature3, pc1_comp2: análogos para el frame t-2 (solo multi-frame).

        Returns:
            seg:   [B, 1, N]  probabilidad de "móvil" por punto.
            feats: [B, D, N]  features por punto (las usa la feature-contrast loss).
        """
        fused = self._extraer(pc1, feature1)                # [B, 2*out_dim, N]

        if not self.use_temporal:
            return self.seg_head(fused), fused

        if pc2 is None or feature2 is None:
            raise ValueError("La rama temporal necesita pc2 y feature2.")

        fused2 = self._extraer(pc2, feature2)               # [B, 2*out_dim, M]

        # La geometría de la correlación va en coordenadas compensadas: si no, el
        # avance del propio carro haría ver a TODO el mundo como si se moviera.
        geom1 = pc1_comp if pc1_comp is not None else pc1
        cor = self.correlator(geom1, pc2, fused, fused2)    # correlación contra t-1

        if self.use_multiframe:
            if pc3 is None or feature3 is None:
                raise ValueError("Multi-frame necesita pc3 y feature3 (t-2).")
            fused3 = self._extraer(pc3, feature3)
            geom1b = pc1_comp2 if pc1_comp2 is not None else pc1
            cor2 = self.correlator2(geom1b, pc3, fused, fused3)  # correlación contra t-2
            feats = torch.cat([fused, cor, cor2], dim=1)    # [B, 3*feat_dim, N]
        else:
            feats = torch.cat([fused, cor], dim=1)          # [B, 2*feat_dim, N]

        return self.seg_head(feats), feats


def build_model(args, logger=None):
    """Crea la red y la manda a GPU. Reporta cuántos parámetros tiene."""
    net = SegmentationNet(args).cuda()
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    msg = f"Modelo de segmentación cargado. Parámetros: {n_params:.2f}M"
    (logger.info if logger else print)(msg)
    return net
