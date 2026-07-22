"""Extractor de características por punto para la nube radar.

Es el mismo extractor que mejor funcionó en el repo anterior
(Supervised-Multimodal-Tracking): una PointNet++ (set abstraction + feature
propagation) a la que le agregamos una rama de estadísticas globales. La idea es
que cada punto termine con un vector de 128 canales que mezcla:

* lo *local*  -> qué hay en el vecindario del punto (PointNet++), y
* lo *global* -> resumen de toda la nube (media, desviación, min, max, rango).

Las operaciones de vecindario (furthest point sampling, ball query, agrupación)
vienen de `external/lib` (pointnet2), que se compila dentro del Docker con
`python setup.py install`. Por eso este módulo solo corre en el contenedor.

Formato de entrada esperado por PointNet++:
    pc       : [B, N, 3]   coordenadas del punto (radar frame)
    features : [B, N, C]   atributos del punto (acá C=2: RCS y velocidad radial)
"""
import torch
import torch.nn as nn

from external.lib import pointnet2_utils as pointutils            # noqa: F401  (dep compilada)
from external.lib.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG


class GlobalStatisticsModule(nn.Module):
    """Resume toda la nube en 5 números y los reparte a cada punto.

    Recibe las características de los puntos y calcula, sobre la dimensión
    espacial, cinco estadísticas globales: media, desviación estándar, máximo,
    mínimo y rango (max - min). Sirve para darle a cada punto una noción de "cómo
    es la escena completa", no solo su vecindario.

    Entrada:  x    [B, C, N]
    Salida:   stats[B, 5, N]  (las mismas 5 cifras copiadas en todos los puntos)
    """

    def forward(self, x):
        num_points = x.shape[2]

        # Estadísticas por canal a lo largo de los N puntos.
        mean = x.mean(dim=2, keepdim=True)              # [B, C, 1]
        std = x.std(dim=2, keepdim=True)                # [B, C, 1]
        max_val = x.max(dim=2, keepdim=True)[0]         # [B, C, 1]
        min_val = x.min(dim=2, keepdim=True)[0]         # [B, C, 1]
        range_val = max_val - min_val                   # [B, C, 1]

        # Promediamos entre canales para quedar con un solo valor por estadística.
        stats = torch.cat([
            mean.mean(dim=1, keepdim=True),
            std.mean(dim=1, keepdim=True),
            max_val.mean(dim=1, keepdim=True),
            min_val.mean(dim=1, keepdim=True),
            range_val.mean(dim=1, keepdim=True),
        ], dim=1)                                       # [B, 5, 1]

        # La misma foto global se le pega a todos los puntos.
        return stats.expand(-1, -1, num_points)         # [B, 5, N]


class PNHead(nn.Module):
    """PointNet++ clásica (sin la rama global). Se deja como alternativa liviana.

    Baja la resolución en tres niveles (sa1, sa2, sa3) para capturar contexto y
    luego la vuelve a subir (fp3, fp2, fp1) para tener una característica por cada
    punto original.

    Args:
        sample_point_num: cuántos puntos conserva cada nivel de muestreo.
        in_channels: número de atributos de entrada por punto (acá 2).
    """

    def __init__(self, sample_point_num, in_channels):
        super().__init__()

        # sa1: mira vecindarios chicos (radios 2 y 4 m), saca 32+32 = 64 canales.
        self.sa1 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[2, 4],
            nsamples=[4, 8],
            mlps=[[3 + in_channels, 16, 16, 32], [3 + in_channels, 16, 16, 32]],
        )
        # sa2: vecindarios medianos (4 y 8 m). Entra 3 + 32.
        self.sa2 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[4, 8],
            nsamples=[8, 16],
            mlps=[[3 + 32, 32, 32], [3 + 32, 32, 64]],
        )
        # sa3: vecindarios grandes (8 y 16 m). Entra 3 + 64.
        self.sa3 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[8, 16],
            nsamples=[16, 32],
            mlps=[[3 + 64, 64, 64], [3 + 64, 64, 64]],
        )

        # Feature propagation: devuelve la información a la resolución original.
        self.fp3 = PointnetFPModule(mlp=[128, 128])
        self.fp2 = PointnetFPModule(mlp=[160, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128])

        # Reducen la dimensión después de cada nivel de muestreo.
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(96, 64)
        self.linear3 = nn.Linear(128, 64)

    def forward(self, pc, features):
        # pc: [B, N, 3]  |  features: [B, N, C]
        l0_xyz = pc.contiguous()
        # PointNet++ quiere las features como [B, C, N], por eso trasponemos.
        l0_points = features.transpose(1, 2).contiguous()

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l3_xyz, l0_points                        # l0_points: [B, 128, N]


class LocalGlobalFusionSimple(nn.Module):
    """PointNet++ + rama global. Es el extractor que dio el mejor mIoU.

    Igual que `PNHead`, pero antes de subir la resolución mezcla las
    características más profundas (`l3_points`) con las estadísticas globales de la
    escena. Esa fusión local-global es la contribución del trabajo.

    Args:
        sample_point_num: puntos que conserva cada nivel de muestreo.
        in_channels: atributos de entrada por punto (acá 2).
    """

    def __init__(self, sample_point_num, in_channels):
        super().__init__()
        self.sample_point_num = sample_point_num

        self.sa1 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[2, 4],
            nsamples=[4, 8],
            mlps=[[3 + in_channels, 16, 16, 32], [3 + in_channels, 16, 16, 32]],
        )
        self.sa2 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[4, 8],
            nsamples=[8, 16],
            mlps=[[3 + 32, 32, 32], [3 + 32, 32, 64]],
        )
        self.sa3 = PointnetSAModuleMSG(
            npoint=sample_point_num,
            radii=[8, 16],
            nsamples=[16, 32],
            mlps=[[3 + 64, 64, 64], [3 + 64, 64, 64]],
        )

        # Rama global + convolución que fusiona los 64 canales locales con las 5
        # estadísticas globales y vuelve a 64.
        self.global_stats = GlobalStatisticsModule()
        self.fusion_conv = nn.Conv1d(64 + 5, 64, 1)

        self.fp3 = PointnetFPModule(mlp=[128, 128])
        self.fp2 = PointnetFPModule(mlp=[160, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128])

        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(96, 64)
        self.linear3 = nn.Linear(128, 64)

    def forward(self, pc, features):
        # pc: [B, N, 3]  |  features: [B, N, C]
        l0_xyz = pc.contiguous()
        l0_points = features.transpose(1, 2).contiguous()

        # 1. Características locales, bajando la resolución en tres niveles.
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # 2. Estadísticas globales del nivel más profundo y fusión local-global.
        global_stats = self.global_stats(l3_points)            # [B, 5, N3]
        l3_fused = torch.cat([l3_points, global_stats], dim=1)  # [B, 69, N3]
        l3_fused = self.fusion_conv(l3_fused)                   # [B, 64, N3]

        # 3. Se sube la resolución con las características ya fusionadas.
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_fused)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l3_xyz, l0_points                        # l0_points: [B, 128, N]
