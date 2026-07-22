"""Figuras para comparar la segmentación predicha contra el ground truth.

Se generan dos variantes por frame:

* `figura_comparacion` -> 3 paneles:  GT (BEV) | proyección RGB | Predicción (BEV)
* `figura_bev`         -> 2 paneles:  GT (BEV) | Predicción (BEV)   (sin la imagen)

En los BEV se mira la escena desde arriba: el eje vertical es X (hacia adelante
del carro) y el horizontal es Y (izquierda/derecha), igual que en las figuras del
trabajo anterior. Código de colores y marcadores:

    rojo (punto chico)     -> estático
    verde (estrella)       -> móvil segun el ground truth
    cian (bolita)          -> móvil segun la predicción
"""
import os

import matplotlib
matplotlib.use("Agg")                                   # sin pantalla (corre en Docker)
import matplotlib.pyplot as plt
import numpy as np

from external.vod.frame.transformations import (
    canvas_crop,
    homogeneous_transformation,
    project_3d_to_2d,
)

# Colores y marcadores: mismo criterio en todos los paneles.
#   estático  -> rojo, punto chico (en gris no se veía contra el fondo)
#   GT móvil  -> estrella verde
#   predicción-> círculo cian
COLOR_ESTATICO = "#E53935"      # rojo
COLOR_MOVIL_GT = "#00E676"      # verde -> lo que de verdad se mueve
COLOR_MOVIL_PRED = "#00B0FF"    # cian  -> lo que la red dice que se mueve

MARCA_GT = "*"                  # estrella
MARCA_PRED = "o"                # bolita

LIM_X = (0, 75)                 # metros hacia adelante
LIM_Y = (-30, 30)               # metros a los lados


def _panel_bev(ax, puntos, mascara, titulo, color_movil, marca):
    """Dibuja un panel en vista de pájaro (BEV) coloreando móvil vs estático.

    Los estáticos van en rojo chiquito y los móviles resaltados con el marcador
    que se pida (estrella para el GT, bolita para la predicción).
    """
    ax.set_facecolor("#37474F")                          # fondo oscuro: resalta todo

    estaticos = ~mascara
    # Ojo: en BEV el eje horizontal es Y y el vertical es X.
    ax.scatter(puntos[estaticos, 1], puntos[estaticos, 0],
               s=9, c=COLOR_ESTATICO, alpha=0.85, linewidths=0)
    tam = 150 if marca == MARCA_GT else 60
    ax.scatter(puntos[mascara, 1], puntos[mascara, 0],
               s=tam, c=color_movil, marker=marca,
               edgecolors="black", linewidths=0.6)

    ax.set_xlim(LIM_Y)
    ax.set_ylim(LIM_X)
    ax.set_xlabel("Y (m) - Izquierda/Derecha", fontsize=9)
    ax.set_ylabel("X (m) - Adelante", fontsize=9)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15)


def _panel_proyeccion(ax, imagen, puntos, mascara_gt, mascara_pred, transforms):
    """Dibuja la imagen de la cámara con los puntos del radar encima.

    Se proyectan los puntos 3D del radar al plano de la imagen. Cada punto se
    pinta según lo que predijo la red, y los que el GT dice que son móviles se
    marcan además con un aro verde, para ver de una los aciertos y los errores.
    """
    if imagen is None:
        ax.text(0.5, 0.5, "sin imagen", ha="center", va="center")
        ax.axis("off")
        return

    ax.imshow(imagen)

    # Radar -> cámara -> plano imagen.
    hom = np.hstack([puntos[:, :3], np.ones((len(puntos), 1), dtype=np.float32)])
    en_camara = homogeneous_transformation(hom, transforms.t_camera_radar)
    profundidad = en_camara[:, 2]
    uvs = project_3d_to_2d(en_camara, transforms.camera_projection_matrix)

    # Nos quedamos solo con los que caen dentro de la imagen y están adelante.
    visibles = canvas_crop(uvs, imagen.shape, profundidad)

    vis_est = visibles & ~mascara_pred & ~mascara_gt
    ax.scatter(uvs[vis_est, 0], uvs[vis_est, 1], s=10,
               c=COLOR_ESTATICO, alpha=0.7, linewidths=0)

    # GT como estrella verde, predicción como bolita cian.
    vis_gt = visibles & mascara_gt
    ax.scatter(uvs[vis_gt, 0], uvs[vis_gt, 1], s=150, c=COLOR_MOVIL_GT,
               marker=MARCA_GT, edgecolors="black", linewidths=0.6)
    vis_mov = visibles & mascara_pred
    ax.scatter(uvs[vis_mov, 0], uvs[vis_mov, 1], s=55, c=COLOR_MOVIL_PRED,
               marker=MARCA_PRED, edgecolors="black", linewidths=0.6)

    ax.set_title("Proyección en cámara RGB", fontsize=11, fontweight="bold")
    ax.axis("off")


def _leyenda(fig):
    """Leyenda común abajo de la figura."""
    manejadores = [
        plt.Line2D([0], [0], marker="o", color="w", label="Estático",
                   markerfacecolor=COLOR_ESTATICO, markersize=8),
        plt.Line2D([0], [0], marker=MARCA_GT, color="w", label="Móvil (GT)",
                   markerfacecolor=COLOR_MOVIL_GT, markeredgecolor="black", markersize=15),
        plt.Line2D([0], [0], marker=MARCA_PRED, color="w", label="Móvil (predicción)",
                   markerfacecolor=COLOR_MOVIL_PRED, markeredgecolor="black", markersize=10),
    ]
    fig.legend(handles=manejadores, loc="lower center", ncol=3, frameon=False, fontsize=9)


def figura_comparacion(ruta, puntos, mascara_gt, mascara_pred, imagen, transforms,
                       clip, frame, metricas=None):
    """Figura de 3 paneles: GT (BEV) | proyección RGB | Predicción (BEV).

    Args:
        ruta: dónde guardar el .png (se crean las carpetas que falten).
        puntos: (N, 3) nube radar del frame.
        mascara_gt / mascara_pred: (N,) bool con lo real y lo predicho.
        imagen: imagen RGB del frame (o None).
        transforms: FrameTransformMatrix del frame (para proyectar).
        clip, frame: nombres para el título.
        metricas: dict opcional (mIoU, IoU móvil...) que se pone en el título.
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.5, 1]})

    _panel_bev(axes[0], puntos, mascara_gt, "Ground Truth (BEV)", COLOR_MOVIL_GT, MARCA_GT)
    _panel_proyeccion(axes[1], imagen, puntos, mascara_gt, mascara_pred, transforms)
    _panel_bev(axes[2], puntos, mascara_pred, "Predicción (BEV)", COLOR_MOVIL_PRED, MARCA_PRED)

    titulo = f"Segmentación móvil/estático — {clip} / frame {frame}"
    if metricas:
        titulo += (f"   |   mIoU {metricas['miou']:.3f} · "
                   f"IoU móvil {metricas['iou_moving']:.3f}")
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    _leyenda(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def figura_bev(ruta, puntos, mascara_gt, mascara_pred, clip, frame, metricas=None):
    """Figura de 2 paneles (sin la imagen del medio): GT (BEV) | Predicción (BEV)."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))

    _panel_bev(axes[0], puntos, mascara_gt, "Ground Truth (BEV)", COLOR_MOVIL_GT, MARCA_GT)
    _panel_bev(axes[1], puntos, mascara_pred, "Predicción (BEV)", COLOR_MOVIL_PRED, MARCA_PRED)

    titulo = f"Segmentación móvil/estático — {clip} / frame {frame}"
    if metricas:
        titulo += (f"   |   mIoU {metricas['miou']:.3f} · "
                   f"IoU móvil {metricas['iou_moving']:.3f}")
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    _leyenda(fig)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
