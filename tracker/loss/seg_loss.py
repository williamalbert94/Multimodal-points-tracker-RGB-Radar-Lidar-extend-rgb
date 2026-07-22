"""Funciones de pérdida para la segmentación de puntos móvil/estático.

La pérdida principal es la misma que usó RaTrack y que dio buenos resultados: un
BCE (entropía cruzada binaria) ponderado que le da más peso a los puntos
estáticos, porque son mayoría y no queremos que la red los ignore. Opcionalmente
se le suma una Soft Dice, que empuja directamente el solape entre la predicción y
el GT (útil cuando hay mucho desbalance de clases, como en radar).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def motion_seg_loss_baseline(pred_cls, gt_cls, w_moving=0.4, w_static=0.6):
    """BCE ponderado entre puntos móviles y estáticos (baseline RaTrack).

    Se calcula el BCE por separado en los puntos móviles y en los estáticos, y se
    combinan con pesos fijos. Separarlos evita que la clase mayoritaria (estática)
    domine el gradiente.

    Args:
        pred_cls: [B, 1, N] probabilidades predichas (salida Sigmoid).
        gt_cls:   [B, N]    etiqueta binaria por punto (1 = móvil, 0 = estático).
        w_moving: peso de los puntos móviles (por defecto 0.4).
        w_static: peso de los puntos estáticos (por defecto 0.6).

    Returns:
        Pérdida escalar.
    """
    gt_bool = gt_cls.bool()
    moving_mask = gt_bool                 # puntos móviles
    static_mask = ~gt_bool                # puntos estáticos

    pred_cls = pred_cls.squeeze(1)        # [B, N]

    # Si alguna clase no tiene puntos en el batch, su término es 0 (sin romper).
    if moving_mask.sum() == 0:
        loss_pos = torch.tensor(0.0, device=pred_cls.device)
    else:
        loss_pos = F.binary_cross_entropy(pred_cls[moving_mask], gt_cls[moving_mask].float())

    if static_mask.sum() == 0:
        loss_neg = torch.tensor(0.0, device=pred_cls.device)
    else:
        loss_neg = F.binary_cross_entropy(pred_cls[static_mask], gt_cls[static_mask].float())

    return w_moving * loss_pos + w_static * loss_neg


def soft_dice_loss(pred_cls, gt_cls, smooth=1e-6):
    """Soft Dice para segmentación binaria.

    Optimiza directamente el coeficiente Dice (el solape entre lo predicho y el
    GT). Es complementaria al BCE cuando hay mucho desbalance de clases.

    Args:
        pred_cls: [B, 1, N] probabilidades predichas.
        gt_cls:   [B, N]    etiqueta binaria por punto.
        smooth:   término pequeño para no dividir por cero.

    Returns:
        1 - Dice promedio del batch (escalar).
    """
    pred = pred_cls.squeeze(1)                        # [B, N]
    gt = gt_cls.float()

    intersection = (pred * gt).sum(dim=-1)            # [B]
    union = pred.sum(dim=-1) + gt.sum(dim=-1)         # [B]

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def focal_loss(pred_cls, gt_cls, alpha=0.75, gamma=2.0, eps=1e-7):
    """Focal loss binaria: le baja el volumen a lo fácil y se concentra en lo difícil.

    El problema acá es que solo ~6% de los puntos son móviles. Con un BCE normal,
    los millones de puntos estáticos fáciles (que la red ya acierta con
    probabilidad 0.99) aportan tantísimo gradiente que ahogan a los pocos puntos
    móviles. La focal multiplica la pérdida de cada punto por (1 - p)^gamma, así
    que un punto ya bien clasificado casi no pesa y la red se enfoca en los que
    todavía falla.

    Args:
        pred_cls: [B, 1, N] probabilidades predichas (salida Sigmoid).
        gt_cls:   [B, N]    etiqueta binaria por punto.
        alpha:    peso de la clase móvil (>0.5 la favorece).
        gamma:    qué tanto se castiga lo ya aprendido (2.0 es lo usual).

    Returns:
        Pérdida escalar.
    """
    p = pred_cls.squeeze(1).clamp(eps, 1.0 - eps)          # [B, N]
    gt = gt_cls.float()

    # p_t es la probabilidad que le da la red a la clase CORRECTA de cada punto.
    p_t = p * gt + (1 - p) * (1 - gt)
    alpha_t = alpha * gt + (1 - alpha) * (1 - gt)

    return (-alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t)).mean()


def feature_contrast_loss(features, gt_cls):
    """Separa en el espacio de features a los puntos móviles de los estáticos.

    Para cada frame calcula el centroide (media) de las features de los puntos
    móviles y el de los estáticos, los normaliza y mide su similitud coseno.
    Como queremos que ambas nubes de features queden bien separadas, penalizamos
    cuando los centroides apuntan parecido: loss = (1 + coseno) / 2, que vale 0
    cuando son opuestos y 1 cuando son iguales.

    Args:
        features: [B, C, N] features por punto del backbone (las fusionadas).
        gt_cls:   [B, N]    etiqueta binaria por punto (1 = móvil, 0 = estático).

    Returns:
        Pérdida escalar en [0, 1].
    """
    total, valid = 0.0, 0
    for b in range(features.shape[0]):
        feat_b = features[b]                          # [C, N]
        moving_mask = gt_cls[b].bool()                # [N]
        static_mask = ~moving_mask

        # Necesitamos que el frame tenga puntos de ambas clases.
        if moving_mask.sum() == 0 or static_mask.sum() == 0:
            continue

        c_mov = F.normalize(feat_b[:, moving_mask].mean(dim=1), dim=0)
        c_sta = F.normalize(feat_b[:, static_mask].mean(dim=1), dim=0)
        cos = (c_mov * c_sta).sum()
        total = total + (1.0 + cos) / 2.0
        valid += 1

    if valid == 0:
        # Sin frames válidos: devolvemos 0 pero atado al grafo (para no romper).
        return features.sum() * 0.0
    return total / valid


class SegLoss(nn.Module):
    """Pérdida combinada de segmentación: BCE ponderado + Dice + feature-contrast.

    Reproduce la combinación del experimento que funcionó (testing20482):
    L = 1.0*BCE + dice_weight*Dice + feat_contrast_weight*Contrast.

    Args:
        w_moving, w_static: pesos del BCE por clase.
        dice_weight: peso de la Soft Dice. Si es 0, no se usa.
        feat_contrast_weight: peso de la feature-contrast. Si es 0, no se usa
            (y no hace falta pasarle `features` al forward).
    """

    def __init__(self, w_moving=0.4, w_static=0.6, dice_weight=0.0,
                 feat_contrast_weight=0.0, focal_weight=0.0,
                 focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.w_moving = w_moving
        self.w_static = w_static
        self.dice_weight = dice_weight
        self.feat_contrast_weight = feat_contrast_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, pred_cls, gt_cls, features=None):
        """
        Args:
            pred_cls: [B, 1, N] probabilidades predichas.
            gt_cls:   [B, N]    etiqueta binaria por punto.
            features: [B, C, N] features por punto (solo si se usa feature-contrast).

        Returns:
            (loss_total, dict con el desglose de cada término para logging).
        """
        bce = motion_seg_loss_baseline(pred_cls, gt_cls, self.w_moving, self.w_static)
        parts = {"bce": bce.item()}
        total = bce

        if self.dice_weight > 0:
            dice = soft_dice_loss(pred_cls, gt_cls)
            total = total + self.dice_weight * dice
            parts["dice"] = dice.item()

        if self.focal_weight > 0:
            focal = focal_loss(pred_cls, gt_cls, self.focal_alpha, self.focal_gamma)
            total = total + self.focal_weight * focal
            parts["focal"] = focal.item()

        if self.feat_contrast_weight > 0 and features is not None:
            contrast = feature_contrast_loss(features, gt_cls)
            total = total + self.feat_contrast_weight * contrast
            parts["contrast"] = float(contrast)

        parts["total"] = total.item()
        return total, parts
