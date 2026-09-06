"""Métricas de segmentación de puntos (móvil vs. estático).

La métrica central es el mIoU, con la misma fórmula de RaTrack: el promedio del
IoU de las dos clases a nivel de punto.

    IoU_móvil    = TP / (TP + FP + FN)
    IoU_estático = TN / (TN + FP + FN)
    mIoU         = 0.5 * (IoU_móvil + IoU_estático)

Por qué se acumulan los conteos (micro-promedio)
------------------------------------------------
Es tentador calcular el IoU en cada batch y promediar al final, pero acá eso
engaña feo: cerca del 40% de los frames de View-of-Delft NO tienen ningún objeto
en movimiento. En esos frames TP, FP y FN quedan todos en cero y, al sumarles el
epsilon que evita la división por cero, el IoU móvil da exactamente 1/3 = 0.333.
Promediando, esos frames "regalan" 0.333 y suben el número aunque el modelo no
haya detectado nada.

Por eso acá se acumulan TP/FP/FN/TN de TODO el split y el IoU se calcula una sola
vez al final. Así el número refleja lo que de verdad acertó el modelo.
"""
import torch

EPS = 1e-20


def metricas_de_conteos(tp, fp, fn, tn):
    """Arma el diccionario de métricas a partir de los conteos acumulados.

    Args:
        tp, fp, fn, tn: conteos (aciertos y errores) ya sumados.

    Returns:
        dict con mIoU, IoU por clase, accuracy, sensibilidad y F1.
    """
    tp, fp, fn, tn = tp + EPS, fp + EPS, fn + EPS, tn + EPS
    iou_moving = tp / (tp + fp + fn)
    iou_static = tn / (tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "miou": 0.5 * (iou_moving + iou_static),
        "iou_moving": iou_moving,
        "iou_static": iou_static,
        "acc": (tp + tn) / (tp + tn + fp + fn),
        "sen": recall,                                     # sensibilidad = recall móvil
        "f1": 2 * precision * recall / (precision + recall + EPS),
    }


def contar_confusion(pred_seg, gt_seg, threshold=0.5, valid=None):
    """Cuenta TP/FP/FN/TN de un batch.

    Args:
        pred_seg:  [B, 1, N] probabilidades predichas (Sigmoid).
        gt_seg:    [B, N]    etiqueta binaria por punto (1 = móvil).
        threshold: umbral para binarizar la predicción.

    Returns:
        (tp, fp, fn, tn) como floats de Python.
    """
    pred = (pred_seg.squeeze(1) > threshold).bool()
    gt = gt_seg.bool()

    # Los puntos marcados para ignorar no cuentan ni a favor ni en contra: se
    # multiplica cada conteo por la máscara de válidos.
    v = torch.ones_like(pred) if valid is None else valid.bool()

    return (
        float(torch.sum(pred & gt & v)),                    # TP
        float(torch.sum(pred & ~gt & v)),                   # FP
        float(torch.sum(~pred & gt & v)),                   # FN
        float(torch.sum(~pred & ~gt & v)),                  # TN
    )


def compute_seg_metrics(pred_seg, gt_seg, threshold=0.5):
    """Métricas de un solo batch (útil para mostrar un frame suelto).

    Ojo: para evaluar una época entera NO se deben promediar estos valores; hay
    que usar `SegMetricAccumulator`, que acumula los conteos.
    """
    return metricas_de_conteos(*contar_confusion(pred_seg, gt_seg, threshold))


class SegMetricAccumulator:
    """Acumula los conteos de toda la época y calcula el mIoU al final.

    Uso:
        acc = SegMetricAccumulator()
        for batch in loader:
            acc.update(pred_seg, gt_seg)
        metricas = acc.average()
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, pred_seg, gt_seg, valid=None):
        """Suma los conteos de un batch (ignorando los puntos no válidos)."""
        tp, fp, fn, tn = contar_confusion(pred_seg, gt_seg, self.threshold, valid)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def average(self):
        """Métricas micro-promediadas de todo lo acumulado."""
        if (self.tp + self.fp + self.fn + self.tn) == 0:
            return {}
        return metricas_de_conteos(self.tp, self.fp, self.fn, self.tn)
