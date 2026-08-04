"""Loop de entrenamiento y validación para la segmentación de puntos.

Flujo por época:
  1. Entrenar: por cada batch se muestrean los puntos, se arman las etiquetas GT
     por punto (a partir de las cajas móviles), se predice la segmentación y se
     retropropaga la pérdida BCE ponderada.
  2. Validar (cada `val_every` épocas): igual pero sin gradientes, para medir el
     mIoU en el split de validación.
  3. Guardar el mejor modelo según el mIoU de validación.

Todo lo relevante (pérdida, mIoU, IoU por clase, learning rate) se registra en
MLflow época a época.
"""
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tracker.dataset.gt_labels import moving_point_labels_batch
from tracker.loss import SegLoss
from tracker.metrics import SegMetricAccumulator
from tracker.model import build_model
from .collate import sample_points


def _prepare_batch(batch, num_points, in_channels, device, gt_margen=0.0,
                   min_puntos=0, multiframe=False):
    """Prepara un batch para el modelo: muestrea puntos y arma el GT.

    El dataset entrega una tupla larga; para segmentación solo usamos el frame
    `raw_pc0` (índice 0) con sus features (índice 2), sus cajas `lbl1` (índice 12)
    y sus transformaciones `transforms1` (índice 14).

    Muestreamos coordenadas y features JUNTAS (con los mismos índices) para que
    cada punto conserve su RCS y su velocidad. Luego calculamos la etiqueta GT
    sobre esos mismos puntos muestreados.

    Returns:
        pc1:     [B, 3, N] coordenadas (en `device`).
        ft1:     [B, C, N] features radar (en `device`).
        gt_cls:  [B, N]    etiqueta GT por punto, float (en `device`).
    """
    raw_pc0 = batch[0]        # tupla de B arrays [N_i, 3]  -> frame actual (t+1)
    raw_pc1 = batch[1]        # tupla de B arrays [M_i, 3]  -> frame anterior (t)
    features0 = batch[2]      # tupla de B arrays [N_i, C]
    features1 = batch[3]      # features del frame anterior
    raw_pc0_comp = batch[4]   # frame actual compensado por ego-movimiento
    lbl1 = batch[12]          # tupla de B dicts de cajas móviles
    transforms1 = batch[14]   # tupla de B FrameTransformMatrix

    # Concatenamos coords + features + nube compensada por muestra, para
    # muestrear todo con los MISMOS índices y que cada punto conserve su
    # velocidad y su posición compensada. Luego se separan.
    combinado = [np.hstack([pc, ft, cp[:, :3]])
                 for pc, ft, cp in zip(raw_pc0, features0, raw_pc0_comp)]
    combinado = sample_points(combinado, num_points)          # [B, N, 3+C+3]

    c = 3 + in_channels
    pc1 = combinado[:, :, :3].permute(0, 2, 1).contiguous().to(device)          # [B, 3, N]
    ft1 = combinado[:, :, 3:c].permute(0, 2, 1).contiguous().to(device)         # [B, C, N]
    # Las 3 últimas columnas son la nube compensada (van al final del hstack).
    pc1_comp = combinado[:, :, -3:].permute(0, 2, 1).contiguous().to(device)    # [B, 3, N]

    # El frame anterior se muestrea aparte (tiene otra cantidad de puntos).
    prev = [np.hstack([pc, ft]) for pc, ft in zip(raw_pc1, features1)]
    prev = sample_points(prev, num_points)                    # [B, N, 3+C]
    pc2 = prev[:, :, :3].permute(0, 2, 1).contiguous().to(device)               # [B, 3, N]
    ft2 = prev[:, :, 3:c].permute(0, 2, 1).contiguous().to(device)              # [B, C, N]

    # Etiqueta GT por punto (1 = móvil) usando las cajas del mismo frame.
    gt_cls, ignorar = moving_point_labels_batch(
        list(lbl1), pc1, list(transforms1), margen=gt_margen,
        min_puntos=min_puntos, devolver_ignorar=True)
    # `valido` es lo contrario de ignorar: los puntos que sí entran a la cuenta.
    valido = (~ignorar).float().to(device)

    out = {
        "pc1": pc1, "ft1": ft1,
        "pc2": pc2, "ft2": ft2, "pc1_comp": pc1_comp,
        "gt": gt_cls.float().to(device), "valido": valido,
    }

    # Multi-frame: segundo frame de referencia t-2 (índices 17-19 de la tupla).
    # pc3/ft3 = radar t-2 muestreado; pc1_comp2 = nube actual compensada a t-2,
    # muestreada con los MISMOS índices que pc1/ft1 (por eso va junto al frame
    # actual). El correlador usa pc1_comp2 (actual en el sistema de t-2) contra
    # pc3 (t-2).
    if multiframe:
        raw_pc2b = batch[17]
        features2b = batch[18]
        raw_pc0_comp2 = batch[19]

        # nube actual + su compensación a t-2, muestreadas con los mismos índices
        combinado2 = [np.hstack([pc, ft, cp[:, :3]])
                      for pc, ft, cp in zip(raw_pc0, features0, raw_pc0_comp2)]
        combinado2 = sample_points(combinado2, num_points)
        out["pc1_comp2"] = combinado2[:, :, -3:].permute(0, 2, 1).contiguous().to(device)

        prev2 = [np.hstack([pc, ft]) for pc, ft in zip(raw_pc2b, features2b)]
        prev2 = sample_points(prev2, num_points)
        out["pc3"] = prev2[:, :, :3].permute(0, 2, 1).contiguous().to(device)
        out["ft3"] = prev2[:, :, 3:c].permute(0, 2, 1).contiguous().to(device)

    return out


def _run_epoch(net, loader, criterion, optimizer, args, device, train=True):
    """Corre una época completa (train o val) y devuelve las métricas promedio.

    Args:
        net: la red de segmentación.
        loader: DataLoader del split correspondiente.
        criterion: la pérdida (SegLoss).
        optimizer: el optimizador (solo se usa en train).
        args: configuración.
        device: 'cuda'.
        train: True para entrenar, False para validar.

    Returns:
        dict con las métricas promedio de la época (incluye 'loss').
    """
    net.train() if train else net.eval()
    acc = SegMetricAccumulator()
    loss_sum, n_batches = 0.0, 0

    multiframe = bool(getattr(args, "multiframe", False)) or bool(getattr(args, "lidar_temporal", False))
    for batch in loader:
        d = _prepare_batch(
            batch, args.num_points, args.in_channels, device,
            getattr(args, "gt_box_margin", 0.0),
            int(getattr(args, "min_obj_points", 0)),
            multiframe=multiframe)
        gt_cls, valido = d["gt"], d["valido"]

        with torch.set_grad_enabled(train):
            if multiframe:
                seg, feats = net(d["pc1"], d["ft1"], d["pc2"], d["ft2"], d["pc1_comp"],
                                 pc3=d["pc3"], feature3=d["ft3"], pc1_comp2=d["pc1_comp2"])
            else:
                seg, feats = net(d["pc1"], d["ft1"], d["pc2"], d["ft2"], d["pc1_comp"])
            loss, _ = criterion(seg, gt_cls, feats, valid=valido)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_sum += loss.item()
        n_batches += 1
        # Se acumulan los conteos del batch; el IoU se calcula al final de la
        # época (micro-promedio). Promediar IoUs por batch inflaría el número:
        # los frames sin objetos móviles regalan un 1/3 (ver seg_metrics).
        acc.update(seg.detach(), gt_cls, valid=valido)

    metrics = acc.average()
    metrics["loss"] = loss_sum / max(n_batches, 1)
    return metrics


def _save_checkpoint(net, path, epoch, miou):
    """Guarda los pesos del modelo junto con la época y el mIoU alcanzado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": net.state_dict(), "epoch": epoch, "miou": miou}, path)


def run_train_seg(args, logger, train_loader, val_loader, mlflow_logger=None):
    """Entrena la red de segmentación y valida periódicamente.

    Args:
        args: configuración (epochs, lr, val_every, checkpoint_dir, etc.).
        logger: logger estándar para mensajes por consola/archivo.
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        mlflow_logger: `MLflowLogger` opcional para registrar la corrida.
    """
    device = "cuda"
    net = build_model(args, logger)
    criterion = SegLoss(
        w_moving=getattr(args, "w_moving", 0.4),
        w_static=getattr(args, "w_static", 0.6),
        dice_weight=getattr(args, "dice_weight", 0.0),
        feat_contrast_weight=getattr(args, "feat_contrast_weight", 0.0),
        focal_weight=getattr(args, "focal_weight", 0.0),
        focal_alpha=getattr(args, "focal_alpha", 0.75),
        focal_gamma=getattr(args, "focal_gamma", 2.0),
    )
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=getattr(args, "weight_decay", 1e-4))
    # ReduceLROnPlateau: baja el LR cuando el mIoU de validación deja de mejorar
    # (es el scheduler del config que funcionó). Se da un paso solo cuando hay
    # una validación nueva, usando su mIoU (mode='max').
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",
        factor=getattr(args, "scheduler_factor", 0.5),
        patience=getattr(args, "scheduler_patience", 5),
        min_lr=getattr(args, "scheduler_min_lr", 1e-6),
    )

    checkpoint_dir = os.path.join(getattr(args, "checkpoint_dir", "./checkpoints"), args.exp_name)
    val_every = int(getattr(args, "val_every", 2))
    best_miou = 0.0

    # Early stopping: si el mIoU de validación no mejora en `paciencia`
    # validaciones seguidas, se corta. En el experimento C el pico fue en la
    # época 22 y se gastaron 73 épocas más sin superarlo.
    paciencia = int(getattr(args, "early_stop_patience", 0))
    sin_mejorar = 0

    for epoch in range(args.epochs):
        lr_actual = optimizer.param_groups[0]["lr"]
        logger.info(f"\n===== Época {epoch + 1}/{args.epochs}  (lr={lr_actual:.6f}) =====")

        # ── Entrenamiento ────────────────────────────────────────────────────
        train_metrics = _run_epoch(net, train_loader, criterion, optimizer, args, device, train=True)
        logger.info(f"[train] loss={train_metrics['loss']:.4f}  "
                    f"mIoU={train_metrics['miou']:.4f}  "
                    f"IoU_móvil={train_metrics['iou_moving']:.4f}  "
                    f"IoU_estático={train_metrics['iou_static']:.4f}  "
                    f"F1={train_metrics['f1']:.4f}")
        if mlflow_logger:
            mlflow_logger.log_metrics(train_metrics, step=epoch, prefix="train_")
            mlflow_logger.log_metrics({"lr": lr_actual}, step=epoch)

        # ── Validación ───────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0 or epoch == args.epochs - 1:
            val_metrics = _run_epoch(net, val_loader, criterion, optimizer, args, device, train=False)
            logger.info(f"[ val ] loss={val_metrics['loss']:.4f}  "
                        f"mIoU={val_metrics['miou']:.4f}  "
                        f"IoU_móvil={val_metrics['iou_moving']:.4f}  "
                        f"IoU_estático={val_metrics['iou_static']:.4f}  "
                        f"F1={val_metrics['f1']:.4f}")
            if mlflow_logger:
                mlflow_logger.log_metrics(val_metrics, step=epoch, prefix="val_")

            # Guardar el mejor modelo según mIoU de validación.
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                sin_mejorar = 0
                ckpt_path = os.path.join(checkpoint_dir, "best_miou_model.pth")
                _save_checkpoint(net, ckpt_path, epoch, best_miou)
                logger.info(f"🏆 Nuevo mejor mIoU={best_miou:.4f} -> guardado en {ckpt_path}")
                if mlflow_logger:
                    mlflow_logger.log_metrics({"best_miou": best_miou}, step=epoch)
                    mlflow_logger.log_artifact(ckpt_path)
            else:
                sin_mejorar += 1

            # El scheduler solo avanza cuando hay una validación nueva.
            scheduler.step(val_metrics["miou"])

            if paciencia and sin_mejorar >= paciencia:
                logger.info(f"\n⏹  Early stopping: {sin_mejorar} validaciones seguidas "
                            f"sin superar mIoU={best_miou:.4f}. Se corta en la época {epoch + 1}.")
                break

    # Guardar también el último modelo.
    last_path = os.path.join(checkpoint_dir, "last_model.pth")
    _save_checkpoint(net, last_path, epoch, best_miou)
    logger.info(f"\nEntrenamiento terminado. Mejor mIoU={best_miou:.4f}. "
                f"Último modelo en {last_path}")
    return net
