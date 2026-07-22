"""Punto de entrada del entrenamiento de SEGMENTACIÓN de puntos.

Uso (dentro del contenedor Docker, con el entorno `mira` activo):

    python -m tracker.runner.train --config tracker/config/seg_train.yaml

Qué hace:
  1. Lee la configuración YAML.
  2. Arma los DataLoaders de train y validación (View-of-Delft).
  3. Conecta con MLflow (si está disponible) y registra los hiperparámetros.
  4. Lanza el loop de entrenamiento/validación (ver train_utils/trainer.py).
"""
import argparse
import logging
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from tracker.config import load_config
from tracker.dataset import TrackingDataVOD
from tracker.logging import MLflowLogger
from tracker.runner.train_utils import custom_collate_fn, run_train_seg


def setup_logger(exp_name):
    """Configura un logger que escribe a consola y a un archivo por experimento."""
    log_dir = "/project/tracker/logs" if os.path.isdir("/project") else "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.log")

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def build_loader(args, train):
    """Construye un DataLoader para train o validación.

    Args:
        args: configuración.
        train: True -> split de entrenamiento (baraja y descarta último batch);
               False -> split de validación.
    """
    # El dataset elige el split según args.eval (ver datagen_vod).
    args.eval = not train
    dataset = TrackingDataVOD(args, args.dataset_path)
    args.eval = False   # se restaura para no afectar otras construcciones

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 4),
        shuffle=bool(train and getattr(args, "shuffle", False)),
        drop_last=train,
        collate_fn=custom_collate_fn,
    )


def main(config_path):
    args = load_config(config_path)

    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(args, "cuda_device", "0"))

    # Semilla para que las corridas sean reproducibles (muestreo de puntos, init).
    seed = int(getattr(args, "seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger = setup_logger(args.exp_name)
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO DE SEGMENTACIÓN DE PUNTOS (móvil/estático) — View-of-Delft")
    logger.info("=" * 70)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)

    # DataLoaders.
    logger.info("Cargando split de entrenamiento...")
    train_loader = build_loader(args, train=True)
    logger.info("Cargando split de validación...")
    val_loader = build_loader(args, train=False)

    # MLflow.
    mlflow_logger = MLflowLogger(
        experiment=getattr(args, "mlflow_experiment", "segmentacion_vod"),
        run_name=args.exp_name,
        enabled=getattr(args, "mlflow_enabled", True),
    )
    mlflow_logger.log_params(vars(args))

    try:
        run_train_seg(args, logger, train_loader, val_loader, mlflow_logger)
    finally:
        mlflow_logger.close()


if __name__ == "__main__":
    # Silenciar avisos ruidosos (numba, etc.) para que el log quede legible.
    warnings.filterwarnings("ignore")
    logging.getLogger("numba").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Entrenamiento de segmentación de puntos.")
    parser.add_argument("--config", type=str, default="tracker/config/seg_train.yaml",
                        help="Ruta al archivo de configuración YAML.")
    cli = parser.parse_args()
    main(cli.config)
