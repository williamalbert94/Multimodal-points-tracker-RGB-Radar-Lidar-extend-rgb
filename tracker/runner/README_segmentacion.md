# Entrenamiento de segmentación de puntos (móvil / estático)

Primera etapa del tracker: la red aprende a decidir, punto por punto de la nube
radar, si pertenece a un objeto en movimiento (**móvil = 1**) o no
(**estático = 0**). La métrica principal es el **mIoU** y todo se registra en
**MLflow**.

## Qué hace cada pieza

| Módulo | Rol |
|--------|-----|
| `tracker/model/feature_extractor.py` | PointNet++ con fusión local-global (el extractor que mejor segmentó). |
| `tracker/model/segnet.py` | Red completa: extractor + cabeza de segmentación supervisada. |
| `tracker/loss/seg_loss.py` | BCE ponderado (0.4 móvil / 0.6 estático) + Soft Dice opcional. |
| `tracker/metrics/seg_metrics.py` | mIoU, IoU por clase, accuracy, sensibilidad, F1. |
| `tracker/dataset/gt_labels.py` | Etiqueta GT por punto: lleva las cajas al frame radar y marca los puntos que caen adentro. |
| `tracker/logging/mlflow_utils.py` | Registro en MLflow (tolerante a fallos). |
| `tracker/runner/train_utils/trainer.py` | Loop de train/val y checkpoint por mejor mIoU. |
| `tracker/runner/train.py` | Punto de entrada. |
| `tracker/config/seg_train.yaml` | Configuración (todo comentado). |

## Cómo se arma el GT de segmentación

El dataset entrega las cajas de los objetos **en movimiento** en coordenadas de
cámara. Para cada frame:

1. Se lleva cada caja al frame del radar (`get_bbx_param`, igual que el repo
   anterior).
2. Se marca como **móvil** todo punto radar que caiga dentro de alguna caja
   orientada (Open3D). El resto queda **estático**.

El frame que se segmenta es `raw_pc0`, emparejado con sus cajas `lbl1` y sus
transformaciones `transforms1` (así lo entrega `datagen_vod`).

## Correr el entrenamiento (dentro del Docker)

Requisitos que ya cubre la imagen `mira`: PyTorch 2.2 + CUDA 11.8, Open3D, y la
librería compilada `external/lib` (pointnet2). El cliente de MLflow viene en
`requirements/extras.txt` y el servidor corre como servicio aparte del
`docker-compose` en `http://mlflow:5000` (ya inyectado como
`MLFLOW_TRACKING_URI`).

```bash
# 1. Levantar los servicios (incluye el servidor MLflow):
docker compose -f docker/docker-compose.yml up -d

# 2. Entrar al contenedor de entrenamiento:
docker exec -it tracker_multimodal_mira bash

# 3. (una sola vez) compilar pointnet2 e instalar el cliente MLflow:
cd /project/external/lib && python setup.py install && cd /project
pip install -r requirements/extras.txt

# 4. Lanzar el entrenamiento de segmentación:
python -m tracker.runner.train --config tracker/config/seg_train.yaml
```

- Los checkpoints quedan en `tracker/checkpoints/<exp_name>/`
  (`best_miou_model.pth` = mejor mIoU de validación; `last_model.pth` = último).
- Las métricas por época (loss, mIoU, IoU_móvil, IoU_estático, F1, lr) se ven en
  la UI de MLflow: `http://localhost:5000`.

## Notas

- Si MLflow no está instalado o el servidor no responde, el entrenamiento **no se
  cae**: sigue corriendo sin registrar (se avisa por consola).
- Para probar el extractor liviano sin la rama global, cambiar en el YAML
  `extractor: PNHead`.
- La fusión temprana LiDAR-radar está en `none` por defecto (solo radar). Se
  activa desde el YAML (`fusion: radar_base`, etc.); si se usa un modo con más
  canales, ajustar también `in_channels`.
