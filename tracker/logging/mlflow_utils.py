"""Conexión con MLflow para registrar el entrenamiento.

MLflow corre como un servicio aparte del docker-compose (`http://mlflow:5000`) y
la dirección llega por la variable de entorno `MLFLOW_TRACKING_URI`, que ya está
inyectada en el contenedor de entrenamiento. Acá solo la usamos.

El wrapper es "tolerante a fallos" a propósito: si el paquete `mlflow` no está
instalado (viene en `requirements/extras.txt`, no en el entorno base) o el
servidor no responde, el entrenamiento NO se cae; simplemente no se registra nada
y se avisa por consola. Así el pipeline sirve con o sin MLflow disponible.
"""
import os

try:
    import mlflow
    _MLFLOW_OK = True
except Exception:                                          # pragma: no cover
    mlflow = None
    _MLFLOW_OK = False


class MLflowLogger:
    """Envuelve las llamadas a MLflow que usa el trainer.

    Args:
        experiment: nombre del experimento en MLflow.
        run_name: nombre de la corrida (por ejemplo, el nombre del experimento).
        enabled: si es False, el logger no hace nada (modo silencioso).
    """

    def __init__(self, experiment="segmentacion_vod", run_name=None, enabled=True):
        self.active = False
        if not enabled:
            return
        if not _MLFLOW_OK:
            print("[mlflow] paquete no instalado (ver requirements/extras.txt); "
                  "se entrena sin registrar en MLflow.")
            return

        try:
            # Si no hay URI configurada, se usa el ./mlruns local por defecto.
            uri = os.environ.get("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=run_name)
            self.active = True
            print(f"[mlflow] registrando en '{experiment}' "
                  f"(uri={uri or 'local ./mlruns'})")
        except Exception as e:                              # pragma: no cover
            print(f"[mlflow] no se pudo conectar ({e}); se entrena sin registrar.")

    def log_params(self, params: dict):
        """Registra los hiperparámetros de la corrida (una sola vez)."""
        if not self.active:
            return
        try:
            # MLflow no acepta valores None ni tipos raros: los pasamos a str.
            clean = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                     for k, v in params.items()}
            mlflow.log_params(clean)
        except Exception as e:                              # pragma: no cover
            print(f"[mlflow] fallo registrando params: {e}")

    def log_metrics(self, metrics: dict, step=None, prefix=""):
        """Registra métricas escalares en un paso dado (por ejemplo, la época).

        Args:
            metrics: dict {nombre: valor}.
            step: número de paso (época).
            prefix: prefijo opcional, p. ej. 'train_' o 'val_'.
        """
        if not self.active:
            return
        try:
            for name, value in metrics.items():
                mlflow.log_metric(f"{prefix}{name}", float(value), step=step)
        except Exception as e:                              # pragma: no cover
            print(f"[mlflow] fallo registrando métricas: {e}")

    def log_artifact(self, path):
        """Sube un archivo (por ejemplo, un checkpoint) a la corrida."""
        if not self.active or not path or not os.path.exists(path):
            return
        try:
            mlflow.log_artifact(path)
        except Exception as e:                              # pragma: no cover
            print(f"[mlflow] fallo subiendo artefacto: {e}")

    def close(self):
        """Cierra la corrida al terminar el entrenamiento."""
        if not self.active:
            return
        try:
            mlflow.end_run()
        except Exception:                                   # pragma: no cover
            pass
