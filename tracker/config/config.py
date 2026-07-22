"""Carga de configuración desde un archivo YAML.

El resto del código espera un objeto `args` al que se le leen atributos con punto
(args.num_points, args.dataset_path, etc.). Acá convertimos el diccionario del
YAML en ese objeto con `SimpleNamespace`, para no tener que andar con
`config["clave"]` por todos lados.
"""
import os
from types import SimpleNamespace

import yaml


def load_config(path):
    """Lee un YAML y lo devuelve como un objeto con atributos.

    Args:
        path: ruta al archivo .yaml.

    Returns:
        SimpleNamespace con cada clave del YAML como atributo.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return SimpleNamespace(**data)
