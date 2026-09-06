"""Post-procesamiento de la segmentación para subir el IoU de la clase móvil.

La red predice punto por punto, sin saber que los puntos de un mismo objeto van
juntos. Eso deja dos tipos de error muy típicos:

* puntos sueltos marcados como móviles en medio de la nada (falsos positivos), y
* huecos: puntos estáticos dentro de un objeto que sí se está moviendo (falsos
  negativos).

Acá corregimos ambos usando la vecindad espacial, que es información gratis que
la métrica por punto no aprovecha:

1. `suavizado_knn`  -> cada punto promedia su probabilidad con la de sus vecinos
                       más cercanos. Limpia el ruido "sal y pimienta".
2. `filtro_clusters` -> agrupa los puntos móviles con DBSCAN y bota los grupos
                       demasiado pequeños (los puntos sueltos).

Además el clustering nos deja los objetos ya armados, que es justo lo que se
escribe en los .txt con el formato de RaTrack.
"""
import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError:                                     # pragma: no cover
    DBSCAN = None


def suavizado_knn(puntos, prob, k=5, radio=1.5, alpha=0.6):
    """Suaviza las probabilidades mezclándolas con las de los vecinos cercanos.

    Los objetos móviles son grupos compactos de puntos, así que si un punto dice
    "soy móvil" pero todos sus vecinos dicen lo contrario, lo más probable es que
    sea ruido. Ahora bien, en radar un objeto puede tener solo 2 o 3 returns, así
    que promediar a lo bruto se lleva por delante detecciones buenas. Por eso se
    mezcla: `alpha` de la probabilidad propia y el resto de los vecinos.

    Args:
        puntos: (N, 3) coordenadas.
        prob:   (N,)   probabilidad de "móvil" que sacó la red.
        k:      cuántos vecinos se miran (incluye el propio punto).
        radio:  distancia máxima para considerar a alguien vecino (metros).
        alpha:  cuánto pesa la probabilidad propia (1.0 = no suaviza nada).

    Returns:
        (N,) probabilidades suavizadas.
    """
    n = len(puntos)
    if n == 0 or k <= 1 or alpha >= 1.0:
        return prob

    # Matriz de distancias (N es chico en radar, ~300 puntos: cabe de sobra).
    d = np.linalg.norm(puntos[:, None, :] - puntos[None, :, :], axis=-1)
    k_ef = min(k, n)
    vecinos = np.argsort(d, axis=1)[:, :k_ef]           # (N, k)

    suave = np.empty(n, dtype=np.float32)
    for i in range(n):
        idx = vecinos[i]
        idx = idx[d[i, idx] <= radio]                   # solo los que están cerca
        suave[i] = alpha * prob[i] + (1 - alpha) * prob[idx].mean() if len(idx) else prob[i]
    return suave


def filtro_clusters(puntos, mascara, eps=2.0, min_samples=1, min_puntos=2):
    """Agrupa los puntos móviles y descarta los grupos muy chicos.

    Un objeto real (carro, peatón, ciclista) deja varios returns de radar juntos.
    Un punto móvil aislado casi siempre es un falso positivo, así que botarlo
    sube la precisión sin perder casi nada de recall.

    Args:
        puntos: (N, 3) coordenadas.
        mascara: (N,) bool con la predicción de "móvil".
        eps: radio de vecindad de DBSCAN (metros).
        min_samples: mínimo de vecinos que DBSCAN pide para formar núcleo.
        min_puntos: tamaño mínimo del grupo para conservarlo.

    Returns:
        mascara_limpia: (N,) bool ya filtrada.
        clusters: lista de arrays con los índices (sobre la nube original) de
            cada objeto detectado.
    """
    idx_mov = np.where(mascara)[0]
    if len(idx_mov) == 0 or DBSCAN is None:
        return mascara.copy(), []

    etiquetas = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(puntos[idx_mov])

    mascara_limpia = np.zeros_like(mascara)
    clusters = []
    for etq in set(etiquetas):
        if etq == -1:                                   # ruido segun DBSCAN
            continue
        miembros = idx_mov[etiquetas == etq]
        if len(miembros) < min_puntos:
            continue
        mascara_limpia[miembros] = True
        clusters.append(miembros)

    return mascara_limpia, clusters


def postprocesar(puntos, prob, umbral=0.5, usar_knn=True, k=5, radio_knn=1.5,
                 alpha=0.6, usar_clusters=True, eps=2.0, min_samples=1, min_puntos=2):
    """Aplica todo el post-procesamiento y devuelve la máscara final.

    Orden: suavizado por vecinos -> umbral -> filtro de clusters.

    Args:
        puntos: (N, 3) coordenadas.
        prob:   (N,) probabilidad cruda de la red.
        umbral: punto de corte para decidir "móvil".
        usar_knn / usar_clusters: permiten apagar cada etapa.
        (el resto son los parámetros de cada etapa)

    Returns:
        mascara: (N,) bool final.
        clusters: lista de arrays de índices (vacía si no se usó clustering).
        prob_final: (N,) probabilidades después del suavizado.
    """
    prob_final = (suavizado_knn(puntos, prob, k=k, radio=radio_knn, alpha=alpha)
                  if usar_knn else prob)
    mascara = prob_final > umbral

    clusters = []
    if usar_clusters:
        mascara, clusters = filtro_clusters(
            puntos, mascara, eps=eps, min_samples=min_samples, min_puntos=min_puntos)

    return mascara, clusters, prob_final
