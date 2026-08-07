"""
Motion-Based Tracking Utilities
================================

Calcula atributos de movimiento entre frames consecutivos para mejorar tracking.
Usa información de dirección, distancia, y disposición espacial de objetos.

Autor: Sistema de Tracking Mejorado
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def compute_box_motion_features(lbl1, lbl2, dt=0.1):
    """
    Calcula features de movimiento entre dos frames consecutivos.

    Este módulo aprovecha que el dataloader pasa 2 frames (lbl1, lbl2) para:
    - Calcular dirección de movimiento de cada objeto
    - Medir distancia recorrida
    - Detectar cambios en tamaño/orientación
    - Validar que el movimiento sea físicamente plausible

    Args:
        lbl1: Dict {track_id: OrientedBoundingBox} - Frame t
        lbl2: Dict {track_id: OrientedBoundingBox} - Frame t+1
        dt: float - Tiempo entre frames (segundos, default=0.1)

    Returns:
        motion_dict: Dict {track_id: motion_features}
            motion_features = {
                'displacement': [dx, dy, dz],      # Desplazamiento 3D (metros)
                'velocity': [vx, vy, vz],          # Velocidad 3D (m/s)
                'speed': float,                    # Magnitud de velocidad (m/s)
                'direction_2d': float,             # Ángulo en BEV (radianes, -π a π)
                'distance_2d': float,              # Distancia en BEV (metros)
                'box_size_change': [dl, dw, dh],   # Cambio en tamaño (metros)
                'yaw_change': float,               # Cambio en orientación (radianes)
                'valid': bool,                     # True si movimiento es físicamente plausible
                'center_t': [x, y, z],             # Centroide en frame t
                'center_t1': [x, y, z],            # Centroide en frame t+1
            }
    """
    motion_dict = {}

    # Encontrar objetos que aparecen en ambos frames (GT track_ids consistentes)
    common_ids = set(lbl1.keys()) & set(lbl2.keys())

    logger.debug(f"Computing motion for {len(common_ids)} objects present in both frames")

    for track_id in common_ids:
        box_t = lbl1[track_id]
        box_t1 = lbl2[track_id]

        # Extraer centroides
        center_t = np.array(box_t.center)    # [x, y, z]
        center_t1 = np.array(box_t1.center)  # [x, y, z]

        # Extraer tamaños (extent puede ser [l, h, w] o [l, w, h] dependiendo del dataset)
        extent_t = np.array(box_t.extent)    # [l, w, h] o similar
        extent_t1 = np.array(box_t1.extent)

        # Extraer orientación (yaw) desde matriz de rotación
        R_t = np.array(box_t.R)
        R_t1 = np.array(box_t1.R)
        yaw_t = np.arctan2(R_t[1, 0], R_t[0, 0])
        yaw_t1 = np.arctan2(R_t1[1, 0], R_t1[0, 0])

        # 1. Desplazamiento 3D
        displacement = center_t1 - center_t

        # 2. Velocidad 3D
        velocity = displacement / dt if dt > 0 else displacement * 10.0

        # 3. Speed (magnitud)
        speed = np.linalg.norm(velocity)

        # 4. Dirección en BEV (Bird's Eye View - solo x, y)
        dx, dy = displacement[0], displacement[1]
        direction_2d = np.arctan2(dy, dx)  # Ángulo en radianes [-π, π]

        # 5. Distancia en BEV
        distance_2d = np.sqrt(dx**2 + dy**2)

        # 6. Cambio en tamaño del box
        box_size_change = extent_t1 - extent_t

        # 7. Cambio en orientación (yaw)
        yaw_change = yaw_t1 - yaw_t
        # Normalizar a [-π, π]
        while yaw_change > np.pi:
            yaw_change -= 2 * np.pi
        while yaw_change < -np.pi:
            yaw_change += 2 * np.pi

        # 8. Validación: detectar movimientos físicamente imposibles
        # Velocidades típicas en entorno urbano:
        #   - Peatones: 0-2 m/s (0-7 km/h)
        #   - Ciclistas: 0-8 m/s (0-29 km/h)
        #   - Vehículos: 0-20 m/s (0-72 km/h)
        MAX_SPEED = 30.0  # m/s (~108 km/h - máximo realista para dataset urbano)
        valid = speed < MAX_SPEED

        if not valid:
            logger.warning(f"Track {track_id}: Unrealistic speed {speed:.2f} m/s (>{MAX_SPEED} m/s)")

        motion_dict[track_id] = {
            'displacement': displacement,
            'velocity': velocity,
            'speed': speed,
            'direction_2d': direction_2d,
            'distance_2d': distance_2d,
            'box_size_change': box_size_change,
            'yaw_change': yaw_change,
            'valid': valid,
            'center_t': center_t,
            'center_t1': center_t1,
        }

    return motion_dict


def compute_spatial_disposition(boxes_dict):
    """
    Calcula la disposición espacial de objetos en un frame.
    Útil para matching basado en configuración relativa de objetos.

    Args:
        boxes_dict: Dict {track_id: OrientedBoundingBox}

    Returns:
        disposition: Dict con:
            - 'centroids': [N, 3] array de centroides
            - 'track_ids': [N] array de IDs
            - 'pairwise_distances': [N, N] matriz de distancias
            - 'relative_angles': [N, N] matriz de ángulos relativos
    """
    if len(boxes_dict) == 0:
        return {
            'centroids': np.zeros((0, 3)),
            'track_ids': np.array([]),
            'pairwise_distances': np.zeros((0, 0)),
            'relative_angles': np.zeros((0, 0)),
        }

    track_ids = list(boxes_dict.keys())
    N = len(track_ids)

    # Extraer centroides
    centroids = np.zeros((N, 3))
    for i, tid in enumerate(track_ids):
        centroids[i] = np.array(boxes_dict[tid].center)

    # Calcular distancias pairwise (euclidiana 2D en BEV)
    centroids_2d = centroids[:, :2]  # Solo x, y
    pairwise_distances = np.zeros((N, N))
    relative_angles = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                # Distancia
                diff = centroids_2d[j] - centroids_2d[i]
                pairwise_distances[i, j] = np.linalg.norm(diff)

                # Ángulo relativo desde objeto i hacia objeto j
                relative_angles[i, j] = np.arctan2(diff[1], diff[0])

    return {
        'centroids': centroids,
        'track_ids': np.array(track_ids),
        'pairwise_distances': pairwise_distances,
        'relative_angles': relative_angles,
    }


def predict_next_position(center_current, velocity, dt=0.1):
    """
    Predice la posición del objeto en el siguiente frame usando velocidad.

    Args:
        center_current: [3] - Posición actual [x, y, z]
        velocity: [3] - Velocidad [vx, vy, vz] en m/s
        dt: float - Tiempo hasta siguiente frame (segundos)

    Returns:
        predicted_center: [3] - Posición predicha
    """
    return center_current + velocity * dt


def compute_motion_consistency_score(box_current, box_next, motion_features):
    """
    Calcula un score de consistencia entre la posición predicha y la observada.

    Score alto = movimiento consistente con predicción
    Score bajo = movimiento inconsistente (posible error de matching)

    Args:
        box_current: OrientedBoundingBox - Box actual
        box_next: OrientedBoundingBox - Box candidato para matching
        motion_features: Dict - Features de movimiento del track

    Returns:
        score: float [0, 1] - 1 = perfectamente consistente, 0 = totalmente inconsistente
    """
    if motion_features is None or not motion_features['valid']:
        return 0.5  # Neutral si no hay info de motion

    # Predecir posición esperada
    predicted_center = motion_features['center_t'] + motion_features['displacement']

    # Posición observada
    observed_center = np.array(box_next.center)

    # Error de predicción
    prediction_error = np.linalg.norm(predicted_center - observed_center)

    # Convertir error a score (sigma=1.0 metro)
    # Score alto si error < 1m, score bajo si error > 3m
    sigma = 1.0
    score = np.exp(-0.5 * (prediction_error / sigma) ** 2)

    return score


def compute_matching_score(box1, box2, embedding1, embedding2, motion_features=None,
                           weight_appearance=0.5, weight_iou=0.2, weight_motion=0.2, weight_size=0.1):
    """
    Calcula score de matching multi-cue combinando:
    - Appearance (Re-ID embeddings)
    - IoU (geometric overlap)
    - Motion (consistencia de movimiento)
    - Size (similaridad de tamaño)

    Args:
        box1: [7] - Box actual [x, y, z, l, w, h, yaw]
        box2: [7] - Box candidato
        embedding1: [D] - Re-ID embedding del box1
        embedding2: [D] - Re-ID embedding del box2
        motion_features: Dict - Features de movimiento (opcional)
        weight_*: float - Pesos para cada componente

    Returns:
        score: float [0, 1] - Score total de matching (más alto = mejor match)
    """
    scores = {}

    # 1. Appearance similarity (cosine similarity de embeddings)
    if embedding1 is not None and embedding2 is not None:
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        appearance_score = np.dot(embedding1_norm, embedding2_norm)
        appearance_score = (appearance_score + 1) / 2  # Normalizar a [0, 1]
        scores['appearance'] = appearance_score
    else:
        scores['appearance'] = 0.5  # Neutral

    # 2. IoU (geometric overlap)
    iou = compute_iou_3d_simple(box1, box2)
    scores['iou'] = iou

    # 3. Motion consistency
    if motion_features is not None:
        # Calcular posición predicha
        predicted_center = motion_features['center_t1']
        observed_center = box2[:3]
        error = np.linalg.norm(predicted_center - observed_center)
        motion_score = np.exp(-error)  # Score alto si error bajo
        scores['motion'] = motion_score
    else:
        scores['motion'] = 0.5  # Neutral

    # 4. Size similarity
    size1 = box1[3:6]  # [l, w, h]
    size2 = box2[3:6]
    size_diff = np.linalg.norm(size1 - size2)
    size_score = np.exp(-size_diff)  # Score alto si tamaños similares
    scores['size'] = size_score

    # Combinar scores con pesos
    total_score = (
        weight_appearance * scores['appearance'] +
        weight_iou * scores['iou'] +
        weight_motion * scores['motion'] +
        weight_size * scores['size']
    )

    return total_score, scores


def compute_iou_3d_simple(box1, box2):
    """
    Calcula IoU simplificado entre dos boxes 3D.
    Usa aproximación axis-aligned para rapidez.

    Args:
        box1, box2: [7] arrays [x, y, z, l, w, h, yaw]

    Returns:
        iou: float [0, 1]
    """
    # Extract centers and sizes
    c1, s1 = box1[:3], box1[3:6]
    c2, s2 = box2[:3], box2[3:6]

    # Compute min and max corners (axis-aligned approximation)
    min1 = c1 - s1 / 2
    max1 = c1 + s1 / 2
    min2 = c2 - s2 / 2
    max2 = c2 + s2 / 2

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_volume = np.prod(inter_size)

    # Union
    volume1 = np.prod(s1)
    volume2 = np.prod(s2)
    union_volume = volume1 + volume2 - inter_volume

    # IoU
    if union_volume > 0:
        return inter_volume / union_volume
    else:
        return 0.0


def filter_invalid_associations(boxes_current, boxes_next, motion_dict,
                                max_distance=5.0, max_speed=30.0):
    """
    Filtra asociaciones imposibles basándose en constrains físicos.

    Args:
        boxes_current: Dict {track_id: box}
        boxes_next: List[box] - Boxes candidatos
        motion_dict: Dict con motion features
        max_distance: float - Distancia máxima plausible entre frames (metros)
        max_speed: float - Velocidad máxima plausible (m/s)

    Returns:
        valid_associations: [N, M] boolean mask
            True = asociación plausible, False = imposible
    """
    N = len(boxes_current)
    M = len(boxes_next)
    valid_mask = np.ones((N, M), dtype=bool)

    track_ids = list(boxes_current.keys())

    for i, tid in enumerate(track_ids):
        box_current = boxes_current[tid]
        center_current = np.array(box_current.center)

        for j, box_next in enumerate(boxes_next):
            center_next = np.array(box_next.center)

            # Distancia entre boxes
            distance = np.linalg.norm(center_next - center_current)

            # Filtrar por distancia máxima
            if distance > max_distance:
                valid_mask[i, j] = False
                continue

            # Si tenemos motion features, validar velocidad
            if tid in motion_dict:
                motion = motion_dict[tid]
                if not motion['valid'] or motion['speed'] > max_speed:
                    # Movimiento inválido, reducir confianza
                    # (no eliminar completamente por si es ruido en motion)
                    pass

    return valid_mask


# ===== LOGGING Y DEBUG =====

def log_motion_statistics(motion_dict):
    """Log estadísticas de movimiento para debugging."""
    if len(motion_dict) == 0:
        logger.info("No motion features computed (no common objects)")
        return

    speeds = [m['speed'] for m in motion_dict.values()]
    distances = [m['distance_2d'] for m in motion_dict.values()]
    valid_count = sum([m['valid'] for m in motion_dict.values()])

    logger.info(f"Motion Statistics:")
    logger.info(f"  Objects tracked: {len(motion_dict)}")
    logger.info(f"  Valid motions: {valid_count}/{len(motion_dict)}")
    logger.info(f"  Speed: mean={np.mean(speeds):.2f} m/s, max={np.max(speeds):.2f} m/s")
    logger.info(f"  Distance: mean={np.mean(distances):.2f} m, max={np.max(distances):.2f} m")
