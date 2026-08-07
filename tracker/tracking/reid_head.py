"""Cabeza Re-ID sobre las features del backbone Q.

Entrada por instancia (caja móvil):
    apariencia = [feat_max (768) ; feat_avg (768)] = 1536  (features Q pooleadas
                 de los puntos radar MÓVILES dentro de la caja)
    geometría  = [x, y, z, l, w, h, yaw] = 7

Salida: embedding L2-normalizado de dimensión `embedding_dim`, entrenable con
triplet loss sobre identidades de track (GT). Réplica adaptada de la cabeza de
la referencia (`reid_features.py`), pero tomando las features Q ya pooleadas en
vez de correr un backbone por caja.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDHead(nn.Module):
    def __init__(self, appear_dim=1536, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.app_encoder = nn.Sequential(
            nn.LayerNorm(appear_dim),           # normaliza las features Q crudas
            nn.Linear(appear_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
        )
        self.box_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.LayerNorm(64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, embedding_dim), nn.LayerNorm(embedding_dim),
        )

    def forward(self, appear, box):
        """appear [B,1536], box [B,7] -> embedding L2-norm [B,D]."""
        a = self.app_encoder(appear)
        g = self.box_encoder(box)
        e = self.fusion(torch.cat([a, g], dim=1))
        return F.normalize(e, p=2, dim=1)


def features_a_tensor(det, device):
    """De un dict de detección {feat_max, feat_avg, boxes} -> (appear [M,1536], box [M,7])."""
    fmax = torch.from_numpy(np.asarray(det["feat_max"], np.float32))
    favg = torch.from_numpy(np.asarray(det["feat_avg"], np.float32))
    appear = torch.cat([fmax, favg], dim=1).to(device)
    box = torch.from_numpy(np.asarray(det["boxes"], np.float32)).to(device)
    return appear, box


def batch_hard_triplet(emb_a, ids_a, emb_b, ids_b, margin=0.3):
    """Triplet con hard mining entre dos frames (t y t+1).

    Ancla en cada emb de A; positivo = mismo id en B; negativo más difícil =
    id distinto más cercano en B. Devuelve (loss, n_triplets).
    """
    if len(emb_a) == 0 or len(emb_b) == 0:
        return None, 0
    # distancias euclidianas A x B
    d = torch.cdist(emb_a, emb_b)                      # [Na, Nb]
    ids_a = ids_a.view(-1, 1)
    ids_b = ids_b.view(1, -1)
    same = ids_a == ids_b                              # [Na, Nb]
    losses = []
    for i in range(len(emb_a)):
        pos = same[i]
        neg = ~pos
        if pos.any() and neg.any():
            hardest_pos = d[i][pos].max()             # positivo más lejano
            hardest_neg = d[i][neg].min()             # negativo más cercano
            losses.append(F.relu(hardest_pos - hardest_neg + margin))
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)
