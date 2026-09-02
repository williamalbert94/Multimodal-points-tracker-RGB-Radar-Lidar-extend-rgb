#!/usr/bin/env bash
# Ablacao leave-one-out das PISTAS de associacao do GalleryTracker.
#
# O custo de associacao combina cinco pistas com pesos que o rastreador
# renormaliza para 1. Zerar uma de cada vez mede o quanto ela contribui.
# Ordem de --weights: APARENCIA GEOMETRIA DENSIDADE MOVIMENTO ESPACIAL.
#
# Deteccoes: proxy "caixa GT + filtro por segmentacao" (so objetos moveis, sem
# `rider`), de modo que o que varia entre linhas e SO a associacao.
#
# Valores medidos (MOTA / IDSW):
#   completa      79.18 / 12      sem aparencia  79.53 /  5
#   sem densidade 79.18 / 12      sem geometria  78.87 / 18
#   sem movimento 78.67 / 22      sem espacial   79.07 / 14
#
# Uso (dentro do contentor):  bash scripts/ablacao_pistas_rastreador.sh
set -euo pipefail
cd /project
source scripts/comum.sh
preparar_ambiente

DETS=${DETS:-${SAIDA}/detections_gtseg_val_mov_norider.pkl}
if [ ! -f "$DETS" ]; then
  echo "[erro] faltam as deteccoes: $DETS" >&2
  echo "       gere-as com scripts/gerar_deteccoes_gtseg.sh" >&2
  exit 1
fi

rodar() {  # $1 = nome, $2..$6 = pesos
  local nome=$1; shift
  echo "===== ${nome} ====="
  $PY -u -m tracker.tracking.track_inference \
      --detections "$DETS" --gt-moving-only --cada 100000 \
      --weights "$@" --out "${SAIDA}/abl_${nome}"
}

#        nome            APAR GEOM DENS MOVI ESPA
rodar completa           0.30 0.20 0.10 0.20 0.20
rodar sem_aparencia      0.00 0.20 0.10 0.20 0.20
rodar sem_geometria      0.30 0.00 0.10 0.20 0.20
rodar sem_densidade      0.30 0.20 0.00 0.20 0.20
rodar sem_movimento      0.30 0.20 0.10 0.00 0.20
rodar sem_espacial       0.30 0.20 0.10 0.20 0.00

echo
echo "Metricas de cada variante: ${SAIDA}/abl_<nome>/metrics.txt"
