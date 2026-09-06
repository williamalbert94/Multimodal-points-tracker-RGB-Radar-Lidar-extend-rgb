#!/usr/bin/env bash
# Deteccoes "caixa GT + filtro por segmentacao" — a via usada nas ablacoes.
#
# Mantem a caixa GT do objeto e so decide se a conserva, segundo o backbone
# encontrar ao menos `--min-pts` pontos de radar moveis dentro dela. E um proxy
# de detector: a localizacao e perfeita e o recall e realista (so entra o que o
# radar de facto percebe). NAO e um detector 3D — serve como limite superior.
#
# `rider` fica de fora por padrao: VoD anota cada ciclista duas vezes (uma caixa
# `Cyclist` e outra `rider`, praticamente sobrepostas), o que duplicaria o
# objeto no GT, nas deteccoes e nas metricas.
#
# Uso (dentro do contentor):  bash scripts/gerar_deteccoes_gtseg.sh
set -euo pipefail
cd /project
source scripts/comum.sh
preparar_ambiente
exigir_checkpoint

$PY -u -m tracker.tracking.precompute_detections_gtseg \
    --config "$CFG" --checkpoint "$CKPT" \
    --umbral 0.5 --min-pts 1 --moving-only --split val \
    --out "${SAIDA}/detections_gtseg_val_mov_norider.pkl"

echo
echo "Deteccoes em ${SAIDA}/detections_gtseg_val_mov_norider.pkl"
