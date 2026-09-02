#!/usr/bin/env bash
# Ablacao de FUSAO na segmentacao movel/estatico (tabela mIoU / IoU / F1).
#
# Cada linha e a mesma receita com UMA variavel trocada, avaliada no split de
# validacao (1288 quadros) com limiar 0.50. As metricas sao micro-medias:
# acumulam-se TP/FP/FN/TN sobre todo o split.
#
#   seg_exp_Q_lidarflow        LiDAR-Radar + MASG   <- configuracao completa
#   seg_exp_R_radaronly        Radar + MASG
#   seg_exp_S_nomasg           LiDAR-Radar, sem MASG
#   seg_exp_T_nomasg_radaronly Radar, sem MASG
#
# Valores publicados (mIoU / IoU_movel / IoU_estatico / F1):
#   Q 0.773 / 0.574 / 0.973 / 0.729      R 0.721 / 0.479 / 0.964 / 0.648
#   S 0.741 / 0.518 / 0.965 / 0.683      T 0.705 / 0.448 / 0.961 / 0.619
#
# Uso (dentro do contentor):  bash scripts/ablacao_fusao_segmentacao.sh
set -euo pipefail
cd /project
source scripts/comum.sh
preparar_ambiente

for EXP in seg_exp_Q_lidarflow seg_exp_R_radaronly seg_exp_S_nomasg seg_exp_T_nomasg_radaronly; do
  CK="tracker/checkpoints/${EXP}/best_miou_model.pth"
  if [ ! -f "$CK" ]; then
    echo "[pular] sem checkpoint para ${EXP} (${CK})"
    continue
  fi
  echo "===== ${EXP} ====="
  $PY -u -m tracker.runner.inference_seg \
      --config "tracker/config/${EXP}.yaml" \
      --checkpoint "$CK" \
      --output "${SAIDA}" \
      --umbral 0.5 --cada 100000
done

echo
echo "Metricas de cada variante: ${SAIDA}/<exp>/metrics.txt (coluna 'limiar 0.50')."
echo "Valores reproduzidos: Q 0.7734 | R 0.7211 | S 0.7414 | T 0.7048 (mIoU)."
