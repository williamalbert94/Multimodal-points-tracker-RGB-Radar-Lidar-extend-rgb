#!/usr/bin/env bash
# Ablacao da TAXA DE ATUALIZACAO DA CAIXA 3D (limite superior do rastreador).
#
# Isola quanto custa refrescar a caixa a taxa reduzida. A identidade e a
# segmentacao por ponto sao as do GT e permanecem a 10 Hz (a taxa nativa do
# View of Delft); so a caixa muda de taxa. Nos quadros sem caixa o objeto NAO
# desaparece — continua existindo pela segmentacao — e sua caixa e reconstruida
# dos pontos de radar atribuidos a ele (PCA + prior de classe).
#
# Como 0,15 s nao cai na grelha de 0,10 s, esse intervalo e obtido alternando
# saltos de 2 e 1 quadro (media 0,15 s = 6,7 Hz).
#
# Valores medidos (MOTA / HOTA / IDSW / ML):
#   10  Hz (0,10 s)  99.14 / 57.73 /  0 / 0.0     deslocamento do objeto: 0,37 m (19% do comprimento)
#   6,7 Hz (0,15 s)  56.84 / 46.04 / 12 / 0.0                             0,55 m (28%)
#   5   Hz (0,20 s)  33.20 / 41.13 / 19 / 2.6                             0,73 m (37%)
#
# Nota de reprodutibilidade: os valores acima foram medidos sobre os 1289
# quadros do split de validacao tal como exportados; este script usa as listas
# de clipes do proprio repo (1292 quadros, 3 a mais). As diferencas ficam na
# terceira casa decimal.
#
# Uso (dentro do contentor):  bash scripts/ablacao_taxa_caixa.sh
set -euo pipefail
cd /project
source scripts/comum.sh
preparar_ambiente

for MODO in 1 0.15 2; do
  case "$MODO" in
    1)    ETQ="10hz"  ;;
    0.15) ETQ="6_7hz" ;;
    2)    ETQ="5hz"   ;;
  esac
  PKL="${SAIDA}/det_caixa_${ETQ}.pkl"
  echo "===== caixa a ${ETQ} ====="
  $PY -u -m tracker.tracking.gerar_deteccoes_taxa_caixa "$PKL" "$MODO"
  $PY -u -m tracker.tracking.track_inference \
      --detections "$PKL" --gt-moving-only --cada 100000 \
      --out "${SAIDA}/track_caixa_${ETQ}"
  # HOTA sob o mesmo protocolo do RaTrack (IoU sobre conjuntos de pontos)
  if [ -f tools/run_hota.py ]; then
    printf "HOTA: "; $PY -u tools/run_hota.py proposto "${SAIDA}/track_caixa_${ETQ}" | tail -1
  fi
done

echo
echo "Metricas de cada taxa: ${SAIDA}/track_caixa_<taxa>/metrics.txt"
