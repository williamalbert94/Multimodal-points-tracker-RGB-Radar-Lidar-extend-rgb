#!/usr/bin/env bash
# Definicoes compartilhadas pelos scripts de ablacao.
#
# Todos rodam DENTRO do contentor. A imagem nao traz a extensao CUDA do
# PointNet++ compilada nem o torch/lib no caminho do linker, entao as duas
# variaveis abaixo sao obrigatorias — sem elas o import de `pointnet2_cuda`
# falha com ModuleNotFoundError ou com "libc10.so: cannot open shared object".
set -euo pipefail

CFG=${CFG:-tracker/config/seg_exp_Q_lidarflow.yaml}
CKPT=${CKPT:-tracker/checkpoints/seg_exp_Q_lidarflow/best_miou_model.pth}
SAIDA=${SAIDA:-tracker/results}
PY=/opt/conda/envs/mira/bin/python

preparar_ambiente() {
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh && conda activate mira
  export LD_LIBRARY_PATH=/opt/conda/envs/mira/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=/project/external/lib/build/lib.linux-x86_64-cpython-39:${PYTHONPATH:-}
  if [ ! -d /project/external/lib/build ]; then
    echo "[aviso] extensao pointnet2 nao compilada; compilando agora..."
    (cd /project/external/lib && python setup.py install >/dev/null)
  fi
}

exigir_checkpoint() {
  if [ ! -f "$CKPT" ]; then
    echo "[erro] falta o checkpoint: $CKPT" >&2
    echo "       baixe os pesos indicados no README (secao Prerequisites)." >&2
    exit 1
  fi
}
