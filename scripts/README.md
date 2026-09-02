# Scripts de ablacao

Reproduzem as tabelas do documento. Todos rodam **dentro do contentor**, a
partir de `/project`, e escrevem em `tracker/results/` (ignorado pelo git).

```bash
docker compose -f docker/docker-compose.yml run --rm tracker_multimodal_mira \
    -lc "bash scripts/<script>.sh"
```

| script | tabela que reproduz |
|---|---|
| `gerar_deteccoes_gtseg.sh` | pre-requisito das duas ablacoes de rastreamento |
| `ablacao_fusao_segmentacao.sh` | mIoU / IoU movel / IoU estatico / F1 por configuracao de fusao |
| `ablacao_pistas_rastreador.sh` | sAMOTA / MOTA / IDSW por pista de associacao (leave-one-out) |
| `ablacao_taxa_caixa.sh` | limite superior do rastreador com a caixa 3D a 10; 6,7 e 5 Hz |

## Pre-requisitos

- **Pesos**: `tracker/checkpoints/<exp>/best_miou_model.pth` (ver README principal).
- **Dataset**: View of Delft em `/project/view_of_delft_PUBLIC`.
- **Extensao CUDA**: `comum.sh` compila `external/lib` se ainda nao estiver, e
  poe `torch/lib` no `LD_LIBRARY_PATH` — sem isso o import de `pointnet2_cuda`
  falha.

## Observacao sobre as metricas

O `sAMOTA` que `track_inference.py` imprime e de **um unico ponto de operacao**.
O valor comparavel ao de Pan et al. vem do varrimento de recall do AB3DMOT
(`tracker/tracking/amota_ab3dmot.py`). Nao misture os dois numa mesma coluna.
