"""HOTA (Luiten et al., IJCV 2021) para o protocolo interno deste trabalho.

Segue TrackEval: para cada limiar de localização alfa,
    DetA(a) = |TP| / (|TP| + |FN| + |FP|)
    AssA(a) = (1/|TP|) * soma_{c in TP} |TPA(c)| / (|TPA(c)|+|FNA(c)|+|FPA(c)|)
    HOTA(a) = sqrt(DetA(a) * AssA(a))
e HOTA é a média sobre alfa em {0.05, 0.10, ..., 0.95}.

A associação por quadro usa o mesmo esquema de duas passadas do TrackEval: uma
primeira passada acumula a contagem de co-ocorrências entre cada par
(id_gt, id_pred) e a segunda resolve o emparelhamento maximizando
`similaridade * pontuação_de_alinhamento_global`, o que torna o resultado
invariante à ordem dos quadros.

Similaridade: IoU BEV rotacionada, a mesma de `mot_metrics`.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)


def hota(frames, sim_fn):
    """frames: lista de (gt_ids, pred_ids, gt_boxes, pred_boxes) por quadro.
    sim_fn(gt_boxes, pred_boxes) -> matriz [n_gt, n_pred] de similaridade em [0,1].
    Devolve dict com HOTA, DetA, AssA (médias sobre alfa) e as curvas por alfa."""
    gt_ids_all = sorted({int(i) for f in frames for i in f[0]})
    pr_ids_all = sorted({int(i) for f in frames for i in f[1]})
    gmap = {g: k for k, g in enumerate(gt_ids_all)}
    pmap = {p: k for k, p in enumerate(pr_ids_all)}
    nG, nP = len(gmap), len(pmap)
    if nG == 0 or nP == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}

    sims = []
    pot = np.zeros((nG, nP))           # co-ocorrências ponderadas (passada 1)
    gt_count = np.zeros(nG)
    pr_count = np.zeros(nP)
    for gi, pi, gb, pb in frames:
        gi = [gmap[int(x)] for x in gi]
        pi = [pmap[int(x)] for x in pi]
        s = sim_fn(gb, pb) if len(gi) and len(pi) else np.zeros((len(gi), len(pi)))
        sims.append((gi, pi, s))
        for g in gi:
            gt_count[g] += 1
        for p in pi:
            pr_count[p] += 1
        for a, g in enumerate(gi):
            for b, p in enumerate(pi):
                pot[g, p] += s[a, b]

    # pontuação de alinhamento global (passada 2)
    denom = gt_count[:, None] + pr_count[None, :] - pot
    align = np.divide(pot, denom, out=np.zeros_like(pot), where=denom > 0)

    out = {"alphas": [], "DetA": [], "AssA": [], "HOTA": []}
    for a in ALPHAS:
        TPA = np.zeros((nG, nP)); TP = 0
        for gi, pi, s in sims:
            if not len(gi) or not len(pi):
                continue
            score = s * align[np.ix_(gi, pi)]
            score[s < a - 1e-9] = 0.0
            r, c = linear_sum_assignment(-score)
            for x, y in zip(r, c):
                if s[x, y] >= a - 1e-9 and score[x, y] > 0:
                    TPA[gi[x], pi[y]] += 1
                    TP += 1
        FN = int(gt_count.sum()) - TP
        FP = int(pr_count.sum()) - TP
        det = TP / (TP + FN + FP) if (TP + FN + FP) else 0.0
        if TP:
            g_tot = TPA.sum(axis=1, keepdims=True)      # todas as det. desse gt id
            p_tot = TPA.sum(axis=0, keepdims=True)
            den = gt_count[:, None] + pr_count[None, :] - TPA
            with np.errstate(divide="ignore", invalid="ignore"):
                A = np.where(den > 0, TPA / den, 0.0)
            ass = float((A * TPA).sum() / TP)
        else:
            ass = 0.0
        out["alphas"].append(float(a)); out["DetA"].append(det)
        out["AssA"].append(ass); out["HOTA"].append(float(np.sqrt(det * ass)))
    for k in ("DetA", "AssA", "HOTA"):
        out[k + "_mean"] = float(np.mean(out[k]))
    return out
