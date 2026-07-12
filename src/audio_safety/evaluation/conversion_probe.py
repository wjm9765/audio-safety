"""Run 4 Stage B (fast): matched text-vs-audio mechanism adjudication.

Pure numpy (CPU-testable). Given residual-stream activations for matched
harmful/benign items in both modalities, this decides which mechanism best
explains the audio>text safety gap:

  (i)   generic modality drift / calibration
  (ii)  perception / semantic degradation
  (iii) refusal under-writing / conversion failure   <- the target
  (iv)  modality-gated readout

It never forces a four-way call: MIXED / UNRESOLVED are first-class outcomes
(cross-check Round 5). Key measurements, per the cross-check:

- c_R (refusal) read at the decision position P2 via the FROZEN, causally
  validated axis r_A (out-of-sample by construction).
- c_H (harmfulness) read at a content / pre-assistant position via an
  ITEM-GROUPED CROSS-FITTED DIM, so preservation is not asserted by in-sample
  P2 construction.
- Specificity uses the benign-centered double difference G(u) and a
  variance-standardized (not isotropic) random-direction null; r_A-r_H overlap is
  reported, never assumed away.
- Block-level writer Δc_R(l) from consecutive post-block residual differences,
  with a telescoping check; no attention-vs-MLP inference (that is Stage C).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

TEXT = "text"
AUDIO = "audio"


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, eps)


def _z(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, sd) with a floored sd for stable z-scoring."""
    return float(np.mean(values)), max(float(np.std(values)), 1e-9)


def item_grouped_folds(item_ids: Sequence[str], k: int, seed: int) -> list[np.ndarray]:
    """Assign each unique item to one of k folds; return per-fold row-index arrays."""
    uniq = sorted(set(item_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % k for i in range(len(uniq))}
    folds: list[list[int]] = [[] for _ in range(k)]
    for row, item in enumerate(item_ids):
        folds[fold_of[item]].append(row)
    return [np.asarray(f, dtype=int) for f in folds]


def cross_fit_dim(
    hidden: np.ndarray,
    harmful_mask: np.ndarray,
    item_ids: Sequence[str],
    *,
    k: int,
    seed: int,
) -> np.ndarray:
    """Item-grouped cross-fitted DIM(harmful - benign) projection per row.

    For each fold, the harmful-minus-benign direction is estimated on the OTHER
    folds' items and applied to this fold's rows, so a row's score never uses its
    own item to build the direction (guards against in-sample inflation).
    """
    n = hidden.shape[0]
    scores = np.full(n, np.nan)
    folds = item_grouped_folds(item_ids, k, seed)
    all_idx = np.arange(n)
    for f in range(k):
        test = folds[f]
        if test.size == 0:
            continue
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        tr_h = train[harmful_mask[train]]
        tr_b = train[~harmful_mask[train]]
        if tr_h.size == 0 or tr_b.size == 0:
            continue
        direction = _unit(hidden[tr_h].mean(0) - hidden[tr_b].mean(0))
        scores[test] = hidden[test] @ direction
    return scores


def _index(meta: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    """(modality, safety) -> {item_id: row}."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for row, m in enumerate(meta):
        key = (str(m["modality"]), str(m["safety_label"]))
        item = str(m["item_id"])
        out.setdefault(key, {})[item] = row
    return out


def paired_by_item(
    scores: np.ndarray,
    meta: Sequence[dict[str, Any]],
    *,
    safety: str,
    field_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align per-item text and audio scores for one safety label."""
    idx = _index(meta)
    text_idx = idx.get((TEXT, safety), {})
    audio_idx = idx.get((AUDIO, safety), {})
    items = sorted(set(text_idx) & set(audio_idx))
    t, a, kept = [], [], []
    for item in items:
        rt, ra = text_idx[item], audio_idx[item]
        if np.isnan(scores[rt]) or np.isnan(scores[ra]):
            continue
        if field_valid is not None and not (field_valid[rt] and field_valid[ra]):
            continue
        t.append(scores[rt])
        a.append(scores[ra])
        kept.append(item)
    return np.asarray(t), np.asarray(a), kept


def paired_mean_diff_ci(
    a: np.ndarray, b: np.ndarray, *, n_boot: int, alpha: float, seed: int
) -> dict[str, float]:
    """Paired mean difference mean(a-b) with a by-item bootstrap CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"n": int(d.size), "mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def readout_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of ``scores`` for binary ``labels`` (both classes required)."""
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    ok = ~np.isnan(scores)
    scores, labels = scores[ok], labels[ok]
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def conditional_gap(hidden_cr: np.ndarray, axis: np.ndarray, meta: Sequence[dict[str, Any]]):
    """Benign-centered, itemwise text-minus-audio gap along ``axis`` at P2.

    ``G_i = (c_TH,i - c_TB,i) - (c_AH,i - c_AB,i)`` where ``c = <h, unit(axis)>``.
    Returns (per_item_G, item_ids). Only items present in all four cells count.
    """
    proj = hidden_cr @ _unit(axis)
    idx = _index(meta)
    cells = {
        "TH": idx.get((TEXT, "harmful"), {}),
        "TB": idx.get((TEXT, "benign"), {}),
        "AH": idx.get((AUDIO, "harmful"), {}),
        "AB": idx.get((AUDIO, "benign"), {}),
    }
    items = (
        sorted(set.intersection(*(set(c) for c in cells.values())))
        if all(cells.values())
        else []
    )
    g, kept = [], []
    for item in items:
        rows = {name: cell[item] for name, cell in cells.items()}
        if any(np.isnan(proj[r]) for r in rows.values()):
            continue
        g.append((proj[rows["TH"]] - proj[rows["TB"]]) - (proj[rows["AH"]] - proj[rows["AB"]]))
        kept.append(item)
    return np.asarray(g), kept


def variance_standardized_null(
    hidden_cr: np.ndarray,
    meta: Sequence[dict[str, Any]],
    *,
    n_directions: int,
    seed: int,
) -> np.ndarray:
    """Null distribution of |mean G(u)| along variance-standardized random u.

    Directions are drawn ~ N(0, diag(var)) then unit-normalized, so they respect
    the residual stream's marginal anisotropy instead of being an artificially
    easy isotropic null (cross-check Round 5, Q5).
    """
    std = hidden_cr.std(axis=0)
    rng = np.random.default_rng(seed)
    out = np.empty(n_directions)
    for t in range(n_directions):
        u = _unit(rng.standard_normal(hidden_cr.shape[1]) * std)
        g, _ = conditional_gap(hidden_cr, u, meta)
        out[t] = abs(float(g.mean())) if g.size else np.nan
    return out[~np.isnan(out)]


def block_writer_gap(
    cr_by_layer_text: np.ndarray,
    cr_by_layer_audio: np.ndarray,
    *,
    input_proj_text: np.ndarray | None = None,
    input_proj_audio: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-block write onto r_A, text minus audio (harmful items, mean over items).

    ``cr_by_layer`` rows are ``c_R(l) = <out(l) at P2, r_A>`` for l=0..L-1. Block
    l's write is ``Δc_R(l) = c_R(l) - c_R(l-1)`` (block 0 uses the embedding input
    projection when provided). Additivity of the residual stream makes this the
    exact combined (attention+MLP) write of block l; a telescoping residual is
    reported as a numerical check. No attention-vs-MLP split is inferred.
    """
    text_mean = cr_by_layer_text.mean(0)
    audio_mean = cr_by_layer_audio.mean(0)
    n_layers = text_mean.shape[0]

    def deltas(layer_means: np.ndarray, input_proj: np.ndarray | None) -> np.ndarray:
        prev = np.empty(n_layers)
        prev[0] = float(input_proj.mean()) if input_proj is not None else 0.0
        prev[1:] = layer_means[:-1]
        return layer_means - prev

    d_text = deltas(text_mean, input_proj_text)
    d_audio = deltas(audio_mean, input_proj_audio)
    # telescoping: sum of block writes should equal out(L-1) - input
    tel_text = float(d_text.sum() - (text_mean[-1] - (
        float(input_proj_text.mean()) if input_proj_text is not None else 0.0)))
    return {
        "delta_text": d_text.tolist(),
        "delta_audio": d_audio.tolist(),
        "delta_text_minus_audio": (d_text - d_audio).tolist(),
        "telescoping_residual": tel_text,
    }


def adjudicate_conversion(
    cr_hidden: np.ndarray,
    ch_hidden: np.ndarray,
    meta: Sequence[dict[str, Any]],
    r_a: np.ndarray,
    *,
    n_cross_fit_folds: int = 5,
    n_random_directions: int = 999,
    harmfulness_preserved_max_sd: float = 0.3,
    refusal_underdriven_min_sd: float = 0.3,
    specificity_min_ratio: float = 2.0,
    readout_min_auroc: float = 0.65,
    n_boot: int = 2000,
    ci_alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Adjudicate the four mechanisms; return status + quantified evidence.

    ``cr_hidden`` = raw residual at P2 (r_A's site); ``ch_hidden`` = raw residual
    at the content position (for cross-fitted r_H). ``meta`` rows carry
    ``item_id, modality, safety_label, behavior_label``.
    """
    item_ids = [str(m["item_id"]) for m in meta]
    modality = np.array([str(m["modality"]) for m in meta])
    safety = np.array([str(m["safety_label"]) for m in meta])
    behavior = np.array([str(m.get("behavior_label")) for m in meta])
    harmful_mask = safety == "harmful"

    # --- refusal coordinate c_R at P2 via frozen r_A (out-of-sample) ---
    c_r = cr_hidden @ _unit(r_a)

    # readout gate: does r_A separate refusal vs compliance within each modality?
    readout = {}
    for mod in (TEXT, AUDIO):
        sel = (modality == mod) & harmful_mask & np.isin(
            behavior, ["policy_refusal", "harmful_compliance"]
        )
        labels = (behavior[sel] == "policy_refusal").astype(int)
        readout[mod] = readout_auroc(c_r[sel], labels)
    readout_ok = all(
        (not np.isnan(readout[m])) and readout[m] >= readout_min_auroc for m in (TEXT, AUDIO)
    )

    # raw under-activation: harmful text vs audio c_R (paired), z-scored
    t_cr, a_cr, _ = paired_by_item(c_r, meta, safety="harmful")
    _, sd_cr = _z(c_r[harmful_mask])
    cr_diff = paired_mean_diff_ci(t_cr, a_cr, n_boot=n_boot, alpha=ci_alpha, seed=seed)
    d_R = cr_diff["mean"] / sd_cr if cr_diff["n"] else float("nan")

    # --- harmfulness c_H at content position via cross-fitted DIM ---
    c_h = cross_fit_dim(ch_hidden, harmful_mask, item_ids, k=n_cross_fit_folds, seed=seed)
    t_ch, a_ch, _ = paired_by_item(c_h, meta, safety="harmful")
    valid_ch = c_h[harmful_mask & ~np.isnan(c_h)]
    _, sd_ch = _z(valid_ch) if valid_ch.size else (0.0, 1.0)
    ch_diff = paired_mean_diff_ci(t_ch, a_ch, n_boot=n_boot, alpha=ci_alpha, seed=seed)
    d_H = ch_diff["mean"] / sd_ch if ch_diff["n"] else float("nan")
    # audio-native harmfulness readout: does cross-fit c_H separate harmful vs benign in audio?
    audio_sel = modality == AUDIO
    audio_native_auroc = readout_auroc(
        c_h[audio_sel], (safety[audio_sel] == "harmful").astype(int)
    )

    # --- specificity at P2: G along r_A vs r_H@P2 vs variance-standardized null ---
    g_ra, _ = conditional_gap(cr_hidden, r_a, meta)
    r_h_p2 = _unit(
        cr_hidden[harmful_mask].mean(0) - cr_hidden[~harmful_mask].mean(0)
    )
    g_rh, _ = conditional_gap(cr_hidden, r_h_p2, meta)
    null_abs = variance_standardized_null(
        cr_hidden, meta, n_directions=n_random_directions, seed=seed
    )
    g_ra_abs = abs(float(g_ra.mean())) if g_ra.size else float("nan")
    g_rh_abs = abs(float(g_rh.mean())) if g_rh.size else float("nan")
    null_95 = float(np.quantile(null_abs, 0.95)) if null_abs.size else float("nan")
    denom = max(g_rh_abs, null_95, 1e-9)
    specificity_ratio = g_ra_abs / denom if not np.isnan(g_ra_abs) else float("nan")
    g_ra_ci = paired_mean_diff_ci(
        g_ra, np.zeros_like(g_ra), n_boot=n_boot, alpha=ci_alpha, seed=seed
    ) if g_ra.size else {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    ra_rh_overlap = float(abs(np.dot(_unit(r_a), r_h_p2)))

    # --- soft adjudication (never forces a 4-way call) ---
    reasons: list[str] = []
    # Harmfulness is "preserved" when it is still linearly RECOVERABLE from the
    # audio states (cross-fit r_H separates harmful vs benign in audio); d_H is the
    # magnitude change, reported descriptively (cross-check Q2/Q6: recoverability is
    # the crisp signal, not a raw in-sample projection threshold).
    harmfulness_preserved = (not np.isnan(audio_native_auroc)) and (
        audio_native_auroc >= readout_min_auroc
    )
    if not readout_ok:
        status = "UNRESOLVED"
        reasons.append(
            f"r_A readout AUROC below {readout_min_auroc} in a modality "
            f"(text={readout[TEXT]:.2f}, audio={readout[AUDIO]:.2f}); projections not comparable"
        )
    elif not harmfulness_preserved:
        status = "PERCEPTION"
        reasons.append(
            f"harmfulness not linearly recoverable from audio "
            f"(audio-native r_H AUROC={audio_native_auroc:.2f} < {readout_min_auroc}, "
            f"d_H={d_H:.2f} SD) -> semantic/perception branch (F1)"
        )
    else:
        under_driven = (not np.isnan(d_R)) and d_R >= refusal_underdriven_min_sd
        survives_centering = (not np.isnan(g_ra_ci["ci_low"])) and g_ra_ci["ci_low"] > 0
        specific = (not np.isnan(specificity_ratio)) and specificity_ratio >= specificity_min_ratio
        if harmfulness_preserved and under_driven and survives_centering and specific:
            status = "CONVERSION"
            reasons.append(
                f"harmfulness preserved (d_H={d_H:.2f} SD), refusal under-driven "
                f"(d_R={d_R:.2f} SD), benign-centered gap survives (G_R CI "
                f"{g_ra_ci['ci_low']:.3f}..{g_ra_ci['ci_high']:.3f}), specific "
                f"(ratio={specificity_ratio:.2f})"
            )
        elif under_driven and not (survives_centering and specific):
            status = "DRIFT"
            reasons.append(
                f"raw c_R gap present (d_R={d_R:.2f} SD) but not benign-specific "
                f"(G_R survives={survives_centering}, specificity ratio={specificity_ratio:.2f}) "
                "-> generic modality drift / calibration"
            )
        elif harmfulness_preserved and not under_driven:
            status = "READOUT"
            reasons.append(
                f"harmfulness preserved and c_R not under-driven (d_R={d_R:.2f} SD); "
                "any behavioral gap points at modality-gated readout / orthogonal pressure (F2)"
            )
        else:
            status = "MIXED"
            reasons.append(
                f"no single mechanism dominates (d_H={d_H:.2f}, d_R={d_R:.2f}, "
                f"specificity={specificity_ratio:.2f})"
            )

    return {
        "status": status,
        "reasons": reasons,
        "note": (
            "fast direction-finding adjudication, not a pre-registered §0 verdict; "
            "clamp rescue (positive only) and component split are follow-up evidence"
        ),
        "readout_auroc": readout,
        "harmfulness": {
            "d_H_sd": d_H,
            "paired_diff": ch_diff,
            "audio_native_auroc": audio_native_auroc,
        },
        "refusal_underactivation": {"d_R_sd": d_R, "paired_diff": cr_diff},
        "specificity": {
            "G_rA_abs": g_ra_abs,
            "G_rH_p2_abs": g_rh_abs,
            "random_null_95": null_95,
            "specificity_ratio": specificity_ratio,
            "G_rA_ci": g_ra_ci,
            "rA_rH_overlap_cos": ra_rh_overlap,
        },
    }
