"""Pre-registered GO / NO-GO / AMBIGUOUS decision rules (design.md §0).

The thresholds come from config schemas that mirror the pre-registered tables.
Analysis code and skills should call these functions rather than re-deriving a
verdict.
"""

from dataclasses import dataclass, field

from audio_safety.config.schema import AudioRdoDecisionConfig, DecisionConfig


@dataclass(frozen=True)
class DecisionResult:
    status: str  # "GO" | "WEAK-GO" | "NO-GO" | "AMBIGUOUS"
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AudioRdoGateMetrics:
    """Heldout metrics for the Audio-RDO gate.

    Fields ending in ``_pp`` are percentage points. Optional fields can be left
    unset while a stage is incomplete; the decision will then be AMBIGUOUS unless
    a hard NO-GO condition is already visible.
    """

    genuine_style_gap_pp: float | None = None
    decoding_failure_share: float | None = None
    add_rr_pp: float | None = None
    benign_orr_add_pp: float | None = None
    ablation_asr_pp: float | None = None
    rdo_beats_mdsteer_c2r: bool | None = None
    rdo_beats_sarsteer_text: bool | None = None
    escape_spearman: float | None = None
    escape_auroc: float | None = None
    restoration_rr_pp: float | None = None
    restored_fraction: float | None = None
    benign_orr_restore_pp: float | None = None


def decide(
    mean_pairwise_cosine: float,
    permutation_p: float,
    dominant_axis_disagrees: bool,
    cfg: DecisionConfig,
    causal_axes_validated: bool | None = None,
) -> DecisionResult:
    """Apply the pre-registered decision table.

    GO (H1)   : mpc < go_max_cosine AND dominant axis differs across >=2 families
                AND permutation p < p_threshold AND cone axes passed causal ablation.
    NO-GO (H0): mpc >= nogo_min_cosine (single-axis collapse).
    AMBIGUOUS : everything else.

    ``causal_axes_validated=None`` means ablation hasn't been run; a GO is then
    downgraded to AMBIGUOUS with an explicit reason (never silently granted).
    """
    reasons: list[str] = []

    if mean_pairwise_cosine >= cfg.nogo_min_cosine:
        reasons.append(
            f"mean pairwise cosine {mean_pairwise_cosine:.3f} >= "
            f"{cfg.nogo_min_cosine} (single-axis collapse)"
        )
        return DecisionResult("NO-GO", reasons)

    go_conditions = {
        f"mpc {mean_pairwise_cosine:.3f} < {cfg.go_max_cosine}": (
            mean_pairwise_cosine < cfg.go_max_cosine
        ),
        "dominant axis differs across >=2 families": dominant_axis_disagrees,
        f"permutation p {permutation_p:.4f} < {cfg.p_threshold}": (permutation_p < cfg.p_threshold),
        "cone axes passed causal ablation": causal_axes_validated is True,
    }

    if all(go_conditions.values()):
        reasons.extend(f"PASS: {name}" for name in go_conditions)
        return DecisionResult("GO", reasons)

    for name, ok in go_conditions.items():
        reasons.append(f"{'PASS' if ok else 'FAIL'}: {name}")
    if causal_axes_validated is None:
        reasons.append("NOTE: causal ablation not yet run — GO cannot be granted")
    return DecisionResult("AMBIGUOUS", reasons)


def _missing(name: str, value: object | None, reasons: list[str]) -> bool:
    if value is None:
        reasons.append(f"MISSING: {name}")
        return True
    return False


def decide_audio_rdo_gate(
    metrics: AudioRdoGateMetrics,
    cfg: AudioRdoDecisionConfig,
    *,
    max_decoding_failure_share: float,
) -> DecisionResult:
    """Apply the Audio-RDO gate from design.md §0.

    Strong GO requires behavior validity, axis validity, baseline win, style
    mediation, and restoration. Weak GO is returned when the RDO axis itself is
    validated and beats baselines but style/restoration evidence is incomplete or
    below threshold.
    """
    reasons: list[str] = []

    if (
        metrics.decoding_failure_share is not None
        and metrics.decoding_failure_share > max_decoding_failure_share
    ):
        reasons.append(
            f"decoding failure share {metrics.decoding_failure_share:.3f} > "
            f"{max_decoding_failure_share:.3f}"
        )
        return DecisionResult("NO-GO", reasons)

    if metrics.add_rr_pp is not None and metrics.add_rr_pp < cfg.min_add_rr_pp:
        reasons.append(f"addition RR +{metrics.add_rr_pp:.1f}pp < {cfg.min_add_rr_pp:.1f}pp")
        return DecisionResult("NO-GO", reasons)

    if (
        metrics.benign_orr_add_pp is not None
        and metrics.benign_orr_add_pp > cfg.max_benign_orr_pp
    ):
        reasons.append(
            f"addition benign ORR +{metrics.benign_orr_add_pp:.1f}pp > "
            f"{cfg.max_benign_orr_pp:.1f}pp"
        )
        return DecisionResult("NO-GO", reasons)

    if (
        metrics.ablation_asr_pp is not None
        and metrics.ablation_asr_pp < cfg.min_ablation_asr_pp
    ):
        reasons.append(
            f"ablation ASR +{metrics.ablation_asr_pp:.1f}pp < "
            f"{cfg.min_ablation_asr_pp:.1f}pp"
        )
        return DecisionResult("NO-GO", reasons)

    if metrics.rdo_beats_mdsteer_c2r is False:
        reasons.append("RDO axis does not beat MDSteer-c2r at matched ORR")
        return DecisionResult("NO-GO", reasons)

    if metrics.rdo_beats_sarsteer_text is False:
        reasons.append("SARSteer-style text vector beats or matches RDO at matched ORR")
        return DecisionResult("NO-GO", reasons)

    missing = False
    missing |= _missing("genuine_style_gap_pp", metrics.genuine_style_gap_pp, reasons)
    missing |= _missing("decoding_failure_share", metrics.decoding_failure_share, reasons)
    missing |= _missing("add_rr_pp", metrics.add_rr_pp, reasons)
    missing |= _missing("benign_orr_add_pp", metrics.benign_orr_add_pp, reasons)
    missing |= _missing("ablation_asr_pp", metrics.ablation_asr_pp, reasons)
    missing |= _missing("rdo_beats_mdsteer_c2r", metrics.rdo_beats_mdsteer_c2r, reasons)
    missing |= _missing("rdo_beats_sarsteer_text", metrics.rdo_beats_sarsteer_text, reasons)

    if missing:
        return DecisionResult("AMBIGUOUS", reasons)

    axis_valid = (
        metrics.add_rr_pp >= cfg.min_add_rr_pp
        and metrics.benign_orr_add_pp <= cfg.max_benign_orr_pp
        and metrics.ablation_asr_pp >= cfg.min_ablation_asr_pp
    )
    baseline_win = metrics.rdo_beats_mdsteer_c2r and metrics.rdo_beats_sarsteer_text
    behavior_valid = metrics.genuine_style_gap_pp >= cfg.min_genuine_style_gap_pp

    style_valid = (
        (metrics.escape_spearman is not None and metrics.escape_spearman >= cfg.min_escape_spearman)
        or (metrics.escape_auroc is not None and metrics.escape_auroc >= cfg.min_escape_auroc)
    )
    restoration_valid = (
        (
            metrics.restoration_rr_pp is not None
            and metrics.restoration_rr_pp >= cfg.min_restoration_rr_pp
        )
        or (
            metrics.restored_fraction is not None
            and metrics.restored_fraction >= cfg.min_restored_fraction
        )
    ) and (
        metrics.benign_orr_restore_pp is not None
        and metrics.benign_orr_restore_pp <= cfg.max_benign_orr_pp
    )

    if axis_valid and baseline_win and behavior_valid and style_valid and restoration_valid:
        reasons.extend(
            [
                "PASS: genuine style effect remains after decoding failures are excluded",
                "PASS: RDO axis passes addition / ablation / benign retention",
                "PASS: RDO beats MDSteer-c2r and SARSteer-style baselines",
                "PASS: benign-controlled escape predicts harmful compliance",
                "PASS: axis-coordinate restoration recovers refusal without benign ORR",
            ]
        )
        return DecisionResult("GO", reasons)

    if axis_valid and baseline_win:
        reasons.append(
            "WEAK GO: RDO axis is validated, but style mediation/restoration is incomplete"
        )
        if not behavior_valid:
            reasons.append(
                f"style gap {metrics.genuine_style_gap_pp:.1f}pp < "
                f"{cfg.min_genuine_style_gap_pp:.1f}pp"
            )
        if not style_valid:
            reasons.append("escape correlation/AUROC below pilot threshold or missing")
        if not restoration_valid:
            reasons.append("coordinate restoration below threshold or benign ORR missing")
        return DecisionResult("WEAK-GO", reasons)

    reasons.append("Audio-RDO gate conditions are not jointly satisfied")
    return DecisionResult("AMBIGUOUS", reasons)
