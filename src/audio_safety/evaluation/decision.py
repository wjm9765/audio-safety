"""Pre-registered GO / NO-GO / AMBIGUOUS decision rule (design.md §0).

The thresholds come from DecisionConfig, which mirrors the pre-registered table.
This function is intentionally the ONLY place the decision is computed — analysis
code and skills must call it rather than re-deriving the verdict.
"""

from dataclasses import dataclass, field

from audio_safety.config.schema import DecisionConfig


@dataclass(frozen=True)
class DecisionResult:
    status: str  # "GO" | "NO-GO" | "AMBIGUOUS"
    reasons: list[str] = field(default_factory=list)


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
