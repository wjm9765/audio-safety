"""Audio-RDO refusal-axis utilities.

The numpy functions implement the gate analysis and stay CPU-testable. The RDO
optimizer at the bottom imports torch lazily and is only used on the GPU server.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from audio_safety.config.schema import AudioRdoConfig
from audio_safety.models.hooks import ResidualStreamIntervention

VALID_LABELS = {
    "policy_refusal",
    "harmful_compliance",
    "benign_answer",
    "decoding_failure",
}


def unit_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("cannot normalize near-zero vector")
    return vector / norm


def signed_occupancy(hidden: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Occ = <h, unit(axis)> for one vector or rows of vectors."""
    return hidden @ unit_vector(axis)


def projection_ablate(hidden: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Remove the signed coordinate along axis from hidden state(s)."""
    axis_u = unit_vector(axis)
    coord = signed_occupancy(hidden, axis_u)
    return hidden - np.expand_dims(coord, axis=-1) * axis_u


def coordinate_restore(
    hidden_style: np.ndarray,
    hidden_neutral: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Restore only the axis coordinate of styled hidden states to neutral."""
    if hidden_style.shape != hidden_neutral.shape:
        raise ValueError(
            f"hidden shape mismatch: styled {hidden_style.shape} vs neutral {hidden_neutral.shape}"
        )
    axis_u = unit_vector(axis)
    current = signed_occupancy(hidden_style, axis_u)
    target = signed_occupancy(hidden_neutral, axis_u)
    return hidden_style + np.expand_dims(target - current, axis=-1) * axis_u


def dim_compliance_to_refusal(refused: np.ndarray, complied: np.ndarray) -> np.ndarray:
    """SARSteer MDSteer-c2r baseline: mean(refused) - mean(complied)."""
    if refused.ndim != 2 or complied.ndim != 2:
        raise ValueError("refused and complied activations must be 2D arrays")
    return unit_vector(refused.mean(axis=0) - complied.mean(axis=0))


def sar_text_refusal_vector(prompt_refusal: np.ndarray, prompt_base: np.ndarray) -> np.ndarray:
    """SARSteer-style text-derived vector: mean(h(text+p_R) - h(text))."""
    if prompt_refusal.shape != prompt_base.shape:
        raise ValueError(
            f"shape mismatch: refusal {prompt_refusal.shape} vs base {prompt_base.shape}"
        )
    return unit_vector((prompt_refusal - prompt_base).mean(axis=0))


def benign_controlled_safety_shift(
    harmful_style: np.ndarray,
    harmful_neutral: np.ndarray,
    benign_style: np.ndarray,
    benign_neutral: np.ndarray,
) -> np.ndarray:
    """Difference-in-differences style shift with benign acoustic confounds removed."""
    shapes = {
        harmful_style.shape,
        harmful_neutral.shape,
        benign_style.shape,
        benign_neutral.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            "all hidden arrays must share shape, got "
            f"{harmful_style.shape}, {harmful_neutral.shape}, "
            f"{benign_style.shape}, {benign_neutral.shape}"
        )
    return (harmful_style - harmful_neutral) - (benign_style - benign_neutral)


def escape_scores(
    harmful_style: np.ndarray,
    harmful_neutral: np.ndarray,
    benign_style: np.ndarray,
    benign_neutral: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Escape = -<Delta_safety, unit(axis)>; larger means lower refusal occupancy."""
    delta = benign_controlled_safety_shift(
        harmful_style,
        harmful_neutral,
        benign_style,
        benign_neutral,
    )
    return -signed_occupancy(delta, axis)


def behavior_counts(labels: Sequence[str]) -> dict[str, int]:
    counts = {label: 0 for label in sorted(VALID_LABELS)}
    for label in labels:
        if label not in VALID_LABELS:
            raise ValueError(f"unknown behavior label {label!r}")
        counts[label] += 1
    return counts


def refusal_rate(labels: Sequence[str], *, exclude_decoding_failure: bool = True) -> float:
    denom = 0
    refusals = 0
    for label in labels:
        if label not in VALID_LABELS:
            raise ValueError(f"unknown behavior label {label!r}")
        if exclude_decoding_failure and label == "decoding_failure":
            continue
        denom += 1
        refusals += int(label == "policy_refusal")
    return float(refusals / denom) if denom else float("nan")


def harmful_compliance_rate(
    labels: Sequence[str],
    *,
    exclude_decoding_failure: bool = True,
) -> float:
    denom = 0
    complied = 0
    for label in labels:
        if label not in VALID_LABELS:
            raise ValueError(f"unknown behavior label {label!r}")
        if exclude_decoding_failure and label == "decoding_failure":
            continue
        denom += 1
        complied += int(label == "harmful_compliance")
    return float(complied / denom) if denom else float("nan")


def percentage_point_delta(after: float, before: float) -> float:
    return 100.0 * (after - before)


@dataclass(frozen=True)
class RdoTrainingBatch:
    """Prepared teacher-forced batch for RDO optimization.

    ``*_inputs`` are tensors/BatchEncoding objects already containing the prompt
    plus target continuation where CE is computed. Labels should be -100 outside
    the continuation span.
    """

    add_inputs: Mapping[str, Any]
    add_labels: Any
    add_token_index: int
    ablate_inputs: Mapping[str, Any] | None = None
    ablate_labels: Any | None = None
    ablate_token_index: int | None = None
    retain_inputs: Mapping[str, Any] | None = None
    retain_token_index: int | None = None


def make_continuation_labels(input_ids: Any, prompt_length: int, ignore_index: int = -100) -> Any:
    """Create CE labels for target continuation tokens only."""
    labels = input_ids.clone()
    labels[:, :prompt_length] = ignore_index
    return labels


def _hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for node in (config, text_config):
        if node is not None and getattr(node, "hidden_size", None) is not None:
            return int(node.hidden_size)
    raise AttributeError("could not locate hidden_size on model.config")


def _input_device(model: Any) -> Any:
    from audio_safety.models.qwen2_audio import model_input_device

    return model_input_device(model)


def _with_labels(inputs: Mapping[str, Any], labels: Any) -> dict[str, Any]:
    data = dict(inputs)
    data["labels"] = labels
    return data


def _kl_retain_loss(base_logits: Any, steered_logits: Any) -> Any:
    import torch.nn.functional as F

    return F.kl_div(
        F.log_softmax(steered_logits.float(), dim=-1),
        F.softmax(base_logits.float(), dim=-1),
        reduction="batchmean",
    )


def _logits_at_token(logits: Any, token_index: int) -> Any:
    seq_len = int(logits.shape[1])
    if token_index < 0:
        token_index = seq_len + token_index
    if token_index < 0 or token_index >= seq_len:
        raise IndexError(f"token_index {token_index} is outside sequence length {seq_len}")
    return logits[:, token_index, :]


def train_audio_rdo_axis(
    model: Any,
    batches: Sequence[RdoTrainingBatch],
    *,
    layer_idx: int,
    cfg: AudioRdoConfig,
) -> np.ndarray:
    """Optimize one audio-conditioned refusal axis at a fixed layer/position.

    This is deliberately low-level: dataset code prepares teacher-forced inputs
    and labels, while this function only owns the RDO objective.
    """
    import torch
    from tqdm.auto import trange

    if not batches:
        raise ValueError("at least one RDO training batch is required")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    device = _input_device(model)
    r = torch.nn.Parameter(torch.randn(_hidden_size(model), device=device))
    optimizer = torch.optim.Adam([r], lr=cfg.learning_rate)
    loss_scale = 1.0 / len(batches)

    for _step in trange(cfg.train_steps, desc=f"RDO train L{layer_idx}", unit="step", leave=False):
        optimizer.zero_grad(set_to_none=True)

        for batch in batches:
            r_unit = r / torch.clamp(torch.linalg.vector_norm(r), min=1e-12)
            total = torch.zeros((), device=device)
            add_inputs = _with_labels(batch.add_inputs, batch.add_labels)
            with ResidualStreamIntervention(
                model,
                layer_idx=layer_idx,
                token_index=batch.add_token_index,
                vector=r_unit,
                mode="add",
                scale=cfg.alpha,
            ):
                add_loss = model(**add_inputs).loss
            total = total + cfg.loss_weights.add * add_loss

            if (
                batch.ablate_inputs is not None
                and batch.ablate_labels is not None
                and batch.ablate_token_index is not None
            ):
                ablate_inputs = _with_labels(batch.ablate_inputs, batch.ablate_labels)
                with ResidualStreamIntervention(
                    model,
                    layer_idx=layer_idx,
                    token_index=batch.ablate_token_index,
                    vector=r_unit,
                    mode="ablate",
                ):
                    ablate_loss = model(**ablate_inputs).loss
                total = total + cfg.loss_weights.ablate * ablate_loss

            if batch.retain_inputs is not None and batch.retain_token_index is not None:
                with torch.no_grad():
                    base_logits = _logits_at_token(
                        model(**batch.retain_inputs).logits,
                        batch.retain_token_index,
                    ).detach()
                with ResidualStreamIntervention(
                    model,
                    layer_idx=layer_idx,
                    token_index=batch.retain_token_index,
                    vector=r_unit,
                    mode="add",
                    scale=cfg.alpha,
                ):
                    steered_logits = _logits_at_token(
                        model(**batch.retain_inputs).logits,
                        batch.retain_token_index,
                    )
                total = total + cfg.loss_weights.retain * _kl_retain_loss(
                    base_logits,
                    steered_logits,
                )

            (loss_scale * total).backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    r_final = r / torch.clamp(torch.linalg.vector_norm(r), min=1e-12)
    return r_final.detach().float().cpu().numpy()
