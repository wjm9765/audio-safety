"""CPU-only utilities for explicit-refusal instability analyses.

This module deliberately measures the presence of caller-defined, explicit
refusal language.  ``explicit_refusal_absent`` means only that none of those
patterns matched; it must not be interpreted as acceptance, task completion, or
harmful compliance.

The paired statistics treat the semantic item as the sampling unit.  Decoding
failures are supplied by the caller, kept auditable, and excluded from the valid
paired denominator rather than being silently folded into either refusal state.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Hashable, Mapping, Sequence
from math import comb
from typing import Any

import numpy as np

EXPLICIT_REFUSAL = "explicit_refusal"
EXPLICIT_REFUSAL_ABSENT = "explicit_refusal_absent"
VALID_REFUSAL_STATUSES = frozenset({EXPLICIT_REFUSAL, EXPLICIT_REFUSAL_ABSENT})

STABLE_REFUSAL = "stable_refusal"
REFUSAL_TO_NONREFUSAL = "refusal_to_nonrefusal"
NONREFUSAL_TO_REFUSAL = "nonrefusal_to_refusal"
STABLE_NONREFUSAL = "stable_nonrefusal"

_APOSTROPHES = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201b": "'",  # single high-reversed-9 quotation mark
        "\u02bc": "'",  # modifier letter apostrophe
        "\u02bb": "'",  # modifier letter turned comma
        "\uff07": "'",  # full-width apostrophe
        "`": "'",
        "\u00b4": "'",  # acute accent commonly used as an apostrophe
    }
)


def normalize_refusal_text(text: str) -> str:
    """Normalize text for literal refusal-pattern matching.

    Matching is case-insensitive, whitespace-insensitive, and treats common
    Unicode apostrophes as ASCII apostrophes.  Other punctuation is preserved so
    the caller remains in control of how broad each literal pattern is.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _normalized_patterns(patterns: Sequence[str]) -> tuple[str, ...]:
    if isinstance(patterns, (str, bytes)):
        raise TypeError("patterns must be a sequence of strings, not a single string")
    normalized: list[str] = []
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, str):
            raise TypeError(f"patterns[{index}] must be a string")
        value = normalize_refusal_text(pattern)
        if not value:
            raise ValueError(f"patterns[{index}] is empty after normalization")
        normalized.append(value)
    if not normalized:
        raise ValueError("patterns must contain at least one non-empty string")
    # Preserve declaration order for reproducible matching while avoiding duplicate work.
    return tuple(dict.fromkeys(normalized))


def _validate_failure_status(status: str) -> str:
    if not isinstance(status, str) or not status.strip():
        raise ValueError("a decoding failure status must be a non-empty string")
    if status in VALID_REFUSAL_STATUSES:
        raise ValueError(
            "a decoding failure status must differ from the two valid refusal statuses"
        )
    return status


def classify_explicit_refusal(
    output: str,
    patterns: Sequence[str],
    *,
    decoding_failure_status: str | None = None,
) -> str:
    """Classify one output using caller-supplied literal refusal patterns.

    If the caller has already diagnosed a decoding failure, pass its non-empty
    status through ``decoding_failure_status``.  It takes precedence over text
    matching and is returned unchanged, preserving the caller's failure taxonomy.
    Otherwise this function returns exactly ``explicit_refusal`` or
    ``explicit_refusal_absent``.

    ``explicit_refusal_absent`` is not evidence of acceptance or compliance.
    """

    normalized_patterns = _normalized_patterns(patterns)
    if decoding_failure_status is not None:
        return _validate_failure_status(decoding_failure_status)
    normalized_output = normalize_refusal_text(output)
    if any(pattern in normalized_output for pattern in normalized_patterns):
        return EXPLICIT_REFUSAL
    return EXPLICIT_REFUSAL_ABSENT


def _validate_failure_statuses(statuses: Collection[str]) -> frozenset[str]:
    if isinstance(statuses, (str, bytes)):
        raise TypeError("failure_statuses must be a collection of strings")
    return frozenset(_validate_failure_status(status) for status in statuses)


def _validate_item_id(item_id: object, *, item_key: str) -> Hashable:
    if item_id is None or (isinstance(item_id, str) and not item_id):
        raise ValueError(f"{item_key!r} must be a non-empty, hashable item identifier")
    try:
        hash(item_id)
    except TypeError as exc:
        raise ValueError(f"{item_key!r} must be hashable, got {item_id!r}") from exc
    return item_id  # type: ignore[return-value]


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
    item_key: str,
    status_key: str,
    failure_statuses: frozenset[str],
) -> dict[Hashable, Mapping[str, Any]]:
    indexed: dict[Hashable, Mapping[str, Any]] = {}
    allowed = VALID_REFUSAL_STATUSES | failure_statuses
    for row_index, row in enumerate(rows):
        if item_key not in row:
            raise ValueError(f"{side} row {row_index} is missing item key {item_key!r}")
        if status_key not in row:
            raise ValueError(f"{side} row {row_index} is missing status key {status_key!r}")
        item_id = _validate_item_id(row[item_key], item_key=item_key)
        status = row[status_key]
        if not isinstance(status, str) or status not in allowed:
            raise ValueError(
                f"{side} row {row_index} has unknown {status_key}={status!r}; "
                f"expected one of {sorted(allowed)!r}"
            )
        if item_id in indexed:
            raise ValueError(f"duplicate {side} row for item {item_id!r}")
        indexed[item_id] = row
    return indexed


def _item_sort_key(item_id: Hashable) -> tuple[str, str]:
    return (f"{type(item_id).__module__}.{type(item_id).__qualname__}", repr(item_id))


def _transition(source_status: str, target_status: str) -> str:
    if source_status == EXPLICIT_REFUSAL:
        if target_status == EXPLICIT_REFUSAL:
            return STABLE_REFUSAL
        return REFUSAL_TO_NONREFUSAL
    if target_status == EXPLICIT_REFUSAL:
        return NONREFUSAL_TO_REFUSAL
    return STABLE_NONREFUSAL


def _exact_mcnemar_two_sided(first_direction: int, reverse_direction: int) -> float:
    """Two-sided exact binomial test over the discordant pairs."""

    n_discordant = first_direction + reverse_direction
    if n_discordant == 0 or first_direction == reverse_direction:
        return 1.0
    smaller = min(first_direction, reverse_direction)
    lower_tail = sum(comb(n_discordant, k) for k in range(smaller + 1)) / (2**n_discordant)
    return float(min(1.0, 2.0 * lower_tail))


def _rate_interval(
    outcomes: Sequence[int],
    *,
    n_bootstrap: int,
    ci_alpha: float,
    seed: np.random.SeedSequence,
) -> dict[str, float | int | None]:
    values = np.asarray(outcomes, dtype=float)
    denominator = int(values.size)
    if denominator == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "denominator": 0}

    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, denominator, size=(n_bootstrap, denominator))
    bootstrap_rates = values[samples].mean(axis=1)
    low, high = np.quantile(bootstrap_rates, [ci_alpha / 2.0, 1.0 - ci_alpha / 2.0])
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "denominator": denominator,
    }


def summarize_paired_transitions(
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    item_key: str = "item_id",
    status_key: str = "refusal_status",
    failure_statuses: Collection[str] = (),
    n_bootstrap: int = 2_000,
    ci_alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize paired, bidirectional explicit-refusal transitions by item.

    Only item IDs present on both sides can form a pair.  A shared pair is valid
    exactly when neither side has one of ``failure_statuses``; failed pairs are
    reported separately.  Percentile bootstrap CIs resample items, using all valid
    items for disagreement and the relevant source-status subset for each
    conditional direction rate.
    """

    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0.0 < ci_alpha < 1.0:
        raise ValueError("ci_alpha must be strictly between 0 and 1")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    failures = _validate_failure_statuses(failure_statuses)
    source = _index_rows(
        source_rows,
        side="source",
        item_key=item_key,
        status_key=status_key,
        failure_statuses=failures,
    )
    target = _index_rows(
        target_rows,
        side="target",
        item_key=item_key,
        status_key=status_key,
        failure_statuses=failures,
    )

    shared = sorted(source.keys() & target.keys(), key=_item_sort_key)
    transition_counts: Counter[str] = Counter()
    failure_source_counts: Counter[str] = Counter()
    failure_target_counts: Counter[str] = Counter()
    source_only_failures = 0
    target_only_failures = 0
    both_failures = 0
    valid_states: list[tuple[bool, bool]] = []

    for item_id in shared:
        source_status = str(source[item_id][status_key])
        target_status = str(target[item_id][status_key])
        source_failed = source_status in failures
        target_failed = target_status in failures
        if source_failed or target_failed:
            if source_failed:
                failure_source_counts[source_status] += 1
            if target_failed:
                failure_target_counts[target_status] += 1
            if source_failed and target_failed:
                both_failures += 1
            elif source_failed:
                source_only_failures += 1
            else:
                target_only_failures += 1
            continue

        transition_counts[_transition(source_status, target_status)] += 1
        valid_states.append((source_status == EXPLICIT_REFUSAL, target_status == EXPLICIT_REFUSAL))

    n_valid = len(valid_states)
    r_to_nr = transition_counts[REFUSAL_TO_NONREFUSAL]
    nr_to_r = transition_counts[NONREFUSAL_TO_REFUSAL]
    n_disagreement = r_to_nr + nr_to_r
    disagreement_outcomes = [int(source_r != target_r) for source_r, target_r in valid_states]
    r_to_nr_outcomes = [int(not target_r) for source_r, target_r in valid_states if source_r]
    nr_to_r_outcomes = [int(target_r) for source_r, target_r in valid_states if not source_r]
    child_seeds = np.random.SeedSequence(seed).spawn(3)

    return {
        "n_source_rows": len(source),
        "n_target_rows": len(target),
        "n_shared_items": len(shared),
        "n_source_only_items": len(source.keys() - target.keys()),
        "n_target_only_items": len(target.keys() - source.keys()),
        "n_valid_pairs": n_valid,
        "transitions": {
            STABLE_REFUSAL: transition_counts[STABLE_REFUSAL],
            REFUSAL_TO_NONREFUSAL: r_to_nr,
            NONREFUSAL_TO_REFUSAL: nr_to_r,
            STABLE_NONREFUSAL: transition_counts[STABLE_NONREFUSAL],
            "disagreement": n_disagreement,
        },
        "rates": {
            "disagreement_rate": float(n_disagreement / n_valid) if n_valid else None,
            "refusal_to_nonrefusal_given_source_refusal": (
                float(np.mean(r_to_nr_outcomes)) if r_to_nr_outcomes else None
            ),
            "nonrefusal_to_refusal_given_source_nonrefusal": (
                float(np.mean(nr_to_r_outcomes)) if nr_to_r_outcomes else None
            ),
        },
        "direction_asymmetry": {
            "method": "exact_two_sided_mcnemar_binomial",
            "n_discordant": n_disagreement,
            REFUSAL_TO_NONREFUSAL: r_to_nr,
            NONREFUSAL_TO_REFUSAL: nr_to_r,
            "p_value": _exact_mcnemar_two_sided(r_to_nr, nr_to_r),
        },
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "confidence_level": 1.0 - ci_alpha,
            "disagreement_rate": _rate_interval(
                disagreement_outcomes,
                n_bootstrap=n_bootstrap,
                ci_alpha=ci_alpha,
                seed=child_seeds[0],
            ),
            "refusal_to_nonrefusal_given_source_refusal": _rate_interval(
                r_to_nr_outcomes,
                n_bootstrap=n_bootstrap,
                ci_alpha=ci_alpha,
                seed=child_seeds[1],
            ),
            "nonrefusal_to_refusal_given_source_nonrefusal": _rate_interval(
                nr_to_r_outcomes,
                n_bootstrap=n_bootstrap,
                ci_alpha=ci_alpha,
                seed=child_seeds[2],
            ),
        },
        "failures": {
            "n_failure_pairs": source_only_failures + target_only_failures + both_failures,
            "source_only": source_only_failures,
            "target_only": target_only_failures,
            "both": both_failures,
            "source_status_counts": dict(sorted(failure_source_counts.items())),
            "target_status_counts": dict(sorted(failure_target_counts.items())),
        },
    }


def _stratum_for_pair(
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    *,
    stratum_keys: Sequence[str],
) -> tuple[Hashable, ...]:
    values: list[Hashable] = []
    for key in stratum_keys:
        if key not in source_row or key not in target_row:
            raise ValueError(f"paired rows must both contain stratum key {key!r}")
        if source_row[key] != target_row[key]:
            raise ValueError(
                f"source and target disagree on stratum key {key!r}: "
                f"{source_row[key]!r} != {target_row[key]!r}"
            )
        try:
            hash(source_row[key])
        except TypeError as exc:
            raise ValueError(f"stratum value for {key!r} must be hashable") from exc
        values.append(source_row[key])
    return tuple(values)


def _selection_hash(seed: int, item_id: Hashable) -> bytes:
    payload = f"{seed}\x1f{type(item_id).__module__}.{type(item_id).__qualname__}\x1f{item_id!r}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def select_transition_cohort(
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    item_key: str = "item_id",
    status_key: str = "refusal_status",
    failure_statuses: Collection[str] = (),
    stratum_keys: Sequence[str] = (),
    stable_refusal_cap_per_stratum: int = 0,
    stable_nonrefusal_cap_per_stratum: int = 0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select an auditable mechanism cohort without assigning compliance labels.

    Every valid discordant pair is returned first and is never capped.  Within
    each caller-defined stratum, deterministic hash-ranked samples of stable
    refusal and stable non-refusal pairs are then added up to their separate caps.
    Failed and unpaired items are not eligible.  A selected stable non-refusal is
    only a control for refusal-marker absence, not a presumed compliant answer.
    """

    for name, cap in (
        ("stable_refusal_cap_per_stratum", stable_refusal_cap_per_stratum),
        ("stable_nonrefusal_cap_per_stratum", stable_nonrefusal_cap_per_stratum),
    ):
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(stratum_keys, (str, bytes)):
        raise TypeError("stratum_keys must be a sequence of field names")
    if any(not isinstance(key, str) or not key for key in stratum_keys):
        raise ValueError("every stratum key must be a non-empty string")

    failures = _validate_failure_statuses(failure_statuses)
    source = _index_rows(
        source_rows,
        side="source",
        item_key=item_key,
        status_key=status_key,
        failure_statuses=failures,
    )
    target = _index_rows(
        target_rows,
        side="target",
        item_key=item_key,
        status_key=status_key,
        failure_statuses=failures,
    )

    candidates: dict[str, dict[tuple[Hashable, ...], list[Hashable]]] = {
        STABLE_REFUSAL: {},
        REFUSAL_TO_NONREFUSAL: {},
        NONREFUSAL_TO_REFUSAL: {},
        STABLE_NONREFUSAL: {},
    }
    strata_by_item: dict[Hashable, tuple[Hashable, ...]] = {}
    for item_id in source.keys() & target.keys():
        source_status = str(source[item_id][status_key])
        target_status = str(target[item_id][status_key])
        if source_status in failures or target_status in failures:
            continue
        stratum = _stratum_for_pair(source[item_id], target[item_id], stratum_keys=stratum_keys)
        transition = _transition(source_status, target_status)
        candidates[transition].setdefault(stratum, []).append(item_id)
        strata_by_item[item_id] = stratum

    selected: list[tuple[Hashable, str, str]] = []
    # Ordering is part of the API: every discordant pair precedes every control.
    for transition in (REFUSAL_TO_NONREFUSAL, NONREFUSAL_TO_REFUSAL):
        all_items = [item for group in candidates[transition].values() for item in group]
        for item_id in sorted(all_items, key=_item_sort_key):
            selected.append((item_id, transition, "discordant"))

    for transition, cap in (
        (STABLE_REFUSAL, stable_refusal_cap_per_stratum),
        (STABLE_NONREFUSAL, stable_nonrefusal_cap_per_stratum),
    ):
        groups = candidates[transition]
        for stratum in sorted(groups, key=repr):
            ranked = sorted(
                groups[stratum],
                key=lambda item_id: (
                    _selection_hash(seed, item_id),
                    _item_sort_key(item_id),
                ),
            )
            for item_id in ranked[:cap]:
                selected.append((item_id, transition, "stable_control"))

    result: list[dict[str, Any]] = []
    for item_id, transition, selection_role in selected:
        stratum = strata_by_item[item_id]
        result.append(
            {
                "item_id": item_id,
                "transition": transition,
                "selection_role": selection_role,
                "stratum": dict(zip(stratum_keys, stratum, strict=True)),
                "source_row": dict(source[item_id]),
                "target_row": dict(target[item_id]),
            }
        )
    return result


__all__ = [
    "EXPLICIT_REFUSAL",
    "EXPLICIT_REFUSAL_ABSENT",
    "NONREFUSAL_TO_REFUSAL",
    "REFUSAL_TO_NONREFUSAL",
    "STABLE_NONREFUSAL",
    "STABLE_REFUSAL",
    "classify_explicit_refusal",
    "normalize_refusal_text",
    "select_transition_cohort",
    "summarize_paired_transitions",
]
