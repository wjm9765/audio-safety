"""Frozen text-jailbreak attack wrappers for the Run 4 §8 attack-induced-flip line.

Deterministic and offline: a frozen wrapper template plus a base request yields
exactly one spoken attack text. No network, no model, so the attack cohort is
reproducible and can be frozen/audited by hashing (design §8.2, §8.6). The output
records reuse the style-variant override schema so the existing render pipeline
(`render_audio_records`) picks them up as new ``style`` conditions with neutral
acoustics — only the wording carries the attack.
"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from audio_safety.data.datasets import AudioRdoPair

REQUEST_SLOT = "{request}"


@dataclass(frozen=True)
class JailbreakTemplate:
    """One frozen jailbreak wrapper. ``template`` must contain ``{request}`` once."""

    id: str
    style_key: str
    family: str
    template: str
    source: str = ""

    @property
    def template_sha256(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()


def load_jailbreak_templates(path: Path) -> list[JailbreakTemplate]:
    """Load and validate the frozen jailbreak template set.

    Fails closed: every template must carry a unique ``style_key`` and exactly one
    ``{request}`` slot so downstream ``style``-keyed records never collide and the
    base request is always substituted.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("templates") or []
    templates: list[JailbreakTemplate] = []
    seen_style_keys: set[str] = set()
    seen_ids: set[str] = set()
    for entry in entries:
        template_text = str(entry["template"])
        if template_text.count(REQUEST_SLOT) != 1:
            raise ValueError(
                f"template {entry.get('id')!r} must contain exactly one {REQUEST_SLOT} slot"
            )
        style_key = str(entry["style_key"])
        template_id = str(entry["id"])
        if style_key in seen_style_keys:
            raise ValueError(f"duplicate style_key {style_key!r} in {path}")
        if template_id in seen_ids:
            raise ValueError(f"duplicate template id {template_id!r} in {path}")
        seen_style_keys.add(style_key)
        seen_ids.add(template_id)
        templates.append(
            JailbreakTemplate(
                id=template_id,
                style_key=style_key,
                family=str(entry.get("family", "jailbreak")),
                template=template_text,
                source=str(entry.get("source", "")),
            )
        )
    if not templates:
        raise ValueError(f"no templates found in {path}")
    return templates


def wrap_request(template: JailbreakTemplate, request_text: str) -> str:
    """Substitute the base request into the wrapper.

    Uses ``str.replace`` (not ``str.format``) so any literal braces in the request
    or in the surrounding wrapper text are never interpreted as format fields.
    """
    return template.template.replace(REQUEST_SLOT, request_text.strip())


def build_attack_variant_records(
    pairs: Iterable[AudioRdoPair],
    templates: Sequence[JailbreakTemplate],
    *,
    safety_labels: Sequence[str] = ("harmful", "benign"),
) -> list[dict[str, object]]:
    """Wrap every (pair-side x template) cell into a style-override record.

    The record schema matches what ``_load_style_text_overrides`` consumes
    (``item_id`` / ``safety_label`` / ``target_style`` / ``styled_text`` plus the
    three self-report gate fields), augmented with freeze metadata
    (``template_id`` / ``template_sha256`` / ``rendered_sha256``) so the exact
    spoken attack text is auditable.
    """
    records: list[dict[str, object]] = []
    for pair in pairs:
        side_text = {"harmful": pair.harmful_text, "benign": pair.benign_text}
        for template in templates:
            for label in safety_labels:
                base = side_text[label]
                # Store the exact spoken text (stripped) and hash THAT, so the
                # frozen rendered_sha256 equals the render-time reference_sha256 (the
                # override loader also strips before use — see families.py).
                styled = wrap_request(template, base).strip()
                records.append(
                    {
                        "item_id": pair.item_id,
                        "category": pair.category,
                        "safety_label": label,
                        "target_style": template.style_key,
                        "styled_text": styled,
                        "base_reference_text": base,
                        "template_id": template.id,
                        "template_family": template.family,
                        "template_source": template.source,
                        "template_sha256": template.template_sha256,
                        "rendered_sha256": hashlib.sha256(
                            styled.encode("utf-8")
                        ).hexdigest(),
                        # The wrapper adds only attack framing, never operational
                        # detail, so the style-override gate fields are constant.
                        # The harmful payload is entirely the base request.
                        "added_operational_detail": False,
                        "refusal_or_warning": False,
                        "content_preservation": "high",
                    }
                )
    return records


def freeze_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    """Per-style-key counts + template hash for a freeze log (design §8.6)."""
    by_style: dict[str, dict[str, object]] = {}
    for record in records:
        style = str(record["target_style"])
        slot = by_style.setdefault(
            style,
            {
                "template_id": record.get("template_id"),
                "template_sha256": record.get("template_sha256"),
                "n": 0,
            },
        )
        slot["n"] = int(slot["n"]) + 1  # type: ignore[arg-type]
    return {"n_records": len(records), "by_style": by_style}
