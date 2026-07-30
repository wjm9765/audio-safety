#!/usr/bin/env -S uv run python
"""Render the Exp3 causal-transport figure from a completed run directory.

Four panels tracing where an acoustic edit's effect on refusal lives:

  A  M2 audio-span transport by layer, real vs the three norm-matched controls
  B  the same cells' common-mode component, which is what separates real in kind
  C  the hand-off: whole audio span vs the single t_AB position
  D  M3 escape/reset, i.e. how much survives resetting the audio span

Usage:
    ./scripts/exp3/plot_transport.py <run_dir> [--out figures/transport.png]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC, TGT = "source_to_target", "target_to_source"
# Validated categorical slots 1-4 (light surface). Aqua and yellow sit below 3:1
# contrast, so every series is direct-labelled -- the relief rule, not decoration.
COLORS = {
    "real": "#2a78d6",
    "random_direction": "#eb6834",
    "wrong_item": "#1baf7a",
    "position_sham": "#eda100",
}
LABELS = {
    "real": "real (paired donor)",
    "random_direction": "random direction",
    "wrong_item": "wrong item",
    "position_sham": "position sham",
}
T_AB_COLOR = "#4a3aa7"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dedddb"


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decompose(rows: list[dict], site: str) -> dict:
    """Return {(condition, layer): (T_mean, T_halfwidth, G_mean, G_halfwidth)}."""
    idx = {(r["pair_id"], r["direction"], r[site], r["condition"]): r for r in rows}
    pairs = sorted({r["pair_id"] for r in rows})
    out: dict = {}
    acc: dict = defaultdict(lambda: ([], []))
    for (pair, direction, layer, cond), _row in idx.items():
        if direction != SRC:
            continue
        a = idx.get((pair, SRC, layer, cond))
        b = idx.get((pair, TGT, layer, cond))
        if a is None or b is None:
            continue
        delta = a["donor_r_tab_margin"] - a["baseline_r_tab_margin"]
        d_ab = a["r_tab_margin"] - a["baseline_r_tab_margin"]
        d_ba = b["r_tab_margin"] - b["baseline_r_tab_margin"]
        sign = (delta > 0) - (delta < 0)
        acc[(cond, layer)][0].append(sign * 0.5 * (d_ab - d_ba))
        acc[(cond, layer)][1].append(0.5 * (d_ab + d_ba))
    for key, (ts, gs) in acc.items():
        half = lambda v: 1.96 * statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0  # noqa: E731
        out[key] = (statistics.mean(ts), half(ts), statistics.mean(gs), half(gs))
    _ = pairs
    return out


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def curve(ax, layers, stats, cond, index) -> float:
    """Draw one condition's curve; return its final y so labels can be de-collided."""
    ys = [stats[(cond, lay)][index] for lay in layers]
    hs = [stats[(cond, lay)][index + 1] for lay in layers]
    color = COLORS[cond]
    ax.fill_between(
        layers,
        [y - h for y, h in zip(ys, hs, strict=True)],
        [y + h for y, h in zip(ys, hs, strict=True)],
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    ax.plot(
        layers,
        ys,
        color=color,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        zorder=3,
    )
    return ys[-1]


def place_labels(ax, ends: dict[str, float], x: float, min_gap: float) -> None:
    """Direct-label each series, nudging apart any labels closer than min_gap.

    Direct labels are not decoration here: two categorical slots fall below 3:1
    contrast on this surface, so the palette validator's relief rule requires
    them. Overlapping labels would defeat that, hence the de-collision pass.
    """
    order = sorted(ends.items(), key=lambda kv: kv[1])
    placed: list[tuple[str, float]] = []
    for cond, y in order:
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((cond, y))
    for cond, y in placed:
        ax.annotate(
            LABELS[cond],
            xy=(x, ends[cond]),
            xytext=(x + 0.45, y),
            color=INK,
            fontsize=8.5,
            va="center",
            fontweight="medium",
            arrowprops=dict(arrowstyle="-", color=COLORS[cond], linewidth=0.9, alpha=0.55),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.run_dir
    out = args.out or run / "figures" / "transport.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    span = decompose(load(run / "span_patch" / "records.jsonl"), "layer")
    read = decompose(load(run / "readout_patch" / "records.jsonl"), "layer")
    esc_rows = load(run / "escape_reset" / "records.jsonl")

    layers = sorted({lay for _c, lay in span})
    conds = ["real", "random_direction", "wrong_item", "position_sham"]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.46, wspace=0.26, right=0.97, left=0.07, top=0.87, bottom=0.09)

    ax = axes[0][0]
    style(ax)
    ends = {cond: curve(ax, layers, span, cond, 0) for cond in conds}
    ax.axhline(0, color=INK2, linewidth=1, alpha=0.5)
    place_labels(ax, ends, layers[-1], min_gap=0.115)
    ax.set_title(
        "A · Transport toward the paired donor",
        color=INK,
        fontsize=11,
        loc="left",
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("LLM layer", color=INK2, fontsize=9.5)
    ax.set_ylabel("T  (direction-odd)", color=INK2, fontsize=9.5)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 5.2)

    ax = axes[0][1]
    style(ax)
    ends = {cond: curve(ax, layers, span, cond, 2) for cond in conds}
    ax.axhline(0, color=INK2, linewidth=1, alpha=0.5)
    place_labels(ax, ends, layers[-1], min_gap=0.155)
    ax.set_title(
        "B · Common-mode push, regardless of donor",
        color=INK,
        fontsize=11,
        loc="left",
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("LLM layer", color=INK2, fontsize=9.5)
    ax.set_ylabel("G  (direction-even)", color=INK2, fontsize=9.5)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 5.2)
    ax.set_ylim(-1.05, 1.55)
    ax.annotate(
        "only `real` sits at zero —\nit transports without pushing",
        xy=(12, 0.018),
        xytext=(8.6, 0.72),
        color=INK2,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=INK2, linewidth=1),
    )

    ax = axes[1][0]
    style(ax)
    sl = [span[("real", lay)][0] for lay in layers]
    sh = [span[("real", lay)][1] for lay in layers]
    ax.fill_between(
        layers,
        [a - b for a, b in zip(sl, sh, strict=True)],
        [a + b for a, b in zip(sl, sh, strict=True)],
        color=COLORS["real"],
        alpha=0.13,
        linewidth=0,
    )
    ax.plot(
        layers,
        sl,
        color=COLORS["real"],
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
    )
    rl = sorted({lay for _c, lay in read})
    ry = [read[("real", lay)][0] for lay in rl]
    rh = [read[("real", lay)][1] for lay in rl]
    ax.errorbar(
        rl,
        ry,
        yerr=rh,
        color=T_AB_COLOR,
        linewidth=2,
        marker="s",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        capsize=3,
    )
    ax.axhline(0, color=INK2, linewidth=1, alpha=0.5)
    ax.annotate(
        "whole audio span",
        xy=(12, sl[2]),
        xytext=(7.4, 1.42),
        color=COLORS["real"],
        fontsize=9,
        fontweight="medium",
        arrowprops=dict(arrowstyle="-", color=COLORS["real"], linewidth=0.9, alpha=0.6),
    )
    ax.annotate(
        "single t_AB position",
        xy=(26, 0.90),
        xytext=(19.5, 0.30),
        color=T_AB_COLOR,
        fontsize=9,
        fontweight="medium",
        arrowprops=dict(arrowstyle="-", color=T_AB_COLOR, linewidth=0.9, alpha=0.6),
    )
    ax.annotate(
        "at L10 the signal is in the audio\ntokens, not yet at the readout",
        xy=(10, 0.05),
        xytext=(11.0, 0.62),
        color=INK2,
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=INK2, linewidth=1),
    )
    ax.set_title(
        "C · Hand-off to the readout position",
        color=INK,
        fontsize=11,
        loc="left",
        fontweight="semibold",
        pad=10,
    )
    ax.set_xlabel("LLM layer", color=INK2, fontsize=9.5)
    ax.set_ylabel("T", color=INK2, fontsize=9.5)
    ax.set_xlim(6, 33.5)
    ax.set_ylim(-0.15, 1.62)

    ax = axes[1][1]
    style(ax)
    idx = {(r["pair_id"], r["direction"], r["reset_layer"], r["condition"]): r for r in esc_rows}
    pairs = sorted({r["pair_id"] for r in esc_rows})
    resets = sorted({r["reset_layer"] for r in esc_rows})

    def esc_T(reset: int, cond: str) -> tuple[float, float]:
        vals = []
        for pair in pairs:
            a = idx.get((pair, SRC, reset, cond))
            b = idx.get((pair, TGT, reset, cond))
            if a is None or b is None:
                continue
            d = a["donor_r_tab_margin"] - a["baseline_r_tab_margin"]
            s = (d > 0) - (d < 0)
            vals.append(
                s
                * 0.5
                * (
                    (a["r_tab_margin"] - a["baseline_r_tab_margin"])
                    - (b["r_tab_margin"] - b["baseline_r_tab_margin"])
                )
            )
        h = 1.96 * statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
        return statistics.mean(vals), h

    names = ["inject at L10\n(no reset)"] + [f"inject L10,\nreset span at L{r}" for r in resets]
    vals = [esc_T(resets[0], "inject_only")] + [esc_T(r, "inject_reset") for r in resets]
    shades = ["#2a78d6", "#7fb0e6", "#b9d3f2"]
    xs = range(len(vals))
    ax.bar(
        xs,
        [v[0] for v in vals],
        yerr=[v[1] for v in vals],
        color=shades[: len(vals)],
        width=0.58,
        capsize=4,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for x, (mean, half) in zip(xs, vals, strict=True):
        pct = mean / vals[0][0] * 100
        ax.annotate(
            f"{mean:+.3f}" + ("" if x == 0 else f"\n{pct:.1f}% survives"),
            xy=(x, mean + half),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
            fontweight="medium",
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, fontsize=8.5, color=INK2)
    ax.set_title(
        "D · Resetting the span removes the effect",
        color=INK,
        fontsize=11,
        loc="left",
        fontweight="semibold",
        pad=10,
    )
    ax.set_ylabel("T", color=INK2, fontsize=9.5)
    ax.set_ylim(0, max(v[0] + v[1] for v in vals) * 1.34)

    fig.suptitle(
        "Qwen2-Audio: a phase-only acoustic edit moves refusal through the audio-token span",
        color=INK,
        fontsize=13.5,
        fontweight="semibold",
        x=0.055,
        ha="left",
        y=0.975,
    )
    fig.text(
        0.055,
        0.935,
        "113-pair mechanism cohort · pv_locked vs pv_standard · bands and bars are 95% CI",
        color=INK2,
        fontsize=9.5,
        ha="left",
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"[exp3] wrote {out}")


if __name__ == "__main__":
    main()
