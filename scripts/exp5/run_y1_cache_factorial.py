#!/usr/bin/env -S uv run python
"""Run Exp5: the Exp4 cache surgery with the first generated token as a third factor."""

from pathlib import Path

from audio_safety.pipelines.kv_routing_cli import run_cli

if __name__ == "__main__":
    run_cli(
        prefix="exp5",
        default_config=Path("configs/experiments/exp5_y1_cache_factorial.yaml"),
        description=__doc__ or "",
    )
