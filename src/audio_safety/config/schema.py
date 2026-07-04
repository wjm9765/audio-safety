"""Pydantic schemas for experiment configuration.

Every tunable (model ID, sample counts, statistical thresholds) lives here and in
``configs/`` — never hardcoded in pipeline code (AGENTS.md).
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Reject unknown keys so config typos fail loudly instead of silently."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(StrictModel):
    """All values optional; see audio_safety.utils.paths for resolution priority."""

    workspace: Path | None = None
    data_dir: Path | None = None
    output_dir: Path | None = None
    cache_dir: Path | None = None


class ModelConfig(StrictModel):
    model_id: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"


class ConeConfig(StrictModel):
    categories: list[str] = Field(min_length=1)
    pairs_per_category: int = 256
    methods: list[str] = Field(default_factory=lambda: ["diff_in_means", "pca"])
    causal_validation: bool = True


class DriftConfig(StrictModel):
    n_contents: int = 150
    families: list[str] = Field(min_length=2)
    min_valid_per_family: int = 100
    style_voice_seeds: int = 3
    benign_control_n: int = 100


class StatsConfig(StrictModel):
    n_permutations: int = 5000
    n_bootstrap: int = 1000
    ci_alpha: float = 0.05


class DecisionConfig(StrictModel):
    """Pre-registered thresholds (design.md §0). Changing these mid-experiment is a
    protocol violation — see AGENTS.md '실험 문서 규약'."""

    go_max_cosine: float = 0.60
    nogo_min_cosine: float = 0.85
    p_threshold: float = 0.05


class ExperimentConfig(StrictModel):
    name: str
    seed: int = 0
    model: ModelConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    cone: ConeConfig
    drift: DriftConfig
    stats: StatsConfig = Field(default_factory=StatsConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
