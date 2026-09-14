"""YAML experiment configs.

Each file in ``configs/`` describes one experiment: which model, which data,
which hyper-parameters. Keeping them as data rather than as arguments buried in
a notebook cell is what makes a run reproducible three years later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["ExperimentConfig", "load_config", "save_config"]


@dataclass
class ExperimentConfig:
    """A single experiment.

    Attributes:
        name: Identifier used for output directories.
        model: Model family, e.g. ``deepseanet``, ``yolov8s``.
        neck: Neck variant for the DeepSeaNet family.
        phi: Compound-scaling coefficient.
        num_classes: Number of object classes.
        image_size: Square training resolution.
        epochs: Training length.
        batch_size: Images per optimisation step.
        optimizer: Optimiser name.
        learning_rate: Initial learning rate.
        weight_decay: L2 penalty coefficient.
        alpha: Weight on the box term of the loss.
        beta: Weight on the L2 term of the loss.
        seed: RNG seed.
        adversarial: Whether UAP adversarial training is enabled.
        uap_epsilon: Maximum perturbation strength.
        uap_warmup_frac: Fraction of training kept perturbation-free.
        data_root: Dataset root directory.
        split: ``random`` or ``grouped``.
        extra: Anything else, passed through untouched.
    """

    name: str = "deepseanet-biskfpn"
    model: str = "deepseanet"
    neck: str = "biskfpn"
    phi: int = 0
    num_classes: int = 6
    image_size: int = 512
    epochs: int = 350
    batch_size: int = 64
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    alpha: float = 50.0
    beta: float = 1.0
    seed: int = 0
    adversarial: bool = False
    uap_epsilon: float = 8 / 255
    uap_warmup_frac: float = 0.2
    data_root: str = "data/brackish"
    split: str = "grouped"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        """Build from a plain dict, routing unknown keys into ``extra``."""
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for k, v in payload.items() if k in known}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        """Flatten back to a dict, merging ``extra`` in at the top level."""
        data = asdict(self)
        data.update(data.pop("extra"))
        return data


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config file.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the file does not exist.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dependency
        raise ImportError("load_config needs PyYAML: pip install pyyaml") from exc
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig.from_dict(payload)


def save_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write a config back out as YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("save_config needs PyYAML: pip install pyyaml") from exc
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    return p
