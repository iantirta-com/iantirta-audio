# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import inspect
import typing as t
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEMUCS_ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
DEMUCS_REMOTE_ROOT = Path(__file__).parent / "vendor" / "demucs" / "remote"


@dataclass(frozen=True, slots=True)
class _CheckpointSpec:
    """Description of a downloadable separation model.
    
    Parameters
    ----------
    signature:
        string unique identification for model checkpoint
    
    url:
        hosted model weights url.
    """

    signature: str
    url: str


def _load_checkpoint_model(package: dict[str, t.Any]):
    assert isinstance(package, dict)

    klass = package["klass"]
    demucs_class: str = klass.__name__

    args = package["args"]
    kwargs = package["kwargs"]
    state = package["state"]

    sig = inspect.signature(klass)
    for key in list(kwargs):
        if key not in sig.parameters:
            warnings.warn(
                f"Dropping nonexistent parameter {key!r}"
            )
            del kwargs[key]

    from . import models
    model_class = getattr(models, demucs_class)
    model = model_class(*args, **kwargs)

    from .vendor.demucs.states import set_state
    set_state(model, state)
    
    return model


def _load_checkpoint(checkpoint: _CheckpointSpec):
    import torch
    
    package: dict[str, t.Any] = torch.hub.load_state_dict_from_url(
        checkpoint.url,
        map_location="cpu",
        check_hash=True,
    )
    try:
        return _load_checkpoint_model(package)
    except ImportError as exc:
        if 'diffq' in exc.args[0]:
            raise ImportError(
                "Demucs Audio separation requires diffq. "
                "Install it with `pip install iantirta-audio[separation]`."
            ) from exc
        raise


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    """Description of a Demucs pretrained model or model bag.
    
    Parameters
    ----------
    name:
        model name
    
    yaml_path:
        Path() to the yaml file.
    """

    name: str
    yaml_path: Path

    _info: dict | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    _models: list | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def info(self) -> dict:
        if self._info is None:
            with self.yaml_path.open("r", encoding="utf-8") as file:
                object.__setattr__(
                    self,
                    "_info",
                    yaml.safe_load(file),
                )
        return self._info

    @property
    def models(self) -> list:
        if self._models is None:
            object.__setattr__(self, "_models", [
                _load_checkpoint(_CHECKPOINTS[signature])
                for signature in self.info["models"]
            ],)
        return self._models

    @property
    def weights(self) -> list[list[float]] | None:
        return self.info.get("weights")

    @property
    def segment(self) -> float | None:
        return self.info.get("segment")


def _parse_remote_files():
    checkpoints: dict[str, _CheckpointSpec] = {}
    models: dict[str, _ModelSpec] = {}
    root = ""

    for line in (DEMUCS_REMOTE_ROOT / "files.txt").read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("root:"):
            root = line.split(":", 1)[1].strip()
            continue

        signature = line.split("-", 1)[0]

        if signature in checkpoints:
            raise ValueError(
                f"Duplicate Demucs model signature: {signature}"
            )

        checkpoints[signature] = _CheckpointSpec(
            signature=signature,
            url=f"{DEMUCS_ROOT_URL}{root}{line}",
        )

    for file in DEMUCS_REMOTE_ROOT.iterdir():
        if file.suffix == ".yaml":
            models[file.stem] = _ModelSpec(
                name=file.stem,
                yaml_path=file,
            )

    return checkpoints, models

_CHECKPOINTS: dict[str, _CheckpointSpec]
_MODELS: dict[str, _ModelSpec]
_CHECKPOINTS, _MODELS = _parse_remote_files()
