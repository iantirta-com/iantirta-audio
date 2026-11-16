from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    import torch
    from iantirta.models.vendor.demucs.apply import BagOfModels, Model


__all__ = [
    "ModelAudioSeparation",
    "SeparationResult"
]


@dataclass(slots=True, frozen=True)
class SeparationResult:
    """
    Wrapper that hold separation result
    
    :param torch.Tensor vocal:
    :param torch.Tensor instrument:
    :param int samplerate:
    """
    vocal: torch.Tensor
    instrument: torch.Tensor
    samplerate: int


@dataclass(slots=True)
class ModelAudioSeparation:
    model_name_or_path: str

    device: str | None = None
    shifts: int = 1
    overlap: float = 0.25
    split: bool = True
    segment: int | None = None
    jobs: int = 0
    progress: bool = True
    callback: Callable[[dict], None] | None = None
    callback_arg: dict | None = None
    logger: logging.Logger | None = field(default=None, repr=False)

    _model: BagOfModels | None = field(default=None, repr=False)
    _audio_channels: int | None = field(default=None, repr=False)
    _samplerate: int | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.device or self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger = self.logger or logging.getLogger(__name__)

        self.logger.info("Running %s | %r", self.model_name_or_path, self)


    # -------------------------
    # API Method
    # -------------------------

    def separate_tensor(
        self,
        wav: torch.Tensor,
        sample_rate: int,
        *,
        logger: logging.Logger | None = None
    ):
        if sample_rate != self._samplerate:
            (logger or self.logger).warning(
                "Different sample rate than model "
                f"Model={self._samplerate} -> {sample_rate}"
            )

        from iantirta.models.vendor.demucs.apply import _replace_dict, apply_model

        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8

        out = apply_model(
            self.model,
            ((wav - mean) / std)[None],
            segment=self.segment,
            shifts=self.shifts,
            split=self.split,
            overlap=self.overlap,
            device=self.device,
            num_workers=self.jobs,
            callback=self.callback,
            callback_arg=_replace_dict(
                self.callback_arg, ("audio_length", wav.shape[1])
            ),
            progress=self.progress,
        )

        out = out * std + mean
        return (wav, dict(zip(self.model.sources, out[0])))

    def separate(
        self,
        audio_path: str | Path,
        mix_method: Literal["subtract", "add"] = "subtract",
        *,
        logger: logging.Logger | None = None
    ):
        """
        Main separate method for audio separation

        :param str | Path audio_path:
        :param Literal['subtract', 'add'] mix_method:
            default = ``subtract``
        :param Optional[logging.Logger] logger:
        """
        if mix_method not in ("subtract", "add",):
            raise ValueError(f"Invalid mix method: {mix_method}")

        import torch

        from iantirta.audio import load_audio

        wav: np.ndarray
        wav, sr = load_audio(
            audio_path,
            sampling_rate=self._samplerate,
            num_channels=self._audio_channels,
        )
        (logger or self.logger).debug(f'Successfully loading audio: sr={sr}, wav={wav.shape}, audio_path={audio_path}',)
        wav: torch.Tensor = torch.from_numpy(wav)
        (logger or self.logger).debug(f'Successfully convert: wav={wav.shape}')

        (logger or self.logger).debug(f"Will separate with mix_method={mix_method}")
        origin, res = self.separate_tensor(wav, sr, logger=logger)

        vocal = res.pop("vocals")

        if mix_method == "add":
            instrument = torch.zeros_like(next(iter(res.values())))
            for i in res.values():
                instrument += i

        elif mix_method == "subtract":
            instrument = origin - vocal

        (logger or self.logger).debug("separation Success")

        return SeparationResult(
            vocal=vocal,
            instrument=instrument,
            samplerate=sr
        )

    # -------------------------
    # Model loading method
    # -------------------------

    @staticmethod
    def _decode_json(value):
        """Decode the json encoding of the model init arguments, in particular
        fractions (e.g. the `segment` param of HTDemucs)."""
        if isinstance(value, dict):
            if value.get("_type") == "fraction":
                return Fraction(value["numerator"], value["denominator"])
            return {key: ModelAudioSeparation._decode_json(item) for key, item in value.items()}
        elif isinstance(value, list):
            return [ModelAudioSeparation._decode_json(item) for item in value]
        return value

    @staticmethod
    def _unflatten_state(tensors: dict[str, Any], structure):
        """Rebuild a nested model state (e.g. diffq packed states) from the flat
        safetensors tensor dict and the json structure stored in its metadata.
        Inverse of `_flatten_state` in `tools/export_hf.py`."""
        def unflatten(node):
            if isinstance(node, dict):
                if "_tensor" in node:
                    return tensors[node["_tensor"]]
                elif "_dict" in node:
                    return {key: unflatten(item) for key, item in node["_dict"]}
                elif "_list" in node:
                    return [unflatten(item) for item in node["_list"]]
                elif "_tuple" in node:
                    return tuple(unflatten(item) for item in node["_tuple"])
                elif "_class" in node:
                    module, name = node["_class"].rsplit(".", 1)
                    return getattr(importlib.import_module(module), name)
                else:
                    raise ValueError(f"Invalid structure node {node}.")
            return node
        return unflatten(structure)

    @staticmethod
    def load_safetensors_model(path: str | Path) -> Model:
        """
        Load a single model from a safetensors file, with the model class and init
        arguments stored as json in the safetensors metadata.
        """
        from iantirta.models.vendor.demucs.states import load_model
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as file:
            metadata = file.metadata()
            tensors = {key: file.get_tensor(key) for key in file.keys()}  # noqa: SIM118

        state: Any
        if 'structure' in metadata:
            state = ModelAudioSeparation._unflatten_state(tensors, json.loads(metadata['structure']))
        else:
            state = tensors

        module, name = metadata['klass'].rsplit(".", 1)
        if module.startswith("demucs."):
            module = f"iantirta.models.vendor.demucs.{module[len('demucs.'):]}"

        klass = getattr(importlib.import_module(module), name)
        args = ModelAudioSeparation._decode_json(json.loads(metadata['args']))
        kwargs = ModelAudioSeparation._decode_json(json.loads(metadata['kwargs']))

        return load_model({'klass': klass, 'args': args, 'kwargs': kwargs, 'state': state})

    def load_model(self):
        import yaml
        from iantirta.models.vendor.demucs.apply import BagOfModels
        from transformers.utils.hub import cached_file, cached_files

        _, name = self.model_name_or_path.rsplit("/", 1)
        if "-" in name and name.startswith("Demucs-"):
            name = name[len("Demucs-"):]

        with open(cached_file(self.model_name_or_path, f"{name}.yaml")) as file:
            bag = yaml.safe_load(file)

        models_path = cached_files(
            self.model_name_or_path, 
            [f"{sig}.safetensors" for sig in bag['models']]
        )
        models = [self.load_safetensors_model(model_path) for model_path in models_path]

        self._model = BagOfModels(models, bag.get('weights'), bag.get('segment'))

        self._audio_channels = self.model.audio_channels
        self._samplerate = self.model.samplerate

    # -------------------------
    # Lazy Property
    # -------------------------

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model
