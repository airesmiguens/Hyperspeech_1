"""Central model registry for baseline and HyperSpeech variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RegistryEntry:
    name: str
    family: str
    backend: str
    enabled_by_default: bool
    params: dict[str, Any]


def get_model_registry() -> dict[str, RegistryEntry]:
    return {
        "catboost": RegistryEntry("catboost", "baseline", "catboost", True, {}),
        "histgb": RegistryEntry("histgb", "baseline", "sklearn", True, {}),
        "realmlp": RegistryEntry("realmlp", "baseline", "sklearn", True, {}),
        "saint": RegistryEntry("saint", "baseline", "torch", False, {}),
        "tabpfn": RegistryEntry("tabpfn", "baseline", "tabpfn", True, {}),
        "ft_transformer": RegistryEntry("ft_transformer", "baseline", "torch", False, {}),
        "tabtransformer": RegistryEntry("tabtransformer", "baseline", "torch", False, {}),
        "dcnv2": RegistryEntry("dcnv2", "baseline", "torch", False, {}),
        "tabnet": RegistryEntry("tabnet", "baseline", "torch", False, {}),
        "node": RegistryEntry("node", "baseline", "torch", False, {}),
        "carte": RegistryEntry("carte", "baseline", "torch", False, {}),
        "hyperspeech_tokenmixer": RegistryEntry("hyperspeech_tokenmixer", "hyperspeech", "torch", True, {}),
        "hyperspeech_mil": RegistryEntry("hyperspeech_mil", "hyperspeech", "torch", True, {}),
        "hyperspeech_calibrated": RegistryEntry("hyperspeech_calibrated", "hyperspeech", "meta", True, {}),
    }
