from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .network.registry import build_network_definition, canonical_network_name


_PROJECT_DIR = Path(__file__).resolve().parents[1]
_ENV_PARAMS_PATH = _PROJECT_DIR / "params" / "env_params.json"


def _load_env_params() -> dict[str, Any]:
    with _ENV_PARAMS_PATH.open("r", encoding="utf-8") as file_obj:
        params = json.load(file_obj)
    if not isinstance(params, dict):
        raise ValueError(f"Expected JSON object in env params file: {_ENV_PARAMS_PATH}")
    if "common" not in params or "networks" not in params:
        raise ValueError(
            "Env params file must contain 'common' and 'networks' sections: "
            f"{_ENV_PARAMS_PATH}"
        )
    if not isinstance(params["common"], dict) or not isinstance(params["networks"], dict):
        raise ValueError(
            "Env params file sections 'common' and 'networks' must both be JSON objects: "
            f"{_ENV_PARAMS_PATH}"
        )
    return params


_ENV_PARAMS = _load_env_params()


def _flatten_settings_block(settings: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in settings.items():
        if key in {"env", "dnl"}:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Env params section {key!r} must be a JSON object in {_ENV_PARAMS_PATH}."
                )
            flattened.update(deepcopy(value))
        else:
            flattened[key] = deepcopy(value)
    return flattened


DNL_COMMON_DEFAULTS: dict[str, Any] = _flatten_settings_block(_ENV_PARAMS["common"])
DNL_NETWORK_SETTINGS: dict[str, dict[str, Any]] = {
    canonical_network_name(network_name): _flatten_settings_block(settings)
    for network_name, settings in _ENV_PARAMS["networks"].items()
}


def get_dnl_settings(network_name: str) -> dict[str, Any]:
    canonical_name = canonical_network_name(network_name)
    settings = deepcopy(DNL_COMMON_DEFAULTS)
    settings.update(deepcopy(DNL_NETWORK_SETTINGS.get(canonical_name, {})))
    if settings.get("action_high") is None:
        settings["action_high"] = float(build_network_definition(canonical_name).default_action_high)
    if settings["max_paths_per_od"] is None:
        settings["max_paths_per_od"] = int(build_network_definition(canonical_name).default_max_paths_per_od)
    return settings
