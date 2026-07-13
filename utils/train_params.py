"""Load network-specific experiment parameters shared by all launchers."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from dnl.network.registry import canonical_network_name, get_default_action_high

_SETTINGS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SETTINGS_DIR.parent


def _resolve_param_file_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = _PROJECT_DIR / path
    return path.resolve()


DEFAULT_NETWORK_ENV_VAR = "DODE_NETWORK_NAME"
MAX_RUNTIME_SECONDS_ENV_VAR = "DODE_MAX_RUNTIME_SECONDS"
TEST_STEP_RUNTIME_SECONDS_ENV_VAR = "DODE_TEST_STEP_RUNTIME_SECONDS"
NUM_ENVS_ENV_VAR = "DODE_NUM_ENVS"
USE_SUBPROC_ENV_VAR = "DODE_USE_SUBPROC"
ACTION_HIGH_ENV_VAR = "DODE_ACTION_HIGH"

_NETWORK_PARAM_FILE_GROUPS = {
    canonical_network_name("melbourne_scats"): (
        _resolve_param_file_path("params/train_params.json"),
        _resolve_param_file_path("params/test_params.json"),
    ),
}


def _get_default_network_name() -> str:
    configured_network = os.environ.get(DEFAULT_NETWORK_ENV_VAR, "melbourne_scats")
    return canonical_network_name(configured_network)


def _apply_common_env_overrides(common_settings: dict[str, Any]) -> dict[str, Any]:
    overridden_settings = deepcopy(common_settings)

    max_runtime_seconds = os.environ.get(MAX_RUNTIME_SECONDS_ENV_VAR)
    if max_runtime_seconds is not None:
        overridden_settings["max_runtime_seconds"] = int(max_runtime_seconds)

    test_step_runtime_seconds = os.environ.get(TEST_STEP_RUNTIME_SECONDS_ENV_VAR)
    if test_step_runtime_seconds is not None:
        overridden_settings["test_step_runtime_seconds"] = float(test_step_runtime_seconds)

    action_high = os.environ.get(ACTION_HIGH_ENV_VAR)
    if action_high is not None:
        overridden_settings["action_high"] = float(action_high)

    return overridden_settings


def _parse_bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean environment value, received {value!r}.")


def _apply_runtime_env_overrides(runtime_settings: dict[str, Any]) -> dict[str, Any]:
    overridden_settings = deepcopy(runtime_settings)

    num_envs = os.environ.get(NUM_ENVS_ENV_VAR)
    if num_envs is not None:
        overridden_settings["num_envs"] = int(num_envs)

    use_subproc = os.environ.get(USE_SUBPROC_ENV_VAR)
    if use_subproc is not None:
        overridden_settings["use_subproc"] = _parse_bool_env(use_subproc)

    return overridden_settings


def resolve_action_high_setting(
    network_name: str,
    common_settings: dict[str, Any] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> float:
    if experiment_settings is not None and "action_high" in experiment_settings:
        return float(experiment_settings["action_high"])
    if common_settings is not None and "action_high" in common_settings:
        return float(common_settings["action_high"])
    from dnl.config import get_dnl_settings

    dnl_settings = get_dnl_settings(canonical_network_name(network_name))
    if "action_high" in dnl_settings:
        return float(dnl_settings["action_high"])
    return float(get_default_action_high(canonical_network_name(network_name)))


@lru_cache(maxsize=None)
def _load_params_file(params_path: str) -> dict[str, dict[str, Any]]:
    resolved_path = Path(params_path)
    with resolved_path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in params file: {resolved_path}")
    return deepcopy(loaded)


def _merge_param_files(params_paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for params_path in params_paths:
        params = _load_params_file(str(params_path))
        for section_name, section_settings in params.items():
            if section_name in merged and section_name != "common":
                raise ValueError(
                    f"Duplicate parameter section {section_name!r} across split params files."
                )
            if section_name == "common" and section_name in merged:
                merged[section_name] = {
                    **deepcopy(merged[section_name]),
                    **deepcopy(section_settings),
                }
            else:
                merged[section_name] = deepcopy(section_settings)
    return merged


def _get_network_settings() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        network_name: _merge_param_files(tuple(params_paths))
        for network_name, params_paths in _NETWORK_PARAM_FILE_GROUPS.items()
    }


def get_network_name(_: str | None = None) -> str:
    return _get_default_network_name()


def get_common_settings(network_name: str | None = None) -> dict[str, Any]:
    selected_network = canonical_network_name(network_name or _get_default_network_name())
    try:
        network_block = _get_network_settings()[selected_network]
    except KeyError as exc:
        raise KeyError(f"Unsupported network settings key: {selected_network}") from exc
    common_settings = network_block.get("common", {})
    if not isinstance(common_settings, dict):
        raise ValueError(f"Expected common settings object for network='{selected_network}'")
    return _apply_common_env_overrides(common_settings)


def get_section_settings(
    section_name: str,
    network_name: str | None = None,
    *,
    apply_runtime_env_overrides: bool = False,
) -> dict[str, Any]:
    selected_network = canonical_network_name(network_name or _get_default_network_name())
    try:
        network_block = _get_network_settings()[selected_network]
    except KeyError as exc:
        raise KeyError(f"Unsupported network settings key: {selected_network}") from exc
    try:
        section_settings = network_block[str(section_name)]
    except KeyError as exc:
        raise KeyError(f"Missing settings section for network='{selected_network}', section='{section_name}'") from exc
    if not isinstance(section_settings, dict):
        raise ValueError(f"Expected settings object for network='{selected_network}', section='{section_name}'")
    resolved_settings = deepcopy(section_settings)
    if apply_runtime_env_overrides:
        resolved_settings = _apply_runtime_env_overrides(resolved_settings)
    return resolved_settings


def get_experiment_settings(experiment_name: str, network_name: str | None = None) -> dict[str, Any]:
    selected_network = canonical_network_name(network_name or get_network_name())
    try:
        network_block = _get_network_settings()[selected_network]
    except KeyError as exc:
        raise KeyError(f"Unsupported network settings key: {selected_network}") from exc
    try:
        experiment_settings = network_block[experiment_name]
    except KeyError as exc:
        raise KeyError(
            f"Missing experiment settings for network='{selected_network}', experiment='{experiment_name}'"
        ) from exc
    if not isinstance(experiment_settings, dict):
        raise ValueError(
            f"Expected experiment settings object for network='{selected_network}', experiment='{experiment_name}'"
        )
    return deepcopy(experiment_settings)
