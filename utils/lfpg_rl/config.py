from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import sys

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODULE_DIR.parents[1]
_METHODS_ROOT_DIR = _PROJECT_DIR / "methods"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from utils import ScenarioDataset, default_scenario_dataset_dir
from utils import (
    get_common_settings,
    get_experiment_settings,
    get_network_name,
    get_section_settings,
    resolve_action_high_setting,
)
from dnl.config import get_dnl_settings
from dnl.network.registry import canonical_network_name, get_default_od_pairs


_VARIANT_FOLDER_TO_EXPERIMENT = {
    "1-1. LFPG-RL": "lfpg_rl",
    "1-2. PPO": "ppo_baseline",
}
_DEFAULT_METHOD_DIR = _METHODS_ROOT_DIR / "1-1. LFPG-RL"
_METHOD_DIR_ENV_VAR = "LFPG_RL_METHOD_DIR"


def _resolve_experiment_name(train_dir: Path) -> str:
    return _VARIANT_FOLDER_TO_EXPERIMENT.get(train_dir.name, "lfpg_rl")


_TRAIN_DIR = Path(os.environ.get(_METHOD_DIR_ENV_VAR, str(_DEFAULT_METHOD_DIR))).resolve()
_EXPERIMENT_NAME = _resolve_experiment_name(_TRAIN_DIR)
_DEFAULT_NETWORK_NAME = canonical_network_name(get_network_name(_EXPERIMENT_NAME))


def _resolve_settings(
    network_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical_name = canonical_network_name(network_name)
    return (
        canonical_name,
        get_common_settings(canonical_name),
        get_experiment_settings(_EXPERIMENT_NAME, canonical_name),
        get_section_settings("rl", canonical_name, apply_runtime_env_overrides=True),
        get_section_settings("ppo_params", canonical_name),
        get_dnl_settings(canonical_name),
    )


def _compose_rl_runtime_params(
    rl_settings: dict[str, Any],
    lfpg_settings: dict[str, Any],
) -> dict[str, Any]:
    runtime_params = dict(rl_settings.get("env_params", {}))
    runtime_params.update(dict(lfpg_settings))
    return runtime_params


(
    _INITIAL_NETWORK_NAME,
    _COMMON_SETTINGS,
    _EXPERIMENT_SETTINGS,
    _RL_SETTINGS,
    _PPO_SETTINGS,
    _DNL_SETTINGS,
) = _resolve_settings(_DEFAULT_NETWORK_NAME)


def _optional_int(settings: dict[str, Any], key: str) -> int | None:
    value = settings.get(key)
    return None if value is None else int(value)


def _optional_bool(settings: dict[str, Any], key: str) -> bool | None:
    value = settings.get(key)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Expected boolean setting for {key!r}, received {value!r}.")
    return None if value is None else bool(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean environment variable {name!r}, received {value!r}.")


def _progress_logging_enabled(network_name: str, dnl_settings: dict[str, Any] | None = None) -> bool:
    canonical_name = canonical_network_name(network_name)
    settings = get_dnl_settings(canonical_name) if dnl_settings is None else dnl_settings
    configured = _optional_bool(settings, "progress_logging")
    default = False if configured is None else configured
    return _env_bool("DODE_DNL_PROGRESS_LOG", default)


class Config:
    PROJECT_DIR = _PROJECT_DIR
    TRAIN_DIR = _TRAIN_DIR
    EXPERIMENT_NAME = _EXPERIMENT_NAME
    SCENARIO_DATASET_DIR = default_scenario_dataset_dir(PROJECT_DIR, _DEFAULT_NETWORK_NAME)
    NETWORK_NAME = _DEFAULT_NETWORK_NAME
    TRAIN_SCENARIO_SPLIT = "train"
    TEST_SCENARIO_SPLIT = "test"

    RESULT_DIR = TRAIN_DIR / "results"
    MODEL_DIR = RESULT_DIR / "models"

    NUM_OD = len(get_default_od_pairs(NETWORK_NAME))
    ACTION_LOW = 0.0
    ACTION_HIGH = resolve_action_high_setting(NETWORK_NAME, _COMMON_SETTINGS, _RL_SETTINGS)
    ROUTE_CHOICE_MODE: str = str(_DNL_SETTINGS["route_choice_mode"])
    STOCHASTIC_LOGIT_SCALE: float = float(_DNL_SETTINGS["stochastic_logit_scale"])
    DNL_SAMPLE_ROUTE_CHOICES: bool = bool(_DNL_SETTINGS["sample_route_choices"])
    DNL_ROUTE_CHOICE_SAMPLING_UNIT: float = float(_DNL_SETTINGS["route_choice_sampling_unit"])
    DNL_MAX_PATHS_PER_OD: int = int(_DNL_SETTINGS["max_paths_per_od"])
    DNL_DUE_MAX_ITERATIONS: int = int(_DNL_SETTINGS["due_max_iterations"])
    DNL_DUE_TOLERANCE: float = float(_DNL_SETTINGS["due_tolerance"])
    DNL_CLEARANCE_STEPS: int = int(_DNL_SETTINGS["clearance_steps"])
    DNL_PARALLEL_KERNELS: bool | str | None = _DNL_SETTINGS["use_parallel_kernels"]
    DNL_NUMBA_THREADS: int | None = _DNL_SETTINGS["numba_threads"]
    DNL_PROGRESS_LOGGING: bool = _progress_logging_enabled(NETWORK_NAME, _DNL_SETTINGS)

    # Melbourne SCATS uses 24 AM-peak 15-minute steps. Each environment step
    # re-solves the DNL model, so conservative vectorization remains preferable.
    MAX_RUNTIME_SECONDS = _optional_int(_COMMON_SETTINGS, "max_runtime_seconds")
    # Internal safety cap used only when runtime-only mode is active.
    RUNTIME_ONLY_TOTAL_TIMESTEPS = 2_000_000_000
    NUM_ENVS = _optional_int(_RL_SETTINGS, "num_envs")
    USE_SUBPROC = _optional_bool(_RL_SETTINGS, "use_subproc")

    ALGORITHM: str = str(_RL_SETTINGS["algorithm"])
    # Tuned for a medium-horizon OD-estimation task:
    # - gamma remains fairly high for 36-step delayed assignment
    # - one full episode per rollout for cleaner advantage estimates
    # - modest entropy bonus to keep exploration alive without destabilizing
    #   the inverse problem
    # - a moderate number of subprocess environments to balance throughput and cost
    PPO_PARAMS: dict[str, Any] = dict(_PPO_SETTINGS)
    RL_ENV_PARAMS: dict[str, Any] = dict(_RL_SETTINGS.get("env_params", {}))
    LFPG_PARAMS: dict[str, Any] = dict(_EXPERIMENT_SETTINGS)
    RL_RUNTIME_PARAMS: dict[str, Any] = _compose_rl_runtime_params(_RL_SETTINGS, _EXPERIMENT_SETTINGS)

    REWARD_CSV_NAME = "episode_rewards.csv"
    STEP_METRICS_CSV_NAME = "step_metrics.csv"
    CONFIG_SNAPSHOT_JSON_NAME = "config_snapshot.json"
    TRAINING_SUMMARY_JSON_NAME = "training_summary.json"
    TEST_SUMMARY_JSON_NAME = "test_summary.json"
    SCENARIO_METRICS_CSV_NAME = "test_scenario_metrics.csv"
    FINAL_MODEL_NAME = "final_model.pt"
    LATEST_MODEL_NAME = "latest_model.pt"
    TEST_MAX_SCENARIOS = _RL_SETTINGS.get("test_max_scenarios")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def configure_network(cls, network_name: str) -> None:
        resolved_network_name, common_settings, lfpg_settings, rl_settings, ppo_settings, dnl_settings = (
            _resolve_settings(network_name)
        )
        cls.NETWORK_NAME = resolved_network_name
        cls.SCENARIO_DATASET_DIR = default_scenario_dataset_dir(cls.PROJECT_DIR, cls.NETWORK_NAME)
        cls.ACTION_HIGH = resolve_action_high_setting(cls.NETWORK_NAME, common_settings, rl_settings)
        cls.NUM_OD = len(get_default_od_pairs(cls.NETWORK_NAME))
        cls.ROUTE_CHOICE_MODE = str(dnl_settings["route_choice_mode"])
        cls.STOCHASTIC_LOGIT_SCALE = float(dnl_settings["stochastic_logit_scale"])
        cls.DNL_SAMPLE_ROUTE_CHOICES = bool(dnl_settings["sample_route_choices"])
        cls.DNL_ROUTE_CHOICE_SAMPLING_UNIT = float(dnl_settings["route_choice_sampling_unit"])
        cls.DNL_MAX_PATHS_PER_OD = int(dnl_settings["max_paths_per_od"])
        cls.DNL_DUE_MAX_ITERATIONS = int(dnl_settings["due_max_iterations"])
        cls.DNL_DUE_TOLERANCE = float(dnl_settings["due_tolerance"])
        cls.DNL_CLEARANCE_STEPS = int(dnl_settings["clearance_steps"])
        cls.DNL_PARALLEL_KERNELS = dnl_settings["use_parallel_kernels"]
        cls.DNL_NUMBA_THREADS = dnl_settings["numba_threads"]
        cls.DNL_PROGRESS_LOGGING = _progress_logging_enabled(cls.NETWORK_NAME, dnl_settings)
        cls.MAX_RUNTIME_SECONDS = _optional_int(common_settings, "max_runtime_seconds")
        cls.NUM_ENVS = _optional_int(rl_settings, "num_envs")
        cls.USE_SUBPROC = _optional_bool(rl_settings, "use_subproc")
        cls.ALGORITHM = str(rl_settings["algorithm"])
        supported_algorithms = {"LFPG-RL"}
        if cls.ALGORITHM not in supported_algorithms:
            raise ValueError(
                "Unsupported LFPG-RL algorithm. "
                f"Received algorithm={cls.ALGORITHM!r} for experiment={cls.EXPERIMENT_NAME!r}, "
                f"network={cls.NETWORK_NAME!r}."
            )
        cls.PPO_PARAMS = dict(ppo_settings)
        cls.RL_ENV_PARAMS = dict(rl_settings.get("env_params", {}))
        cls.LFPG_PARAMS = dict(lfpg_settings)
        cls.RL_RUNTIME_PARAMS = _compose_rl_runtime_params(rl_settings, lfpg_settings)
        cls.TEST_MAX_SCENARIOS = rl_settings.get("test_max_scenarios")

    @classmethod
    def load_scenario_dataset(cls) -> ScenarioDataset:
        return ScenarioDataset(cls.SCENARIO_DATASET_DIR)


def configure_method_dir(method_dir: str | Path | None = None) -> None:
    global _TRAIN_DIR, _EXPERIMENT_NAME

    _TRAIN_DIR = Path(method_dir or os.environ.get(_METHOD_DIR_ENV_VAR, str(_DEFAULT_METHOD_DIR))).resolve()
    _EXPERIMENT_NAME = _resolve_experiment_name(_TRAIN_DIR)
    os.environ[_METHOD_DIR_ENV_VAR] = str(_TRAIN_DIR)

    Config.TRAIN_DIR = _TRAIN_DIR
    Config.EXPERIMENT_NAME = _EXPERIMENT_NAME
    Config.RESULT_DIR = Config.TRAIN_DIR / "results"
    Config.MODEL_DIR = Config.RESULT_DIR / "models"
    Config.configure_network(canonical_network_name(get_network_name(_EXPERIMENT_NAME)))


configure_method_dir(_TRAIN_DIR)

