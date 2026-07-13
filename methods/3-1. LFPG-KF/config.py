from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

_PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from dnl.config import get_dnl_settings
from dnl.network.registry import canonical_network_name
from utils import ScenarioDataset, default_scenario_dataset_dir
from utils import (
    get_common_settings,
    get_experiment_settings,
    get_network_name,
    resolve_action_high_setting,
)


_EXPERIMENT_NAME = "lfpg_kf"
_DEFAULT_NETWORK_NAME = canonical_network_name(get_network_name(_EXPERIMENT_NAME))


def _resolve_settings(network_name: str) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical_name = canonical_network_name(network_name)
    return (
        canonical_name,
        get_common_settings(canonical_name),
        get_experiment_settings(_EXPERIMENT_NAME, canonical_name),
        get_dnl_settings(canonical_name),
    )


_INITIAL_NETWORK_NAME, _COMMON_SETTINGS, _EXPERIMENT_SETTINGS, _DNL_SETTINGS = _resolve_settings(_DEFAULT_NETWORK_NAME)


def _optional_float(settings: dict[str, Any], key: str) -> float | None:
    value = settings.get(key)
    return None if value is None else float(value)


class Config:
    PROJECT_DIR = _PROJECT_DIR
    METHOD_DIR = Path(__file__).resolve().parent
    SCENARIO_DATASET_DIR = default_scenario_dataset_dir(PROJECT_DIR, _DEFAULT_NETWORK_NAME)
    RESULT_DIR = METHOD_DIR / "results"

    NETWORK_NAME = _INITIAL_NETWORK_NAME
    TRAIN_SCENARIO_SPLIT = "train"
    TEST_SCENARIO_SPLIT = "test"

    ACTION_LOW = 0.0
    ACTION_HIGH = resolve_action_high_setting(NETWORK_NAME, _COMMON_SETTINGS, _EXPERIMENT_SETTINGS)
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

    TEST_STEP_RUNTIME_SECONDS = _optional_float(_COMMON_SETTINGS, "test_step_runtime_seconds")
    ALGORITHM = str(_EXPERIMENT_SETTINGS["algorithm"])
    KF_PARAMS: dict[str, Any] = dict(_EXPERIMENT_SETTINGS["kf_params"])
    LFPG_ENABLED = True

    CONFIG_SNAPSHOT_JSON_NAME = "config_snapshot.json"
    TEST_SUMMARY_JSON_NAME = "test_summary.json"
    SCENARIO_METRICS_CSV_NAME = "test_scenario_metrics.csv"

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def configure_network(cls, network_name: str) -> None:
        resolved_network_name, common_settings, experiment_settings, dnl_settings = _resolve_settings(network_name)
        cls.NETWORK_NAME = resolved_network_name
        cls.SCENARIO_DATASET_DIR = default_scenario_dataset_dir(cls.PROJECT_DIR, cls.NETWORK_NAME)
        cls.ACTION_HIGH = resolve_action_high_setting(cls.NETWORK_NAME, common_settings, experiment_settings)
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
        cls.TEST_STEP_RUNTIME_SECONDS = _optional_float(common_settings, "test_step_runtime_seconds")
        cls.ALGORITHM = str(experiment_settings["algorithm"])
        cls.KF_PARAMS = dict(experiment_settings["kf_params"])

    @classmethod
    def load_scenario_dataset(cls) -> ScenarioDataset:
        return ScenarioDataset(cls.SCENARIO_DATASET_DIR)


Config.configure_network(_DEFAULT_NETWORK_NAME)
