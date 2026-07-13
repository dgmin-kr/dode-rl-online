"""Readers for per-scenario test outputs used by figure notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import TEST_OUTPUTS_NPZ_NAME, TEST_STEP_HISTORY_CSV_NAME, load_test_outputs_npz


def trial_has_unified_outputs(trial_dir: str | Path) -> bool:
    trial_path = Path(trial_dir)
    return any(_scenario_output_dirs(trial_path))


def _outputs_from_npz(npz_path: Path) -> dict[str, dict[str, Any]]:
    payload = load_test_outputs_npz(npz_path)
    scenario_ids = tuple(str(value) for value in payload["scenario_ids"].tolist())
    od_labels = tuple(str(value) for value in payload["od_labels"].tolist())
    link_labels = tuple(str(value) for value in payload["link_labels"].tolist())
    observed_link_indices = np.asarray(payload["observed_link_indices"], dtype=np.int64).reshape(-1)
    observation_labels = (
        tuple(str(value) for value in payload["observation_labels"].tolist())
        if "observation_labels" in payload
        else tuple(f"observation_{obs_index}" for obs_index in range(len(observed_link_indices)))
    )
    outputs: dict[str, dict[str, Any]] = {}
    for index, scenario_id in enumerate(scenario_ids):
        output = {
            "scenario_id": scenario_id,
            "od_labels": od_labels,
            "link_labels": link_labels,
            "observed_link_indices": observed_link_indices.copy(),
            "observation_labels": observation_labels,
            "estimated_od_matrix": np.asarray(payload["estimated_od_matrices"][index], dtype=float),
            "simulated_link_flows": np.asarray(payload["simulated_link_flows"][index], dtype=float),
            "target_observations": np.asarray(payload["target_observations"][index], dtype=float),
            "simulated_observations": np.asarray(payload["simulated_observations"][index], dtype=float),
            "source": str(npz_path),
        }
        if "observation_scales" in payload:
            output["observation_scale"] = np.asarray(payload["observation_scales"][index], dtype=float)
        outputs[scenario_id] = output
    return outputs


def _scenario_output_dirs(trial_dir: Path) -> list[Path]:
    if not trial_dir.exists() or not trial_dir.is_dir():
        return []
    return sorted(
        path
        for path in trial_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / TEST_OUTPUTS_NPZ_NAME).exists()
    )


def load_trial_outputs(trial_dir: str | Path) -> dict[str, dict[str, Any]]:
    trial_dir = Path(trial_dir)
    scenario_dirs = _scenario_output_dirs(trial_dir)
    outputs: dict[str, dict[str, Any]] = {}
    for scenario_dir in scenario_dirs:
        outputs.update(_outputs_from_npz(scenario_dir / TEST_OUTPUTS_NPZ_NAME))
    return outputs


def observed_value_matrices(
    scenario_output: dict[str, Any],
    *,
    prefer_detector_observations: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    """Return detector target and simulated observation matrices for plots."""
    _ = prefer_detector_observations
    target_matrix = np.asarray(scenario_output["target_observations"], dtype=float)
    simulated_matrix = np.asarray(scenario_output["simulated_observations"], dtype=float)
    if target_matrix.shape != simulated_matrix.shape or target_matrix.ndim != 2:
        raise ValueError(
            "target_observations and simulated_observations must be matching 2-D arrays "
            f"({target_matrix.shape} vs {simulated_matrix.shape})."
        )
    labels = tuple(str(label) for label in scenario_output.get("observation_labels", ()))
    if len(labels) != target_matrix.shape[1]:
        labels = tuple(f"observation_{index}" for index in range(target_matrix.shape[1]))
    return target_matrix, simulated_matrix, labels, "detector_observation"


def available_scenario_ids(trial_dir: str | Path) -> list[str]:
    return sorted(load_trial_outputs(trial_dir).keys())


def scenario_folder_ids(trial_dir: str | Path) -> list[str]:
    return sorted(path.name for path in _scenario_output_dirs(Path(trial_dir)))


def has_scenario_output_folders(
    trial_dir: str | Path,
    scenario_ids: list[str] | tuple[str, ...] | None = None,
) -> bool:
    folder_ids = set(scenario_folder_ids(trial_dir))
    if not folder_ids:
        return False
    if scenario_ids is None:
        return True
    return all(str(scenario_id) in folder_ids for scenario_id in scenario_ids)


def read_saved_test_scenario_ids(trial_dir: str | Path) -> list[str] | None:
    trial_dir = Path(trial_dir)
    for metrics_name in ("test_scenario_metrics.csv", "scenario_metrics.csv"):
        metrics_path = trial_dir / metrics_name
        if not metrics_path.exists():
            continue
        try:
            metrics_df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if "scenario_id" not in metrics_df.columns:
            continue
        scenario_ids = [
            str(value)
            for value in metrics_df["scenario_id"].dropna().tolist()
            if str(value).strip()
        ]
        if scenario_ids:
            return list(dict.fromkeys(scenario_ids))

    for json_name in ("test_summary.json", "config_snapshot.json"):
        json_path = trial_dir / json_name
        if not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        selected_payload = payload.get("selected_test_scenario_ids")
        if isinstance(selected_payload, list):
            scenario_ids = [str(item) for item in selected_payload if str(item).strip()]
            if scenario_ids:
                return list(dict.fromkeys(scenario_ids))
    return None


def selected_or_available_scenario_ids(trial_dir: str | Path) -> list[str]:
    outputs = load_trial_outputs(trial_dir)
    if not outputs:
        return []
    selected_ids = read_saved_test_scenario_ids(trial_dir)
    if selected_ids:
        selected_available = [scenario_id for scenario_id in selected_ids if scenario_id in outputs]
        if selected_available:
            return selected_available
    return sorted(outputs.keys())


def load_scenario_output(trial_dir: str | Path, scenario_id: str) -> dict[str, Any]:
    outputs = load_trial_outputs(trial_dir)
    scenario_key = str(scenario_id)
    if scenario_key not in outputs:
        raise FileNotFoundError(f"Scenario {scenario_key!r} was not found in {Path(trial_dir)}.")
    return outputs[scenario_key]


def has_scenario_outputs(trial_dir: str | Path, scenario_ids: list[str] | tuple[str, ...] | None = None) -> bool:
    outputs = load_trial_outputs(trial_dir)
    if not outputs:
        return False
    if scenario_ids is None:
        return True
    return all(str(scenario_id) in outputs for scenario_id in scenario_ids)


def load_test_step_history(trial_dir: str | Path, scenario_id: str | None = None) -> pd.DataFrame:
    trial_dir = Path(trial_dir)
    scenario_dirs = _scenario_output_dirs(trial_dir)
    if scenario_dirs:
        frames: list[pd.DataFrame] = []
        for scenario_dir in scenario_dirs:
            if scenario_id is not None and scenario_dir.name != str(scenario_id):
                continue
            scenario_history_path = scenario_dir / TEST_STEP_HISTORY_CSV_NAME
            if not scenario_history_path.exists():
                continue
            try:
                frame = pd.read_csv(scenario_history_path)
            except Exception:
                continue
            frames.append(frame)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if scenario_id is not None and "scenario_id" in combined.columns:
                return combined[combined["scenario_id"].astype(str) == str(scenario_id)].reset_index(drop=True)
            return combined

    return pd.DataFrame()
