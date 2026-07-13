"""Unified numeric result files for training/evaluation trials."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TEST_OUTPUTS_NPZ_NAME = "test_outputs.npz"
TEST_STEP_HISTORY_CSV_NAME = "test_step_history.csv"
TEST_SCENARIO_METRICS_CSV_NAME = "test_scenario_metrics.csv"


@dataclass(frozen=True)
class TestOutputRecord:
    scenario_id: str
    split: str
    scenario_generation_seed: int | None
    simulation_seed: int | None
    od_labels: tuple[str, ...]
    link_labels: tuple[str, ...]
    estimated_od_matrix: np.ndarray
    simulated_link_flows: np.ndarray
    observed_link_indices: np.ndarray
    observation_labels: tuple[str, ...]
    target_observations: np.ndarray
    simulated_observations: np.ndarray
    observation_scale: np.ndarray | None = None
    step_rows: tuple[dict[str, Any], ...] = tuple()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    path = Path(path)
    rows = list(rows)
    if not rows:
        if path.exists():
            path.unlink()
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(path: str | Path, labels: tuple[str, ...] | list[str], matrix: np.ndarray) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_step", *[str(label) for label in labels]])
        for time_step, row in enumerate(np.asarray(matrix, dtype=float)):
            writer.writerow([time_step, *row.tolist()])


def _resolve_record_indices(record: TestOutputRecord, num_links: int) -> np.ndarray:
    indices = np.asarray(record.observed_link_indices, dtype=np.int64).reshape(-1)
    if indices.size <= 0:
        raise ValueError(f"observed_link_indices is empty for scenario {record.scenario_id!r}.")
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= int(num_links):
        raise ValueError(
            f"observed_link_indices out of range for scenario {record.scenario_id!r}: "
            f"valid range is [0, {int(num_links) - 1}], got {indices.tolist()}."
        )
    if len(set(int(index) for index in indices.tolist())) != int(indices.size):
        raise ValueError(f"observed_link_indices contains duplicates for scenario {record.scenario_id!r}.")
    return indices


def write_test_outputs_npz(path: str | Path, records: list[TestOutputRecord] | tuple[TestOutputRecord, ...]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        raise ValueError("Cannot write test outputs without at least one scenario record.")

    first = records[0]
    od_labels = tuple(str(label) for label in first.od_labels)
    link_labels = tuple(str(label) for label in first.link_labels)
    observed_link_indices = _resolve_record_indices(first, len(link_labels))
    observation_labels = tuple(str(label) for label in first.observation_labels)
    if len(observation_labels) != int(observed_link_indices.shape[0]):
        raise ValueError(
            "observation_labels length must match observed_link_indices: "
            f"{len(observation_labels)} vs {int(observed_link_indices.shape[0])}."
        )

    estimated_od_matrices: list[np.ndarray] = []
    simulated_link_flows: list[np.ndarray] = []
    target_observations: list[np.ndarray] = []
    simulated_observations: list[np.ndarray] = []
    observation_scales: list[np.ndarray] = []

    first_target = np.asarray(first.target_observations, dtype=np.float32)
    first_simulated = np.asarray(first.simulated_observations, dtype=np.float32)
    if first_target.ndim != 2 or first_simulated.shape != first_target.shape:
        raise ValueError("target_observations and simulated_observations must be matching 2-D arrays.")
    if first_target.shape[1] != int(observed_link_indices.shape[0]):
        raise ValueError(
            "target_observations width must match observed_link_indices length: "
            f"{first_target.shape[1]} vs {int(observed_link_indices.shape[0])}."
        )

    for record in records:
        if tuple(record.od_labels) != od_labels:
            raise ValueError(f"OD labels differ for scenario {record.scenario_id!r}.")
        if tuple(record.link_labels) != link_labels:
            raise ValueError(f"Link labels differ for scenario {record.scenario_id!r}.")
        if not np.array_equal(_resolve_record_indices(record, len(link_labels)), observed_link_indices):
            raise ValueError(f"observed_link_indices differ for scenario {record.scenario_id!r}.")
        if tuple(str(label) for label in record.observation_labels) != observation_labels:
            raise ValueError(f"Observation labels differ for scenario {record.scenario_id!r}.")

        target_values = np.asarray(record.target_observations, dtype=np.float32)
        simulated_values = np.asarray(record.simulated_observations, dtype=np.float32)
        if target_values.shape != first_target.shape or simulated_values.shape != target_values.shape:
            raise ValueError(
                f"Observation output shape differs for scenario {record.scenario_id!r}: "
                f"expected {first_target.shape}, got {target_values.shape}/{simulated_values.shape}."
            )
        estimated_od_matrices.append(np.asarray(record.estimated_od_matrix, dtype=np.float32))
        simulated_link_flows.append(np.asarray(record.simulated_link_flows, dtype=np.float32))
        target_observations.append(target_values)
        simulated_observations.append(simulated_values)
        if record.observation_scale is not None:
            scale = np.asarray(record.observation_scale, dtype=np.float32).reshape(-1)
            if scale.shape != (len(observation_labels),):
                raise ValueError(
                    f"observation_scale length differs for scenario {record.scenario_id!r}: "
                    f"expected {len(observation_labels)}, got {scale.shape[0]}."
                )
            observation_scales.append(scale)

    payload: dict[str, np.ndarray] = {
        "scenario_ids": np.asarray([record.scenario_id for record in records], dtype=np.str_),
        "splits": np.asarray([record.split for record in records], dtype=np.str_),
        "scenario_generation_seeds": np.asarray(
            [-1 if record.scenario_generation_seed is None else int(record.scenario_generation_seed) for record in records],
            dtype=np.int64,
        ),
        "simulation_seeds": np.asarray(
            [-1 if record.simulation_seed is None else int(record.simulation_seed) for record in records],
            dtype=np.int64,
        ),
        "od_labels": np.asarray(od_labels, dtype=np.str_),
        "link_labels": np.asarray(link_labels, dtype=np.str_),
        "observation_labels": np.asarray(observation_labels, dtype=np.str_),
        "observed_link_indices": np.asarray(observed_link_indices, dtype=np.int64),
        "estimated_od_matrices": np.stack(estimated_od_matrices, axis=0),
        "simulated_link_flows": np.stack(simulated_link_flows, axis=0),
        "target_observations": np.stack(target_observations, axis=0),
        "simulated_observations": np.stack(simulated_observations, axis=0),
    }
    if observation_scales:
        if len(observation_scales) != len(records):
            raise ValueError("observation_scale must be present for every scenario or for none of them.")
        payload["observation_scales"] = np.stack(observation_scales, axis=0)

    temp_path = path.with_name(f".{path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temp_path.replace(path)
    return path


def write_test_step_history_csv(path: str | Path, records: list[TestOutputRecord] | tuple[TestOutputRecord, ...]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        for step_row in record.step_rows:
            rows.append({"scenario_id": record.scenario_id, **dict(step_row)})
    write_rows_csv(path, rows)


def load_test_outputs_npz(path_or_trial_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(path_or_trial_dir)
    if path.is_dir():
        path = path / TEST_OUTPUTS_NPZ_NAME
    if not path.exists():
        raise FileNotFoundError(f"Unified test output file was not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}
