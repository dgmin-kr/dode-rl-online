"""Scenario dataset storage, split selection, and train/test protocol helpers."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_VERSION = 2
SPLIT_DATASET_FILENAMES = {
    "train": "train_dataset.npz",
    "test": "test_dataset.npz",
}
VALID_SPLITS = ("train", "test")
_SPLIT_SEED_OFFSETS = {
    "train": 100_000_000,
    "test": 200_000_000,
}


@dataclass(frozen=True)
class ScenarioSample:
    scenario_id: str
    split: str
    network_name: str
    target_observations: np.ndarray
    observed_link_indices: np.ndarray
    generation_seed: int
    observation_labels: tuple[str, ...] = ()

    @property
    def num_steps(self) -> int:
        return int(self.target_observations.shape[0])

    @property
    def num_observations(self) -> int:
        return int(self.target_observations.shape[1])


def default_scenario_dataset_dir(project_dir: Path, network_name: str) -> Path:
    # The publishable split datasets live directly under data/.
    # network_name is kept in the signature for existing launcher compatibility.
    return Path(project_dir).resolve() / "data"


def resolve_observed_link_indices(
    *,
    num_links: int,
    observed_link_indices: Any,
) -> np.ndarray:
    if observed_link_indices is None:
        raise ValueError("Scenario dataset manifest must define observed_link_indices.")
    indices = np.asarray(observed_link_indices, dtype=np.int64).reshape(-1)

    if indices.ndim != 1 or indices.size <= 0:
        raise ValueError("observed_link_indices must be a non-empty 1-D array.")
    if not bool(np.all(np.isfinite(indices))):
        raise ValueError("observed_link_indices contains non-finite values.")
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= int(num_links):
        raise ValueError(
            "observed_link_indices out of range: "
            f"valid range is [0, {int(num_links) - 1}], got {indices.tolist()}."
        )
    if len(set(int(index) for index in indices.tolist())) != int(indices.size):
        raise ValueError("observed_link_indices must not contain duplicates.")
    return indices.astype(np.int64, copy=False)


def build_protocol_simulation_seed(*, trial_seed: int, split: str, scenario_id: str) -> int:
    split_name = str(split).strip().lower()
    if split_name not in _SPLIT_SEED_OFFSETS:
        raise ValueError(f"Unsupported scenario split: {split!r}. Expected one of {VALID_SPLITS}.")

    scenario_digits = "".join(character for character in str(scenario_id) if character.isdigit())
    if scenario_digits:
        scenario_index = int(scenario_digits)
    else:
        scenario_index = sum((char_index + 1) * ord(character) for char_index, character in enumerate(str(scenario_id)))

    return int(trial_seed) + int(_SPLIT_SEED_OFFSETS[split_name]) + int(scenario_index)


def select_trial_scenario_ids(
    scenario_ids: list[str] | tuple[str, ...],
    *,
    trial_seed: int,
    max_scenarios: int = 1,
) -> tuple[str, ...]:
    ordered_ids = tuple(str(scenario_id) for scenario_id in scenario_ids)
    if not ordered_ids:
        return ()
    selection_count = min(max(int(max_scenarios), 0), len(ordered_ids))
    if selection_count <= 0:
        return ()
    rng = np.random.default_rng(int(trial_seed))
    permutation = rng.permutation(len(ordered_ids))
    return tuple(ordered_ids[int(index)] for index in permutation[:selection_count])


def resolve_trial_scenario_ids(
    scenario_ids: list[str] | tuple[str, ...],
    *,
    trial_seed: int,
    max_scenarios: int = 1,
    explicit_scenario_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    ordered_ids = tuple(str(scenario_id) for scenario_id in scenario_ids)
    if explicit_scenario_ids is not None:
        resolved_ids = tuple(str(scenario_id) for scenario_id in explicit_scenario_ids)
        valid_ids = set(ordered_ids)
        invalid_ids = tuple(scenario_id for scenario_id in resolved_ids if scenario_id not in valid_ids)
        if invalid_ids:
            raise ValueError(
                "Explicitly selected test scenario IDs are not present in the dataset split: "
                f"{list(invalid_ids)}"
            )
        return resolved_ids
    return select_trial_scenario_ids(
        ordered_ids,
        trial_seed=int(trial_seed),
        max_scenarios=int(max_scenarios),
    )


def assign_trial_scenario_ids(
    scenario_ids: list[str] | tuple[str, ...],
    *,
    trial_ids: list[int] | tuple[int, ...],
    max_scenarios: int = 1,
) -> dict[int, tuple[str, ...]]:
    ordered_ids = tuple(str(scenario_id) for scenario_id in scenario_ids)
    ordered_trials = tuple(int(trial_id) for trial_id in trial_ids)
    if not ordered_trials:
        return {}
    if len(set(ordered_trials)) != len(ordered_trials):
        raise ValueError("Trial IDs must be unique when assigning test scenarios without replacement.")

    selection_count = min(max(int(max_scenarios), 0), len(ordered_ids))
    if selection_count <= 0:
        return {trial_id: () for trial_id in ordered_trials}

    total_required = len(ordered_trials) * selection_count
    if total_required > len(ordered_ids):
        raise ValueError(
            "Not enough test scenarios to assign without replacement: "
            f"requested {total_required} selections for {len(ordered_trials)} trials, "
            f"but only {len(ordered_ids)} scenarios are available."
        )

    seed_sequence = np.random.SeedSequence([len(ordered_ids), selection_count, *ordered_trials])
    rng = np.random.default_rng(seed_sequence)
    permutation = rng.permutation(len(ordered_ids))

    assignments: dict[int, tuple[str, ...]] = {}
    cursor = 0
    for trial_id in ordered_trials:
        chunk = permutation[cursor : cursor + selection_count]
        assignments[int(trial_id)] = tuple(ordered_ids[int(index)] for index in chunk)
        cursor += selection_count
    return assignments


class ScenarioDataset:
    def __init__(self, dataset_dir: str | Path) -> None:
        input_path = Path(dataset_dir).resolve()
        explicit_split_path: Path | None = None
        if input_path.is_file():
            self.dataset_dir = input_path.parent
            explicit_split_path = input_path
        else:
            self.dataset_dir = input_path
        self.split_dataset_paths = {
            split_name: self.split_path(self.dataset_dir, split_name)
            for split_name in VALID_SPLITS
        }
        self.manifest_path = self.dataset_dir / "manifest.json"
        self._loaded_samples_by_split: dict[str, dict[str, Any]] = {}

        split_paths_to_load = [
            split_path
            for split_path in (
                [explicit_split_path] if explicit_split_path is not None else self.split_dataset_paths.values()
            )
            if split_path is not None and split_path.exists()
        ]
        if split_paths_to_load:
            manifest: dict[str, Any] | None = None
            for split_path in split_paths_to_load:
                split_manifest = self._load_split_dataset(split_path)
                if manifest is None:
                    manifest = split_manifest
                elif split_manifest != manifest:
                    raise ValueError(f"Split scenario dataset manifest mismatch: {split_path}")
            if explicit_split_path is not None:
                for split_name, split_path in self.split_dataset_paths.items():
                    if split_path != explicit_split_path and split_path.exists():
                        split_manifest = self._load_split_dataset(split_path, expected_split=split_name)
                        if split_manifest != manifest:
                            raise ValueError(f"Split scenario dataset manifest mismatch: {split_path}")
            if manifest is None:
                raise FileNotFoundError(f"No split scenario dataset files were found in {self.dataset_dir}")
        elif self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        else:
            raise FileNotFoundError(
                "Scenario dataset was not found. Expected either "
                f"{tuple(self.split_dataset_paths.values())} or {self.manifest_path}"
            )
        self.manifest = dict(manifest)
        self.network_name = str(manifest["network_name"])
        self._validate_manifest_scientific_status()
        self.link_labels = tuple(str(label) for label in manifest["link_labels"])
        self.od_pairs = tuple(tuple(int(value) for value in pair) for pair in manifest["od_pairs"])
        self.od_labels = tuple(str(label) for label in manifest["od_labels"])
        self.num_steps = int(manifest["num_steps"])
        self.num_links = int(manifest["num_links"])
        self.num_od = int(manifest["num_od"])
        self.observed_link_indices = resolve_observed_link_indices(
            num_links=self.num_links,
            observed_link_indices=manifest.get("observed_link_indices"),
        )
        self.num_observations = int(self.observed_link_indices.shape[0])
        raw_observation_labels = manifest.get("observation_labels")
        if raw_observation_labels is None:
            self.observation_labels = tuple(
                self.link_labels[int(link_index)] for link_index in self.observed_link_indices
            )
        else:
            self.observation_labels = tuple(str(label) for label in raw_observation_labels)
            if len(self.observation_labels) != self.num_observations:
                raise ValueError(
                    "Scenario dataset observation_labels length mismatch: "
                    f"expected {self.num_observations}, got {len(self.observation_labels)}."
                )
        self.manifest = dict(manifest)
        self.manifest["dataset_version"] = DATASET_VERSION
        self.manifest["observed_link_indices"] = [int(index) for index in self.observed_link_indices.tolist()]
        self.manifest["num_observations"] = self.num_observations
        self.manifest["observation_labels"] = list(self.observation_labels)
        self.splits = {
            str(split_name): tuple(str(item["scenario_id"]) for item in split_items)
            for split_name, split_items in manifest["splits"].items()
        }
        self._split_seed_map = {
            str(split_name): {
                str(item["scenario_id"]): int(item["generation_seed"])
                for item in split_items
            }
            for split_name, split_items in manifest["splits"].items()
        }

    def _validate_manifest_scientific_status(self) -> None:
        if str(self.network_name).strip().lower() != "melbourne_scats":
            return
        generation_settings = dict(self.manifest.get("generation_settings", {}))
        direction_mapping_status = str(generation_settings.get("direction_mapping_status", "")).strip().lower()
        link_flow_method = str(generation_settings.get("link_flow_method", "")).strip().lower()
        uses_legacy_forward_only_mapping = (
            "forward dnl link" in link_flow_method
            or "reverse links remain unobserved" in link_flow_method
            or direction_mapping_status in {"legacy_forward_only", "invalid_forward_only_legacy"}
        )
        if not uses_legacy_forward_only_mapping:
            return
        allow_legacy = str(os.environ.get("DODE_ALLOW_UNVERIFIED_MELBOURNE_SCATS", "")).strip().lower()
        if allow_legacy in {"1", "true", "yes", "y"}:
            return
        raise ValueError(
            "This Melbourne SCATS dataset uses the invalid legacy forward-only site-to-link mapping. "
            "Do not use it for training/evaluation. Rebuild data from SCATS detector configuration "
            "sheets and a reviewed detector-to-DNL-link assignment table, or set "
            "DODE_ALLOW_UNVERIFIED_MELBOURNE_SCATS=1 only for forensic inspection of the old artifacts."
        )

    def _load_split_dataset(self, split_path: Path, expected_split: str | None = None) -> dict[str, Any]:
        with np.load(split_path, allow_pickle=False) as payload:
            manifest_json = str(payload["manifest_json"].item())
            manifest = json.loads(manifest_json)
            split_name = str(payload["split_name"].item()) if "split_name" in payload else str(expected_split)
            split_name = self._validate_split(split_name)
            if expected_split is not None and split_name != self._validate_split(expected_split):
                raise ValueError(f"Expected split {expected_split!r}, but {split_path} contains {split_name!r}.")

            scenario_ids = tuple(str(value) for value in np.asarray(payload["scenario_ids"]).tolist())
            if "target_observations" not in payload:
                raise ValueError(
                    f"Split scenario dataset {split_path} is missing target_observations. "
                    "The active data contract does not allow full-link target fallback."
                )
            target_observations = np.asarray(payload["target_observations"], dtype=np.float32)
            if target_observations.shape[0] != len(scenario_ids):
                raise ValueError(f"Split scenario dataset {split_path} has inconsistent observation sample counts.")
            if target_observations.ndim != 3:
                raise ValueError(
                    "target_observations must be shaped [num_scenarios, num_steps, num_observations]; "
                    f"got {target_observations.shape} in {split_path}."
                )
            self._loaded_samples_by_split[split_name] = {
                "path": split_path,
                "scenario_ids": scenario_ids,
                "id_to_index": {scenario_id: index for index, scenario_id in enumerate(scenario_ids)},
                "target_observations": target_observations,
            }
            return manifest

    def get_split_ids(self, split: str) -> tuple[str, ...]:
        split_name = self._validate_split(split)
        return tuple(self.splits.get(split_name, ()))

    def count(self, split: str) -> int:
        return int(len(self.get_split_ids(split)))

    def sample_id(self, split: str, rng: np.random.Generator) -> str:
        scenario_ids = self.get_split_ids(split)
        if not scenario_ids:
            raise ValueError(f"Scenario split {split!r} is empty.")
        selected_index = int(rng.integers(0, len(scenario_ids)))
        return scenario_ids[selected_index]

    def select_trial_ids(
        self,
        split: str,
        *,
        trial_seed: int,
        max_scenarios: int = 1,
    ) -> tuple[str, ...]:
        return select_trial_scenario_ids(
            self.get_split_ids(split),
            trial_seed=int(trial_seed),
            max_scenarios=int(max_scenarios),
        )

    def scenario_path(self, split: str, scenario_id: str) -> Path:
        split_name = self._validate_split(split)
        if split_name in self._loaded_samples_by_split:
            return Path(self._loaded_samples_by_split[split_name]["path"])
        return self.dataset_dir / split_name / f"{scenario_id}.npz"

    def load(self, split: str, scenario_id: str) -> ScenarioSample:
        split_name = self._validate_split(split)
        if split_name in self._loaded_samples_by_split:
            loaded_split = self._loaded_samples_by_split[split_name]
            sample_index = loaded_split["id_to_index"].get(str(scenario_id))
            if sample_index is None:
                raise FileNotFoundError(
                    f"Scenario sample {scenario_id!r} was not found in split {split_name!r}: "
                    f"{loaded_split['path']}"
                )
            return ScenarioSample(
                scenario_id=str(scenario_id),
                split=split_name,
                network_name=self.network_name,
                target_observations=np.asarray(loaded_split["target_observations"][sample_index], dtype=np.float32),
                observed_link_indices=self.observed_link_indices.copy(),
                generation_seed=int(self._split_seed_map[split_name][scenario_id]),
                observation_labels=tuple(self.observation_labels),
            )

        path = self.scenario_path(split_name, scenario_id)
        if not path.exists():
            raise FileNotFoundError(f"Scenario sample was not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if "target_observations" not in payload:
                raise ValueError(
                    f"Scenario sample {path} is missing target_observations. "
                    "The active data contract does not allow full-link target fallback."
                )
            target_observations = np.asarray(payload["target_observations"], dtype=np.float32)
        return ScenarioSample(
            scenario_id=str(scenario_id),
            split=split_name,
            network_name=self.network_name,
            target_observations=target_observations,
            observed_link_indices=self.observed_link_indices.copy(),
            generation_seed=int(self._split_seed_map[split_name][scenario_id]),
            observation_labels=tuple(self.observation_labels),
        )

    def sample(self, split: str, rng: np.random.Generator) -> ScenarioSample:
        scenario_id = self.sample_id(split, rng)
        return self.load(split, scenario_id)

    def load_split(self, split: str, limit: int | None = None) -> list[ScenarioSample]:
        split_name = self._validate_split(split)
        scenario_ids = list(self.get_split_ids(split_name))
        if limit is not None:
            scenario_ids = scenario_ids[: max(int(limit), 0)]
        return [self.load(split_name, scenario_id) for scenario_id in scenario_ids]

    def _validate_split(self, split: str) -> str:
        split_name = str(split).strip().lower()
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unsupported scenario split: {split!r}. Expected one of {VALID_SPLITS}.")
        return split_name

    @staticmethod
    def build_manifest(
        *,
        network_name: str,
        od_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        link_labels: list[str] | tuple[str, ...],
        num_steps: int,
        num_od: int,
        num_links: int,
        dnl_settings: dict[str, Any],
        generation_settings: dict[str, Any],
        splits: dict[str, list[dict[str, Any]]],
        observed_link_indices: list[int] | tuple[int, ...] | np.ndarray,
        observation_labels: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        od_pairs_list = [[int(origin), int(destination)] for origin, destination in od_pairs]
        od_labels = [f"{origin}->{destination}" for origin, destination in od_pairs]
        indices = resolve_observed_link_indices(
            num_links=int(num_links),
            observed_link_indices=observed_link_indices,
        )
        if observation_labels is None:
            labels = [str(link_labels[int(index)]) for index in indices]
        else:
            labels = [str(label) for label in observation_labels]
            if len(labels) != int(indices.shape[0]):
                raise ValueError(
                    "observation_labels length must match observed_link_indices: "
                    f"expected {int(indices.shape[0])}, got {len(labels)}."
                )
        manifest = {
            "dataset_version": DATASET_VERSION,
            "network_name": str(network_name),
            "num_steps": int(num_steps),
            "num_od": int(num_od),
            "num_links": int(num_links),
            "num_observations": int(indices.shape[0]),
            "od_pairs": od_pairs_list,
            "od_labels": od_labels,
            "link_labels": [str(label) for label in link_labels],
            "observed_link_indices": [int(index) for index in indices.tolist()],
            "observation_labels": labels,
            "dnl_settings": dict(dnl_settings),
            "generation_settings": dict(generation_settings),
            "splits": splits,
        }
        return manifest

    @staticmethod
    def split_path(dataset_dir: str | Path, split: str) -> Path:
        split_name = str(split).strip().lower()
        if split_name not in SPLIT_DATASET_FILENAMES:
            raise ValueError(f"Unsupported scenario split: {split!r}. Expected one of {VALID_SPLITS}.")
        return Path(dataset_dir).resolve() / SPLIT_DATASET_FILENAMES[split_name]

    @staticmethod
    def write_manifest(dataset_dir: str | Path, manifest: dict[str, Any]) -> Path:
        dataset_dir = Path(dataset_dir).resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    @staticmethod
    def write_split_datasets(
        dataset_dir: str | Path,
        *,
        manifest: dict[str, Any],
        samples_by_split: dict[str, list[ScenarioSample]],
    ) -> dict[str, Path]:
        dataset_dir = Path(dataset_dir).resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        num_steps = int(manifest["num_steps"])
        num_observations = int(manifest["num_observations"])
        written_paths: dict[str, Path] = {}

        for split_name in VALID_SPLITS:
            samples = list(samples_by_split.get(split_name, []))
            scenario_ids = [str(sample.scenario_id) for sample in samples]
            generation_seeds = [int(sample.generation_seed) for sample in samples]
            if samples:
                target_observations = np.stack(
                    [np.asarray(sample.target_observations, dtype=np.float32) for sample in samples],
                    axis=0,
                )
            else:
                target_observations = np.empty((0, num_steps, num_observations), dtype=np.float32)
            if target_observations.shape[1:] != (num_steps, num_observations):
                raise ValueError(
                    f"Split {split_name!r} target_observations shape mismatch: "
                    f"expected (*, {num_steps}, {num_observations}), got {target_observations.shape}."
                )

            split_path = ScenarioDataset.split_path(dataset_dir, split_name)
            temp_path = split_path.with_name(f".{split_path.name}.tmp")
            if temp_path.exists():
                temp_path.unlink()
            payload = {
                "manifest_json": np.asarray(json.dumps(manifest, indent=2), dtype=np.str_),
                "split_name": np.asarray(split_name, dtype=np.str_),
                "scenario_ids": np.asarray(scenario_ids, dtype=np.str_),
                "generation_seeds": np.asarray(generation_seeds, dtype=np.int64),
                "target_observations": np.asarray(target_observations, dtype=np.float32),
            }
            with temp_path.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            temp_path.replace(split_path)
            written_paths[split_name] = split_path

        return written_paths

    @staticmethod
    def consolidate(
        dataset_dir: str | Path,
        *,
        remove_source_files: bool = True,
    ) -> dict[str, Path]:
        dataset_dir = Path(dataset_dir).resolve()
        dataset = ScenarioDataset(dataset_dir)
        samples_by_split = {
            split_name: dataset.load_split(split_name)
            for split_name in VALID_SPLITS
        }
        split_paths = ScenarioDataset.write_split_datasets(
            dataset_dir,
            manifest=dataset.manifest,
            samples_by_split=samples_by_split,
        )

        split_dataset = ScenarioDataset(dataset_dir)
        for split_name in VALID_SPLITS:
            if split_dataset.get_split_ids(split_name) != dataset.get_split_ids(split_name):
                raise RuntimeError(f"Scenario split {split_name!r} failed ID validation.")
            if split_dataset.count(split_name) != dataset.count(split_name):
                raise RuntimeError(f"Scenario split {split_name!r} failed count validation.")

        if remove_source_files:
            for split_name in VALID_SPLITS:
                split_dir = dataset_dir / split_name
                if split_dir.exists():
                    shutil.rmtree(split_dir)
            manifest_path = dataset_dir / "manifest.json"
            if manifest_path.exists():
                manifest_path.unlink()

        return split_paths

    @staticmethod
    def write_sample(
        dataset_dir: str | Path,
        *,
        split: str,
        scenario_id: str,
        target_observations: np.ndarray,
    ) -> Path:
        split_name = str(split).strip().lower()
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unsupported scenario split: {split!r}. Expected one of {VALID_SPLITS}.")
        split_dir = Path(dataset_dir).resolve() / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        sample_path = split_dir / f"{scenario_id}.npz"
        np.savez_compressed(
            sample_path,
            target_observations=np.asarray(target_observations, dtype=np.float32),
        )
        return sample_path
