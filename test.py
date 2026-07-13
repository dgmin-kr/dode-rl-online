from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from dnl.network.registry import canonical_network_name, get_network_display_name
from utils import ScenarioDataset, default_scenario_dataset_dir
from utils import (
    DEFAULT_NETWORK_ENV_VAR,
    TEST_OUTPUTS_NPZ_NAME,
    TEST_SCENARIO_METRICS_CSV_NAME,
    TEST_STEP_HISTORY_CSV_NAME,
    TEST_STEP_RUNTIME_SECONDS_ENV_VAR,
    TestOutputRecord,
    load_test_outputs_npz,
)


PROJECT_DIR = Path(__file__).resolve().parent
FORCE_KILL_WAIT_SECONDS = 10
WINDOWS_NEW_PROCESS_GROUP = (
    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
)

# This project is fixed to the final refined Melbourne SCATS network.
DEFAULT_NETWORK_NAME = "melbourne_scats"

# Optional explicit RL checkpoints to evaluate.
# Fill these with final_model.pt/latest_model.pt paths when evaluating saved RL models.
LFPG_RL_POLICY = ""
RL_POLICY = ""

# Select only the methods to run here.
# Available options:
#   "lfpg_rl"
#   "ppo_baseline"
#   "lfpg_gd"
#   "w_spsa"
#   "lfpg_kf"
#   "kf"

# Methods run by default. RL methods fail fast unless their policy paths above are configured.
TEST_METHODS = ["lfpg_gd", "w_spsa", "lfpg_kf", "kf"]
DEFAULT_TEST_STEP_RUNTIME_SECONDS = 60 * 15
TEST_OUTPUT_SUFFIX = "test"
NUM_PARALLEL_ENV = 6
# True: launch every selected method for NUM_PARALLEL_ENV scenarios per round.
# False: run each selected method over the full test split sequentially.
DEFAULT_PARALLEL_ALL_EVALUATIONS = True


METHOD_SPECS = {
    "lfpg_rl": {
        "script": PROJECT_DIR / "methods" / "1-1. LFPG-RL" / "run.py",
        "supports_rl_overrides": True,
    },
    "ppo_baseline": {
        "script": PROJECT_DIR / "methods" / "1-2. PPO" / "run.py",
        "supports_rl_overrides": True,
    },
    "lfpg_gd": {
        "script": PROJECT_DIR / "methods" / "2-1. LFPG-GD" / "run.py",
        "supports_rl_overrides": False,
    },
    "w_spsa": {
        "script": PROJECT_DIR / "methods" / "2-2. W-SPSA" / "run.py",
        "supports_rl_overrides": False,
    },
    "lfpg_kf": {
        "script": PROJECT_DIR / "methods" / "3-1. LFPG-KF" / "run.py",
        "supports_rl_overrides": False,
    },
    "kf": {
        "script": PROJECT_DIR / "methods" / "3-2. KF" / "run.py",
        "supports_rl_overrides": False,
    },
}


@dataclass(frozen=True)
class TestLauncherSettings:
    test_step_runtime_seconds: float
    num_steps_per_scenario: int


@dataclass(frozen=True)
class ParallelScenarioTask:
    method: str
    scenario_id: str
    output_suffix: str
    output_dir: Path
    command: list[str]
    log_path: Path


def resolve_launcher_settings(
    args: argparse.Namespace,
    scenario_dataset: ScenarioDataset | None = None,
) -> TestLauncherSettings:
    if scenario_dataset is None:
        scenario_dataset = load_scenario_dataset(args)
    test_step_runtime_seconds = (
        float(args.step_runtime_seconds)
        if args.step_runtime_seconds is not None
        else float(DEFAULT_TEST_STEP_RUNTIME_SECONDS)
    )
    if test_step_runtime_seconds < 0.0:
        raise ValueError("--step-runtime-seconds must be non-negative.")
    num_steps_per_scenario = max(int(scenario_dataset.num_steps), 1)
    return TestLauncherSettings(
        test_step_runtime_seconds=test_step_runtime_seconds,
        num_steps_per_scenario=num_steps_per_scenario,
    )


def build_child_env(network_name: str, launcher_settings: TestLauncherSettings) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env[DEFAULT_NETWORK_ENV_VAR] = canonical_network_name(network_name)
    env[TEST_STEP_RUNTIME_SECONDS_ENV_VAR] = str(launcher_settings.test_step_runtime_seconds)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate selected methods on every scenario in test_dataset.npz. "
            "The default scheduling mode is controlled by DEFAULT_PARALLEL_ALL_EVALUATIONS."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_SPECS.keys()),
        default=list(TEST_METHODS),
        help=(
            "Which methods to launch. "
            f"Default: {' '.join(TEST_METHODS)}. "
            "RL methods require LFPG_RL_POLICY or RL_POLICY at the top of test.py."
        ),
    )
    parser.add_argument(
        "--network",
        type=str,
        default=DEFAULT_NETWORK_NAME,
        help="Network passed through to every run.py. Only melbourne_scats is supported.",
    )
    parser.add_argument(
        "--step-runtime-seconds",
        type=float,
        default=None,
        help=f"Optional per-step runtime cap for each evaluated test step. Default: {DEFAULT_TEST_STEP_RUNTIME_SECONDS}.",
    )
    parser.add_argument(
        "--num-parallel-env",
        type=int,
        default=None,
        help=f"Scenario parallelism per method in --parallel mode. Default: {NUM_PARALLEL_ENV}.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Launch every selected method concurrently for each scenario batch. "
            "This starts len(methods) * --num-parallel-env child Python processes per round. "
            "Overrides DEFAULT_PARALLEL_ALL_EVALUATIONS."
        ),
    )
    mode_group.add_argument(
        "--sequential",
        action="store_true",
        help="Run selected methods sequentially. Overrides DEFAULT_PARALLEL_ALL_EVALUATIONS.",
    )
    return parser.parse_args()


def _configured_policy_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def policy_path_for_method(method: str) -> str | None:
    if method == "lfpg_rl":
        return _configured_policy_path(LFPG_RL_POLICY)
    if method == "ppo_baseline":
        return _configured_policy_path(RL_POLICY)
    return None


def validate_selected_methods(methods: list[str]) -> None:
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate methods are not allowed: {methods}")
    for method in methods:
        if bool(METHOD_SPECS[method]["supports_rl_overrides"]) and policy_path_for_method(method) is None:
            constant_name = "LFPG_RL_POLICY" if method == "lfpg_rl" else "RL_POLICY"
            raise ValueError(
                f"{method} requires {constant_name} to point to a saved policy checkpoint before testing."
            )


def resolve_test_scenario_ids(scenario_dataset: ScenarioDataset) -> list[str]:
    scenario_ids = [str(scenario_id) for scenario_id in scenario_dataset.get_split_ids("test")]
    if not scenario_ids:
        raise ValueError("test_dataset.npz has no test scenarios.")
    return scenario_ids


def load_scenario_dataset(args: argparse.Namespace) -> ScenarioDataset:
    return ScenarioDataset(default_scenario_dataset_dir(PROJECT_DIR, canonical_network_name(args.network)))


def build_command(
    method: str,
    args: argparse.Namespace,
    test_scenario_ids: list[str],
    *,
    output_suffix: str = TEST_OUTPUT_SUFFIX,
) -> list[str]:
    method_spec = METHOD_SPECS[method]
    command = [
        sys.executable,
        str(method_spec["script"]),
        "--trial",
        "1",
        "--output-suffix",
        output_suffix,
    ]

    command.extend(["--network", canonical_network_name(args.network)])
    if bool(method_spec["supports_rl_overrides"]):
        command.append("--eval-only")
        policy_path = policy_path_for_method(method)
        if policy_path is None:
            raise ValueError(f"{method} requires an explicit policy path before testing.")
        command.extend(["--policy-path", str(policy_path)])
    else:
        # Baseline run.py files add seed + trial internally; base 0 makes the
        # effective run seed exactly 1 for this single full-split test pass.
        command.extend(["--seed", "0"])
    command.extend(["--test-max-scenarios", str(len(test_scenario_ids))])
    command.extend(["--selected-test-scenario-ids", *test_scenario_ids])

    return command


def resolve_num_parallel_env(args: argparse.Namespace) -> int:
    value = NUM_PARALLEL_ENV if args.num_parallel_env is None else int(args.num_parallel_env)
    if value <= 0:
        raise ValueError("NUM_PARALLEL_ENV / --num-parallel-env must be positive.")
    return value


def scenario_batches(scenario_ids: list[str], batch_size: int) -> list[list[str]]:
    return [
        list(scenario_ids[start : start + batch_size])
        for start in range(0, len(scenario_ids), batch_size)
    ]


def normalize_output_suffix(output_suffix: str) -> str:
    suffix = str(output_suffix).strip()
    if not suffix:
        raise ValueError("Output suffix must not be empty.")
    if any(separator in suffix for separator in ("/", "\\")) or suffix in {".", ".."}:
        raise ValueError(f"Invalid output suffix: {output_suffix!r}")
    return suffix if suffix.startswith("_") else f"_{suffix}"


def result_name_for_suffix(network_name: str, output_suffix: str) -> str:
    return f"{get_network_display_name(canonical_network_name(network_name))}{normalize_output_suffix(output_suffix)}"


def method_result_dir(method: str) -> Path:
    return Path(METHOD_SPECS[method]["script"]).parent / "results"


def method_output_dir(method: str, network_name: str, output_suffix: str) -> Path:
    return method_result_dir(method) / result_name_for_suffix(network_name, output_suffix)


def method_test_output_dir(method: str, network_name: str) -> Path:
    return method_output_dir(method, network_name, TEST_OUTPUT_SUFFIX)


def safe_suffix_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return cleaned or "scenario"


def scenario_output_dir(method: str, network_name: str, scenario_id: str) -> Path:
    return method_test_output_dir(method, network_name) / safe_suffix_component(scenario_id)


def parallel_batch_output_suffix(batch_index: int, scenario_id: str) -> str:
    return f"{TEST_OUTPUT_SUFFIX}_batch_{batch_index:03d}_{safe_suffix_component(scenario_id)}"


def remove_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = output_parent.resolve()
    if not output_dir.exists():
        return
    resolved_output = output_dir.resolve()
    if resolved_output.parent != resolved_parent:
        raise RuntimeError(f"Refusing to remove unexpected output path: {resolved_output}")
    if resolved_output.is_dir():
        shutil.rmtree(resolved_output)
    else:
        resolved_output.unlink()


def build_popen_kwargs(network_name: str, launcher_settings: TestLauncherSettings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "cwd": PROJECT_DIR,
        "env": build_child_env(network_name, launcher_settings),
    }
    if WINDOWS_NEW_PROCESS_GROUP:
        kwargs["creationflags"] = WINDOWS_NEW_PROCESS_GROUP
    return kwargs


def close_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=FORCE_KILL_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=FORCE_KILL_WAIT_SECONDS)


def run_sequential(args: argparse.Namespace) -> None:
    validate_selected_methods(list(args.methods))
    scenario_dataset = load_scenario_dataset(args)
    tasks = list(args.methods)
    launcher_settings = resolve_launcher_settings(args, scenario_dataset)
    test_scenario_ids = resolve_test_scenario_ids(scenario_dataset)
    print(
        f"[test] per-step runtime cap={launcher_settings.test_step_runtime_seconds:.3f}s; "
        f"steps per scenario={launcher_settings.num_steps_per_scenario}; "
        f"test_scenarios={len(test_scenario_ids)}; "
        f"methods={tasks}; "
        f"method_count={len(tasks)}; "
        "each method receives the full test split once; "
        "per-step cap is enforced inside each method",
        flush=True,
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        for method in tasks:
            command = build_command(method, args, test_scenario_ids)
            print(f"[test] running {method}: {' '.join(command)}", flush=True)
            process = subprocess.Popen(
                command,
                **build_popen_kwargs(args.network, launcher_settings),
            )
            return_code = process.wait()
            print(f"[test] {method} finished with code {return_code}", flush=True)
            if return_code != 0:
                raise SystemExit(f"Run failed: {method}={return_code}")
            process = None
    except KeyboardInterrupt as exc:
        print("[test] interrupted by user; stopping active child process", flush=True)
        raise SystemExit(130) from exc
    finally:
        if process is not None:
            close_process(process)


def remove_stale_parallel_root_outputs(methods: list[str], network_name: str) -> None:
    root_output_names = (
        TEST_OUTPUTS_NPZ_NAME,
        TEST_SCENARIO_METRICS_CSV_NAME,
        TEST_STEP_HISTORY_CSV_NAME,
        "test_summary.json",
        "config_snapshot.json",
    )
    for method in methods:
        final_dir = method_test_output_dir(method, network_name)
        final_dir.mkdir(parents=True, exist_ok=True)
        for output_name in root_output_names:
            output_path = final_dir / output_name
            if output_path.exists():
                output_path.unlink()


def publish_parallel_scenario_output(task: ParallelScenarioTask, network_name: str) -> Path:
    records = _records_from_output_dir(task.output_dir)
    if len(records) != 1:
        raise RuntimeError(
            f"{task.method}/{task.scenario_id} wrote {len(records)} scenario records; expected exactly one."
        )
    record_scenario_id = str(records[0].scenario_id)
    if record_scenario_id != str(task.scenario_id):
        raise RuntimeError(
            f"{task.method}/{task.scenario_id} wrote scenario {record_scenario_id!r}."
        )

    final_dir = method_test_output_dir(task.method, network_name)
    final_dir.mkdir(parents=True, exist_ok=True)
    destination = scenario_output_dir(task.method, network_name, task.scenario_id)
    publish_tmp_dir = destination.with_name(f".{destination.name}_publish_tmp")
    remove_output_dir(publish_tmp_dir)
    remove_output_dir(destination)
    shutil.move(str(task.output_dir), str(publish_tmp_dir))
    publish_tmp_dir.replace(destination)
    return destination


def run_parallel(args: argparse.Namespace) -> None:
    validate_selected_methods(list(args.methods))
    scenario_dataset = load_scenario_dataset(args)
    tasks = list(args.methods)
    launcher_settings = resolve_launcher_settings(args, scenario_dataset)
    test_scenario_ids = resolve_test_scenario_ids(scenario_dataset)
    num_parallel_env = resolve_num_parallel_env(args)
    batches = scenario_batches(test_scenario_ids, num_parallel_env)
    log_dir = PROJECT_DIR / "logs" / "test"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_output_dirs_by_method: dict[str, list[Path]] = {method: [] for method in tasks}
    remove_stale_parallel_root_outputs(tasks, args.network)
    print(
        f"[test] per-step runtime cap={launcher_settings.test_step_runtime_seconds:.3f}s; "
        f"steps per scenario={launcher_settings.num_steps_per_scenario}; "
        f"test_scenarios={len(test_scenario_ids)}; "
        f"methods={tasks}; "
        f"method_count={len(tasks)}; "
        f"num_parallel_env={num_parallel_env}; "
        f"rounds={len(batches)}; "
        f"max_child_processes_per_round={len(tasks) * num_parallel_env}; "
        "each round launches every selected method for each scenario in that batch; "
        "per-step cap is enforced inside each method",
        flush=True,
    )

    try:
        for batch_index, batch_scenario_ids in enumerate(batches, start=1):
            round_tasks: list[ParallelScenarioTask] = []
            for scenario_id in batch_scenario_ids:
                for method in tasks:
                    output_suffix = parallel_batch_output_suffix(batch_index, scenario_id)
                    output_dir = method_output_dir(method, args.network, output_suffix)
                    remove_output_dir(scenario_output_dir(method, args.network, scenario_id))
                    command = build_command(
                        method,
                        args,
                        [scenario_id],
                        output_suffix=output_suffix,
                    )
                    log_path = log_dir / (
                        f"{method}_{safe_suffix_component(scenario_id)}_"
                        f"batch_{batch_index:03d}_{timestamp}.log"
                    )
                    round_tasks.append(
                        ParallelScenarioTask(
                            method=method,
                            scenario_id=str(scenario_id),
                            output_suffix=output_suffix,
                            output_dir=output_dir,
                            command=command,
                            log_path=log_path,
                        )
                    )

            processes: list[tuple[ParallelScenarioTask, subprocess.Popen[bytes], object]] = []
            print(
                f"[test] round {batch_index}/{len(batches)} launching "
                f"{len(round_tasks)} child processes for scenarios={batch_scenario_ids}",
                flush=True,
            )
            try:
                for task in round_tasks:
                    log_file = task.log_path.open("wb")
                    print(
                        f"[test] launching {task.method} scenario={task.scenario_id}: "
                        f"{' '.join(task.command)}",
                        flush=True,
                    )
                    process = subprocess.Popen(
                        task.command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        **build_popen_kwargs(args.network, launcher_settings),
                    )
                    processes.append((task, process, log_file))

                failures: list[tuple[str, str, int]] = []
                for task, process, _ in processes:
                    return_code = process.wait()
                    print(
                        f"[test] {task.method} scenario={task.scenario_id} finished with code {return_code}",
                        flush=True,
                    )
                    if return_code != 0:
                        failures.append((task.method, task.scenario_id, return_code))
                    else:
                        published_dir = publish_parallel_scenario_output(task, args.network)
                        scenario_output_dirs_by_method[task.method].append(published_dir)
                        print(
                            f"[test] saved {task.method} scenario={task.scenario_id} output={published_dir}",
                            flush=True,
                        )

                if failures:
                    detail = ", ".join(
                        f"{method}/{scenario_id}={code}"
                        for method, scenario_id, code in failures
                    )
                    raise SystemExit(f"One or more runs failed in round {batch_index}: {detail}")
            finally:
                for _, process, log_file in processes:
                    close_process(process)
                    log_file.close()

        for method in tasks:
            merge_parallel_method_outputs(
                method=method,
                network_name=args.network,
                scenario_ids=test_scenario_ids,
                source_dirs=scenario_output_dirs_by_method[method],
                num_parallel_env=num_parallel_env,
                num_rounds=len(batches),
            )
    except KeyboardInterrupt as exc:
        print("[test] interrupted by user; stopping launched child processes", flush=True)
        raise SystemExit(130) from exc


def _optional_seed(value: object) -> int | None:
    seed = int(value)
    return None if seed < 0 else seed


def _records_from_output_dir(output_dir: Path) -> list[TestOutputRecord]:
    payload = load_test_outputs_npz(output_dir / TEST_OUTPUTS_NPZ_NAME)
    scenario_ids = [str(value) for value in payload["scenario_ids"].tolist()]
    od_labels = tuple(str(value) for value in payload["od_labels"].tolist())
    link_labels = tuple(str(value) for value in payload["link_labels"].tolist())
    observation_labels = tuple(str(value) for value in payload["observation_labels"].tolist())
    observed_link_indices = np.asarray(payload["observed_link_indices"], dtype=np.int64)
    has_observation_scale = "observation_scales" in payload

    records: list[TestOutputRecord] = []
    for index, scenario_id in enumerate(scenario_ids):
        records.append(
            TestOutputRecord(
                scenario_id=str(scenario_id),
                split=str(payload["splits"][index]),
                scenario_generation_seed=_optional_seed(payload["scenario_generation_seeds"][index]),
                simulation_seed=_optional_seed(payload["simulation_seeds"][index]),
                od_labels=od_labels,
                link_labels=link_labels,
                estimated_od_matrix=np.asarray(payload["estimated_od_matrices"][index], dtype=np.float32),
                simulated_link_flows=np.asarray(payload["simulated_link_flows"][index], dtype=np.float32),
                observed_link_indices=observed_link_indices,
                observation_labels=observation_labels,
                target_observations=np.asarray(payload["target_observations"][index], dtype=np.float32),
                simulated_observations=np.asarray(payload["simulated_observations"][index], dtype=np.float32),
                observation_scale=(
                    np.asarray(payload["observation_scales"][index], dtype=np.float32)
                    if has_observation_scale
                    else None
                ),
            )
        )
    return records


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _rows_ordered_by_scenarios(
    *,
    source_dirs: list[Path],
    csv_name: str,
    scenario_ids: list[str],
) -> list[dict[str, str]]:
    rows_by_scenario: dict[str, list[dict[str, str]]] = {str(scenario_id): [] for scenario_id in scenario_ids}
    for source_dir in source_dirs:
        for row in _read_csv_rows(source_dir / csv_name):
            scenario_id = str(row.get("scenario_id", "")).strip()
            if scenario_id:
                rows_by_scenario.setdefault(scenario_id, []).append(row)

    ordered_rows: list[dict[str, str]] = []
    for scenario_id in scenario_ids:
        ordered_rows.extend(rows_by_scenario.get(str(scenario_id), []))
    return ordered_rows


def _mean_numeric(rows: list[dict[str, str]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _first_json_payload(source_dirs: list[Path], json_name: str) -> dict[str, object]:
    for source_dir in source_dirs:
        path = source_dir / json_name
        if not path.exists():
            continue
        try:
            return dict(json.loads(path.read_text(encoding="utf-8-sig")))
        except Exception:
            continue
    return {}


def _build_merged_summary(
    *,
    method: str,
    network_name: str,
    final_dir: Path,
    source_dirs: list[Path],
    scenario_ids: list[str],
    scenario_metric_rows: list[dict[str, str]],
    num_parallel_env: int,
    num_rounds: int,
) -> dict[str, object]:
    summary = _first_json_payload(source_dirs, "test_summary.json")
    summary["network_name"] = canonical_network_name(network_name)
    summary["trial_index"] = 1
    summary["num_scenarios"] = int(len(scenario_ids))
    summary["selected_test_scenario_ids"] = [str(scenario_id) for scenario_id in scenario_ids]
    summary.pop("scenario_metrics_csv", None)
    summary.pop("test_outputs_npz", None)
    summary.pop("test_step_history_csv", None)
    summary["scenario_output_dirs"] = {
        str(scenario_id): str((final_dir / safe_suffix_component(str(scenario_id))).resolve())
        for scenario_id in scenario_ids
    }
    summary["parallel_launcher"] = {
        "enabled": True,
        "method": method,
        "num_parallel_env": int(num_parallel_env),
        "num_rounds": int(num_rounds),
        "child_processes_total": int(len(source_dirs)),
        "output_suffix": TEST_OUTPUT_SUFFIX,
    }

    if scenario_metric_rows and "episode_reward" in scenario_metric_rows[0]:
        summary["split"] = "test"
        summary["mean_episode_reward"] = _mean_numeric(scenario_metric_rows, "episode_reward")
        summary["mean_episode_mse"] = _mean_numeric(scenario_metric_rows, "episode_mse")
        summary["mean_episode_mae"] = _mean_numeric(scenario_metric_rows, "episode_mae")
        summary["mean_episode_normalized_mse"] = _mean_numeric(
            scenario_metric_rows,
            "episode_normalized_mse",
        )
    elif scenario_metric_rows:
        summary["mean_mse_mean"] = _mean_numeric(scenario_metric_rows, "mse_mean")
        summary["mean_mae_mean"] = _mean_numeric(scenario_metric_rows, "mae_mean")
        summary["mean_normalized_mse"] = _mean_numeric(scenario_metric_rows, "normalized_mse")
        summary["mean_corr_mean"] = _mean_numeric(scenario_metric_rows, "corr_mean")
        summary["mean_elapsed_seconds"] = _mean_numeric(scenario_metric_rows, "elapsed_seconds")
    return summary


def _build_merged_config_snapshot(
    *,
    network_name: str,
    source_dirs: list[Path],
    scenario_ids: list[str],
    num_parallel_env: int,
    num_rounds: int,
) -> dict[str, object]:
    snapshot = _first_json_payload(source_dirs, "config_snapshot.json")
    snapshot["network_name"] = canonical_network_name(network_name)
    snapshot["trial_index"] = 1
    snapshot["evaluation_trial_seed"] = 1
    snapshot["num_test_scenarios"] = int(len(scenario_ids))
    snapshot["selected_test_scenario_ids"] = [str(scenario_id) for scenario_id in scenario_ids]
    snapshot["parallel_launcher"] = {
        "enabled": True,
        "num_parallel_env": int(num_parallel_env),
        "num_rounds": int(num_rounds),
        "output_suffix": TEST_OUTPUT_SUFFIX,
    }
    return snapshot


def merge_parallel_method_outputs(
    *,
    method: str,
    network_name: str,
    scenario_ids: list[str],
    source_dirs: list[Path],
    num_parallel_env: int,
    num_rounds: int,
) -> None:
    final_dir = method_test_output_dir(method, network_name)
    if len(source_dirs) != len(scenario_ids):
        raise RuntimeError(
            f"{method} produced {len(source_dirs)} scenario outputs for {len(scenario_ids)} scenarios."
        )

    records_by_scenario: dict[str, TestOutputRecord] = {}
    for source_dir in source_dirs:
        for record in _records_from_output_dir(source_dir):
            scenario_id = str(record.scenario_id)
            if scenario_id in records_by_scenario:
                raise RuntimeError(f"Duplicate output for {method} scenario {scenario_id}.")
            records_by_scenario[scenario_id] = record

    missing_scenarios = [
        scenario_id
        for scenario_id in scenario_ids
        if str(scenario_id) not in records_by_scenario
    ]
    if missing_scenarios:
        raise RuntimeError(f"{method} missing merged outputs for scenarios: {missing_scenarios}")

    scenario_metric_rows = _rows_ordered_by_scenarios(
        source_dirs=source_dirs,
        csv_name=TEST_SCENARIO_METRICS_CSV_NAME,
        scenario_ids=scenario_ids,
    )
    if len(scenario_metric_rows) != len(scenario_ids):
        raise RuntimeError(
            f"{method} produced {len(scenario_metric_rows)} scenario metric rows for "
            f"{len(scenario_ids)} scenarios."
        )

    final_dir.mkdir(parents=True, exist_ok=True)
    for stale_output_name in (
        TEST_OUTPUTS_NPZ_NAME,
        TEST_SCENARIO_METRICS_CSV_NAME,
        TEST_STEP_HISTORY_CSV_NAME,
    ):
        stale_output_path = final_dir / stale_output_name
        if stale_output_path.exists():
            stale_output_path.unlink()
    (final_dir / "test_summary.json").write_text(
        json.dumps(
            _build_merged_summary(
                method=method,
                network_name=network_name,
                final_dir=final_dir,
                source_dirs=source_dirs,
                scenario_ids=scenario_ids,
                scenario_metric_rows=scenario_metric_rows,
                num_parallel_env=num_parallel_env,
                num_rounds=num_rounds,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (final_dir / "config_snapshot.json").write_text(
        json.dumps(
            _build_merged_config_snapshot(
                network_name=network_name,
                source_dirs=source_dirs,
                scenario_ids=scenario_ids,
                num_parallel_env=num_parallel_env,
                num_rounds=num_rounds,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[test] finalized {method}: scenarios={len(scenario_ids)} output_root={final_dir}",
        flush=True,
    )


def validate_launcher_defaults() -> None:
    if not TEST_METHODS:
        raise ValueError("TEST_METHODS must not be empty.")
    invalid_methods = [method for method in TEST_METHODS if method not in METHOD_SPECS]
    if invalid_methods:
        raise ValueError(f"TEST_METHODS contains unknown methods: {invalid_methods}")
    if len(set(TEST_METHODS)) != len(TEST_METHODS):
        raise ValueError(f"TEST_METHODS contains duplicate methods: {TEST_METHODS}")
    if not isinstance(DEFAULT_PARALLEL_ALL_EVALUATIONS, bool):
        raise TypeError("DEFAULT_PARALLEL_ALL_EVALUATIONS must be a boolean.")
    if float(DEFAULT_TEST_STEP_RUNTIME_SECONDS) < 0.0:
        raise ValueError("DEFAULT_TEST_STEP_RUNTIME_SECONDS must be non-negative.")
    if not str(TEST_OUTPUT_SUFFIX).strip():
        raise ValueError("TEST_OUTPUT_SUFFIX must not be empty.")
    if int(NUM_PARALLEL_ENV) <= 0:
        raise ValueError("NUM_PARALLEL_ENV must be positive.")


def resolve_scheduling_mode(args: argparse.Namespace) -> str:
    if bool(args.sequential):
        return "sequential"
    if bool(args.parallel):
        return "parallel"
    return "parallel" if bool(DEFAULT_PARALLEL_ALL_EVALUATIONS) else "sequential"


def main() -> None:
    validate_launcher_defaults()
    args = parse_args()
    args.network = canonical_network_name(args.network)
    scheduling_mode = resolve_scheduling_mode(args)
    print(
        f"[test] startup: methods={list(args.methods)} schedule={scheduling_mode} "
        f"network={args.network} num_parallel_env={resolve_num_parallel_env(args)} "
        f"output_suffix={TEST_OUTPUT_SUFFIX}",
        flush=True,
    )
    if scheduling_mode == "sequential":
        run_sequential(args)
    else:
        run_parallel(args)


if __name__ == "__main__":
    main()
