from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import Config
from utils import build_protocol_simulation_seed, resolve_trial_scenario_ids
from utils.assignment_guidance import solve_assignment_gradient_step
from utils.baseline_support import (
    aggregate_scenario_runs,
    reset_trial_output_dir,
    run_step_locked_scenario,
    save_sequential_trial_outputs,
    write_rows_csv,
)
from utils.baseline_evaluation import build_result_name, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LFPG-GD (Link flow propagation guided gradient descent) baseline "
            "over the shared online test split."
        )
    )
    parser.add_argument("--trial", type=int, default=1, help="Trial index used for RNG seeding.")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional result-folder suffix. Example: 'test' writes '<network>_test'.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=Config.NETWORK_NAME,
        help="Network name. Only melbourne_scats is supported.",
    )
    parser.add_argument("--seed", type=int, default=3000, help="Base random seed.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Optional override for local row iterations.")
    parser.add_argument("--test-max-scenarios", type=int, default=None, help="Optional cap on evaluated test scenarios.")
    parser.add_argument(
        "--selected-test-scenario-ids",
        nargs="+",
        default=None,
        help="Explicit test scenario IDs to evaluate. Overrides internal trial-seeded scenario selection.",
    )
    return parser.parse_args()


def solve_lfpg_gd_step(
    context,
    params: dict[str, float | int],
) -> tuple[np.ndarray, dict[str, float | int]]:
    local_params = dict(params)
    local_params.setdefault("warm_start", "zero")
    return solve_assignment_gradient_step(
        context,
        local_params,
        feedback_enabled=True,
    )


def run_one_trial(
    trial_index: int,
    seed: int,
    max_iterations: int | None = None,
    test_max_scenarios: int | None = None,
    selected_test_scenario_ids: list[str] | None = None,
    output_suffix: str | None = None,
) -> None:
    if Config.TEST_STEP_RUNTIME_SECONDS is None:
        raise ValueError(
            "Missing test step runtime. Run through test.py or export DODE_TEST_STEP_RUNTIME_SECONDS."
        )
    Config.ensure_dirs()
    set_global_seed(seed)
    evaluation_trial_seed = int(trial_index)
    scenario_dataset = Config.load_scenario_dataset()
    all_test_scenario_ids = list(scenario_dataset.get_split_ids(Config.TEST_SCENARIO_SPLIT))
    selected_test_scenario_ids = list(
        resolve_trial_scenario_ids(
            all_test_scenario_ids,
            trial_seed=evaluation_trial_seed,
            max_scenarios=1 if test_max_scenarios is None else int(test_max_scenarios),
            explicit_scenario_ids=selected_test_scenario_ids,
        )
    )
    test_scenarios = [
        scenario_dataset.load(Config.TEST_SCENARIO_SPLIT, scenario_id)
        for scenario_id in selected_test_scenario_ids
    ]
    if not test_scenarios:
        raise ValueError("Test split is empty. Generate a scenario dataset before running LFPG-GD.")

    trial_result_dir = Config.RESULT_DIR / build_result_name(
        Config.NETWORK_NAME,
        trial_index,
        output_suffix=output_suffix,
    )
    reset_trial_output_dir(trial_result_dir)

    params = dict(Config.LFPG_GD_PARAMS)
    if max_iterations is not None:
        params["max_iterations"] = int(max_iterations)

    started_at = time.perf_counter()
    scenario_runs = []
    step_runtime_seconds = float(Config.TEST_STEP_RUNTIME_SECONDS)
    scenario_runtime_seconds = float(step_runtime_seconds) * float(max(int(scenario_dataset.num_steps), 1))
    print(
        f"[LFPG-GD] selected test scenarios={selected_test_scenario_ids}; "
        f"per-step runtime cap={step_runtime_seconds:.3f}s, "
        f"per-scenario runtime budget={scenario_runtime_seconds:.1f}s "
        f"(steps={int(scenario_dataset.num_steps)}, selected_scenarios={len(test_scenarios)})",
        flush=True,
    )
    for scenario_sample in test_scenarios:
        simulation_seed = build_protocol_simulation_seed(
            trial_seed=evaluation_trial_seed,
            split=Config.TEST_SCENARIO_SPLIT,
            scenario_id=scenario_sample.scenario_id,
        )
        scenario_run = run_step_locked_scenario(
            config=Config,
            scenario_sample=scenario_sample,
            simulation_seed=simulation_seed,
            step_solver=lambda context, local_params=params: solve_lfpg_gd_step(context, local_params),
            step_runtime_seconds=step_runtime_seconds,
        )
        scenario_runs.append(scenario_run)
        print(
            f"[LFPG-GD] scenario={scenario_sample.scenario_id} "
            f"mse={scenario_run.evaluation.mse_mean:,.3f} "
            f"normalized_mse={scenario_run.evaluation.normalized_mse:,.6f}"
        )

    save_sequential_trial_outputs(
        output_dir=trial_result_dir,
        scenario_runs=scenario_runs,
    )

    scenario_rows, summary = aggregate_scenario_runs(
        algorithm_name=Config.ALGORITHM,
        network_name=Config.NETWORK_NAME,
        trial_index=trial_index,
        scenario_runs=scenario_runs,
    )
    summary["elapsed_seconds_total"] = float(time.perf_counter() - started_at)
    summary["elapsed_minutes_total"] = float(summary["elapsed_seconds_total"] / 60.0)
    summary["test_scenario_runtime_seconds"] = float(scenario_runtime_seconds)
    summary["test_step_runtime_seconds"] = float(step_runtime_seconds)

    config_snapshot = {
        "trial_index": int(trial_index),
        "algorithm": Config.ALGORITHM,
        "network_name": Config.NETWORK_NAME,
        "scenario_dataset_dir": str(Config.SCENARIO_DATASET_DIR),
        "train_split": Config.TRAIN_SCENARIO_SPLIT,
        "test_split": Config.TEST_SCENARIO_SPLIT,
        "evaluation_trial_seed": evaluation_trial_seed,
        "num_test_scenarios_total": int(len(all_test_scenario_ids)),
        "num_test_scenarios": int(len(test_scenarios)),
        "selected_test_scenario_ids": selected_test_scenario_ids,
        "num_steps_per_scenario": int(scenario_dataset.num_steps),
        "test_scenario_runtime_seconds": float(scenario_runtime_seconds),
        "test_step_runtime_seconds": float(step_runtime_seconds),
        "action_low": float(Config.ACTION_LOW),
        "action_high": float(Config.ACTION_HIGH),
        "route_choice_mode": Config.ROUTE_CHOICE_MODE,
        "stochastic_logit_scale": float(Config.STOCHASTIC_LOGIT_SCALE),
        "dnl_sample_route_choices": bool(Config.DNL_SAMPLE_ROUTE_CHOICES),
        "dnl_route_choice_sampling_unit": float(Config.DNL_ROUTE_CHOICE_SAMPLING_UNIT),
        "dnl_max_paths_per_od": int(Config.DNL_MAX_PATHS_PER_OD),
        "dnl_due_max_iterations": int(Config.DNL_DUE_MAX_ITERATIONS),
        "dnl_due_tolerance": float(Config.DNL_DUE_TOLERANCE),
        "dnl_clearance_steps": int(Config.DNL_CLEARANCE_STEPS),
        "dnl_parallel_kernels": Config.DNL_PARALLEL_KERNELS,
        "dnl_numba_threads": Config.DNL_NUMBA_THREADS,
        "seed": int(seed),
        "lfpg_gd_params": params,
        "feedback_enabled": True,
        "state_evaluation": "locked_runtime_one_step",
        "uses_explicit_od_prior": False,
    }

    write_rows_csv(trial_result_dir / Config.SCENARIO_METRICS_CSV_NAME, scenario_rows)
    (trial_result_dir / Config.CONFIG_SNAPSHOT_JSON_NAME).write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )
    (trial_result_dir / Config.TEST_SUMMARY_JSON_NAME).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = parse_args()
    Config.configure_network(args.network)
    run_one_trial(
        trial_index=int(args.trial),
        seed=int(args.seed) + int(args.trial),
        max_iterations=args.max_iterations,
        test_max_scenarios=args.test_max_scenarios,
        selected_test_scenario_ids=args.selected_test_scenario_ids,
        output_suffix=args.output_suffix,
    )

