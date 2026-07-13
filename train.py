from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from dnl.network.registry import canonical_network_name
from utils import (
    DEFAULT_NETWORK_ENV_VAR,
    MAX_RUNTIME_SECONDS_ENV_VAR,
    NUM_ENVS_ENV_VAR,
    USE_SUBPROC_ENV_VAR,
    get_section_settings,
)


PROJECT_DIR = Path(__file__).resolve().parent
FORCE_KILL_WAIT_SECONDS = 10

# This project is fixed to the final refined Melbourne SCATS network.
DEFAULT_NETWORK_NAME = "melbourne_scats"


def _default_bool(value: object, fallback: bool) -> bool:
    if value is None:
        return bool(fallback)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Expected boolean launcher setting, received {value!r}.")
    return bool(value)


def _load_default_rl_launcher_settings() -> tuple[int, bool]:
    rl_settings = get_section_settings(
        "rl",
        DEFAULT_NETWORK_NAME,
        apply_runtime_env_overrides=True,
    )
    return (
        int(rl_settings.get("num_envs", 1)),
        _default_bool(rl_settings.get("use_subproc"), False),
    )

# Select only the RL methods to train here.
# Available options:
#   "lfpg_rl"
#   "ppo_baseline"

DEFAULT_METHODS = ["lfpg_rl", "ppo_baseline"]
DEFAULT_TRIALS = [201, 202, 203, 204, 205]
DEFAULT_MAX_RUNTIME_SECONDS = 3600 * 20
DEFAULT_NUM_ENVS, DEFAULT_USE_SUBPROC = _load_default_rl_launcher_settings()
# False: run both methods for trial 201, then 202, ...
# True: launch every method/trial task at once.
DEFAULT_PARALLEL_ALL_TRIALS = True


METHOD_SPECS = {
    "lfpg_rl": {
        "script": PROJECT_DIR / "methods" / "1-1. LFPG-RL" / "run.py",
    },
    "ppo_baseline": {
        "script": PROJECT_DIR / "methods" / "1-2. PPO" / "run.py",
    },
}


@dataclass(frozen=True)
class RunTask:
    method: str
    trial: int


@dataclass(frozen=True)
class LauncherSettings:
    max_runtime_seconds: int
    child_runtime_seconds: int
    method_timeout_seconds: int
    method_timeout_grace_seconds: int


def _require_launcher_default(name: str) -> object:
    if name not in globals():
        raise RuntimeError(
            f"Missing required train.py launcher parameter: {name}. "
            "Define it near the top of train.py before running training."
        )
    return globals()[name]


def validate_launcher_defaults() -> None:
    max_runtime_seconds = int(_require_launcher_default("DEFAULT_MAX_RUNTIME_SECONDS"))
    num_envs = int(_require_launcher_default("DEFAULT_NUM_ENVS"))
    use_subproc = _require_launcher_default("DEFAULT_USE_SUBPROC")
    parallel_all_trials = _require_launcher_default("DEFAULT_PARALLEL_ALL_TRIALS")

    if max_runtime_seconds <= 0:
        raise ValueError("DEFAULT_MAX_RUNTIME_SECONDS must be positive.")
    if num_envs <= 0:
        raise ValueError("DEFAULT_NUM_ENVS must be positive.")
    if not isinstance(use_subproc, bool):
        raise TypeError("DEFAULT_USE_SUBPROC must be a boolean.")
    if not isinstance(parallel_all_trials, bool):
        raise TypeError("DEFAULT_PARALLEL_ALL_TRIALS must be a boolean.")


def resolve_launcher_settings(
    network_name: str,
    max_runtime_seconds_override: int | None = None,
) -> LauncherSettings:
    _ = canonical_network_name(network_name)
    max_runtime_seconds = (
        int(max_runtime_seconds_override)
        if max_runtime_seconds_override is not None
        else int(DEFAULT_MAX_RUNTIME_SECONDS)
    )
    shutdown_window_seconds = max(120, min(900, max_runtime_seconds // 2))
    return LauncherSettings(
        max_runtime_seconds=max_runtime_seconds,
        child_runtime_seconds=max_runtime_seconds,
        method_timeout_seconds=max_runtime_seconds + shutdown_window_seconds,
        method_timeout_grace_seconds=shutdown_window_seconds,
    )


def build_child_env(
    network_name: str,
    launcher_settings: LauncherSettings,
    *,
    num_envs: int,
    use_subproc: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env[DEFAULT_NETWORK_ENV_VAR] = canonical_network_name(network_name)
    env[MAX_RUNTIME_SECONDS_ENV_VAR] = str(launcher_settings.max_runtime_seconds)
    env[NUM_ENVS_ENV_VAR] = str(int(num_envs))
    env[USE_SUBPROC_ENV_VAR] = "true" if bool(use_subproc) else "false"
    return env


def parse_args() -> argparse.Namespace:
    default_methods_text = " ".join(DEFAULT_METHODS)
    parser = argparse.ArgumentParser(
        description=(
            "Train and save policy checkpoints for the selected RL methods. "
            "The default scheduling mode is controlled by DEFAULT_PARALLEL_ALL_TRIALS."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_SPECS.keys()),
        default=list(DEFAULT_METHODS),
        help=(
            "Which RL methods to train. "
            f"Default: {default_methods_text}."
        ),
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRIALS),
        help=f"Trial indices to train. Default: {' '.join(str(trial) for trial in DEFAULT_TRIALS)}.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=DEFAULT_NETWORK_NAME,
        help="Network passed through to every run.py. Only melbourne_scats is supported.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help=f"LFPG-RL vectorized environments per method. Default: {DEFAULT_NUM_ENVS}.",
    )
    parser.add_argument(
        "--use-subproc",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_SUBPROC,
        help=f"Use subprocess vector environments when num-envs > 1. Default: {DEFAULT_USE_SUBPROC}.",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=int,
        default=None,
        help="Optional per-method runtime cap override.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional per-method training stop after this many completed episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional launcher RNG seed base. When omitted, each child receives its trial "
            "index as the seed. When set, each child receives base + trial."
        ),
    )
    parser.add_argument(
        "--evaluate-after-training",
        action="store_true",
        help="Run each method's held-out evaluation immediately after training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each training run from latest_model.pt when available, otherwise final_model.pt.",
    )
    parser.add_argument(
        "--resume-policy-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path to resume from. Intended for single-task resume runs.",
    )
    parser.add_argument("--ppo-learning-rate", type=float, default=None)
    parser.add_argument("--ppo-ent-coef", type=float, default=None)
    parser.add_argument("--ppo-clip-range", type=float, default=None)
    parser.add_argument("--ppo-vf-coef", type=float, default=None)
    parser.add_argument("--ppo-target-kl", type=float, default=None)
    parser.add_argument("--ppo-n-epochs", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=None)
    parser.add_argument("--ppo-gamma", type=float, default=None)
    parser.add_argument("--ppo-gae-lambda", type=float, default=None)
    parser.add_argument("--ppo-initial-policy-mean", type=float, default=None)
    parser.add_argument("--ppo-initial-policy-std", type=float, default=None)
    parser.add_argument("--lfp-a-advantage-weight", type=float, default=None)
    parser.add_argument(
        "--lfp-a-decay-schedule",
        type=str,
        choices=("constant", "linear", "exp"),
        default=None,
    )
    parser.add_argument("--lfp-a-decay-rollouts", type=int, default=None)
    parser.add_argument("--lfp-a-advantage-clip", type=float, default=None)
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately when one trial batch fails.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--grouped",
        action="store_true",
        help=(
            "Run selected methods in parallel within each trial, then process the next trial. "
            "Overrides DEFAULT_PARALLEL_ALL_TRIALS."
        ),
    )
    mode_group.add_argument(
        "--parallel",
        "--all-parallel",
        dest="parallel",
        action="store_true",
        help=(
            "Launch every method/trial task concurrently. "
            "For 2 methods x 5 trials, this starts 10 child Python processes at once."
        ),
    )
    mode_group.add_argument(
        "--sequential",
        action="store_true",
        help="Run selected method/trial tasks sequentially.",
    )
    return parser.parse_args()


def normalize_trials(trials: list[int]) -> list[int]:
    ordered_unique: list[int] = []
    for trial in trials:
        trial_index = int(trial)
        if trial_index not in ordered_unique:
            ordered_unique.append(trial_index)
    if not ordered_unique:
        raise ValueError("At least one trial index must be provided.")
    return ordered_unique


def build_tasks(args: argparse.Namespace) -> list[RunTask]:
    trials = normalize_trials(list(args.trials))
    return [RunTask(method=method, trial=trial) for method in args.methods for trial in trials]


def resolve_child_seed(task: RunTask, args: argparse.Namespace) -> int:
    seed_base = getattr(args, "seed", None)
    if seed_base is None:
        return int(task.trial) % (2**32 - 1)
    return int((int(seed_base) + int(task.trial)) % (2**32 - 1))


def build_command(task: RunTask, args: argparse.Namespace) -> list[str]:
    method_spec = METHOD_SPECS[task.method]
    command = [
        sys.executable,
        str(method_spec["script"]),
        "--trial",
        str(task.trial),
        "--network",
        canonical_network_name(args.network),
    ]
    if not bool(getattr(args, "evaluate_after_training", False)):
        command.append("--train-only")
    if args.num_envs is not None:
        command.extend(["--num-envs", str(args.num_envs)])
    command.append("--use-subproc" if bool(args.use_subproc) else "--no-use-subproc")
    if args.runtime_seconds is not None:
        command.extend(["--runtime-seconds", str(args.runtime_seconds)])
    if getattr(args, "max_episodes", None) is not None:
        command.extend(["--max-episodes", str(int(args.max_episodes))])
    command.extend(["--seed", str(resolve_child_seed(task, args))])
    if bool(getattr(args, "resume", False)):
        command.append("--resume")
    if getattr(args, "resume_policy_path", None) is not None:
        command.extend(["--resume-policy-path", str(args.resume_policy_path)])
    lfpg_override_args = (
        "ppo_learning_rate",
        "ppo_ent_coef",
        "ppo_clip_range",
        "ppo_vf_coef",
        "ppo_target_kl",
        "ppo_n_epochs",
        "ppo_batch_size",
        "ppo_max_grad_norm",
        "ppo_gamma",
        "ppo_gae_lambda",
        "ppo_initial_policy_mean",
        "ppo_initial_policy_std",
        "lfp_a_advantage_weight",
        "lfp_a_decay_schedule",
        "lfp_a_decay_rollouts",
        "lfp_a_advantage_clip",
    )
    for arg_name in lfpg_override_args:
        value = getattr(args, arg_name, None)
        if value is not None:
            command.extend([f"--{arg_name.replace('_', '-')}", str(value)])
    return command


def task_label(task: RunTask) -> str:
    return f"{task.method}[trial={task.trial}]"


def wait_with_grace(
    task: RunTask,
    process: subprocess.Popen[bytes],
    prefix: str,
    launcher_settings: LauncherSettings,
) -> int:
    label = task_label(task)
    started_at = time.monotonic()
    try:
        return process.wait(timeout=launcher_settings.method_timeout_seconds)
    except subprocess.TimeoutExpired:
        elapsed_seconds = time.monotonic() - started_at
        print(
            f"[{prefix}] {label} exceeded {launcher_settings.method_timeout_seconds} seconds "
            f"(elapsed={elapsed_seconds:.1f}s); allowing up to {launcher_settings.method_timeout_grace_seconds} "
            "more seconds for clean shutdown",
            flush=True,
        )
        try:
            return process.wait(timeout=launcher_settings.method_timeout_grace_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"[{prefix}] {label} did not exit during the grace window and will be terminated",
                flush=True,
            )
            process.terminate()
            try:
                return process.wait(timeout=FORCE_KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait(timeout=FORCE_KILL_WAIT_SECONDS)


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
    tasks = build_tasks(args)
    launcher_settings = resolve_launcher_settings(args.network, args.runtime_seconds)
    print(
        f"[train] child training runtime budget={launcher_settings.child_runtime_seconds}s; "
        f"parent timeout={launcher_settings.method_timeout_seconds}s "
        f"with an additional termination grace window of {launcher_settings.method_timeout_grace_seconds}s",
        flush=True,
    )
    for task in tasks:
        command = build_command(task, args)
        print(f"[train] running {task_label(task)}: {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=build_child_env(
                args.network,
                launcher_settings,
                num_envs=int(args.num_envs),
                use_subproc=bool(args.use_subproc),
            ),
        )
        return_code = wait_with_grace(task, process, prefix="train", launcher_settings=launcher_settings)
        print(f"[train] {task_label(task)} finished with code {return_code}", flush=True)
        if return_code != 0:
            raise SystemExit(f"Training run failed: {task_label(task)}={return_code}")


def run_parallel(args: argparse.Namespace) -> None:
    tasks = build_tasks(args)
    launcher_settings = resolve_launcher_settings(args.network, args.runtime_seconds)
    processes: list[tuple[RunTask, subprocess.Popen[bytes]]] = []
    trials = normalize_trials(list(args.trials))
    print(
        "[train] scheduling mode: all method/trial tasks in parallel",
        flush=True,
    )
    print(
        f"[train] methods={args.methods} trials={trials} total_tasks={len(tasks)} network={args.network}",
        flush=True,
    )
    print(
        f"[train] child training runtime budget={launcher_settings.child_runtime_seconds}s; "
        f"parent timeout={launcher_settings.method_timeout_seconds}s "
        f"with an additional termination grace window of {launcher_settings.method_timeout_grace_seconds}s",
        flush=True,
    )

    try:
        for task in tasks:
            command = build_command(task, args)
            print(f"[train] launching {task_label(task)}: {' '.join(command)}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=build_child_env(
                    args.network,
                    launcher_settings,
                    num_envs=int(args.num_envs),
                    use_subproc=bool(args.use_subproc),
                ),
            )
            processes.append((task, process))

        failures: list[tuple[str, int]] = []
        for task, process in processes:
            return_code = wait_with_grace(task, process, prefix="train", launcher_settings=launcher_settings)
            print(f"[train] {task_label(task)} finished with code {return_code}", flush=True)
            if return_code != 0:
                failures.append((task_label(task), return_code))

        if failures:
            detail = ", ".join(f"{label}={code}" for label, code in failures)
            raise SystemExit(f"One or more training runs failed: {detail}")
    finally:
        for _, process in processes:
            close_process(process)


def build_batch_args(args: argparse.Namespace, trial: int) -> SimpleNamespace:
    return SimpleNamespace(
        methods=list(args.methods),
        trials=[int(trial)],
        network=canonical_network_name(args.network),
        num_envs=args.num_envs,
        use_subproc=args.use_subproc,
        runtime_seconds=args.runtime_seconds,
        max_episodes=args.max_episodes,
        seed=args.seed,
        evaluate_after_training=args.evaluate_after_training,
        resume=args.resume,
        resume_policy_path=args.resume_policy_path,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_clip_range=args.ppo_clip_range,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_target_kl=args.ppo_target_kl,
        ppo_n_epochs=args.ppo_n_epochs,
        ppo_batch_size=args.ppo_batch_size,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_initial_policy_mean=args.ppo_initial_policy_mean,
        ppo_initial_policy_std=args.ppo_initial_policy_std,
        lfp_a_advantage_weight=args.lfp_a_advantage_weight,
        lfp_a_decay_schedule=args.lfp_a_decay_schedule,
        lfp_a_decay_rollouts=args.lfp_a_decay_rollouts,
        lfp_a_advantage_clip=args.lfp_a_advantage_clip,
        sequential=False,
        parallel=True,
        grouped=False,
        stop_on_failure=args.stop_on_failure,
    )


def run_grouped_by_trial(args: argparse.Namespace) -> None:
    trials = normalize_trials(list(args.trials))
    failures: list[tuple[int, str]] = []

    print(
        "[train] scheduling mode: parallel within each trial, sequential across trials",
        flush=True,
    )
    print(
        f"[train] methods={args.methods} trials={trials} network={args.network}",
        flush=True,
    )

    for batch_index, trial in enumerate(trials, start=1):
        print(
            f"[train] starting batch {batch_index}/{len(trials)} for trial={trial}",
            flush=True,
        )
        batch_args = build_batch_args(args, trial)
        try:
            run_parallel(batch_args)
        except SystemExit as exc:
            message = str(exc) if str(exc) else f"trial={trial} failed"
            failures.append((int(trial), message))
            print(f"[train] batch trial={trial} failed: {message}", flush=True)
            if args.stop_on_failure:
                raise
        print(
            f"[train] finished batch {batch_index}/{len(trials)} for trial={trial}",
            flush=True,
        )

    if failures:
        failure_text = "; ".join(f"trial={trial}: {message}" for trial, message in failures)
        raise SystemExit(f"One or more trial batches failed: {failure_text}")

    print("[train] all trial batches completed", flush=True)


def resolve_scheduling_mode(args: argparse.Namespace) -> str:
    if bool(args.sequential):
        return "sequential"
    if bool(args.parallel):
        return "parallel"
    if bool(args.grouped):
        return "grouped"
    return "parallel" if bool(DEFAULT_PARALLEL_ALL_TRIALS) else "grouped"


def main() -> None:
    validate_launcher_defaults()
    args = parse_args()
    args.network = canonical_network_name(args.network)
    scheduling_mode = resolve_scheduling_mode(args)
    child_seed_text = "trial" if args.seed is None else f"{args.seed}+trial"
    print(
        f"[train] startup: methods={args.methods} trials={normalize_trials(list(args.trials))} "
        f"schedule={scheduling_mode} network={args.network} "
        f"runtime_seconds={args.runtime_seconds if args.runtime_seconds is not None else DEFAULT_MAX_RUNTIME_SECONDS} "
        f"num_envs={args.num_envs} use_subproc={args.use_subproc} resume={args.resume} "
        f"child_seed={child_seed_text}",
        flush=True,
    )
    if len(sys.argv) == 1:
        print(
            "[train] no command-line arguments supplied; Ctrl+F5 is running the launcher defaults. "
            "Child training can be quiet during DNL rollouts, but progress messages will appear after updates.",
            flush=True,
        )
    if scheduling_mode == "sequential":
        run_sequential(args)
    elif scheduling_mode == "parallel":
        run_parallel(args)
    else:
        run_grouped_by_trial(args)


if __name__ == "__main__":
    main()
