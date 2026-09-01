"""
Run compute_metrics_parallel.py for all (dataset, method, type) combinations
that don't already have their output CSV in EVAL_OUTPUT_DIR. After a successful run,
re-discovers and runs again until nothing is left to do (runs clean).

  --all          Run all methods; ignores --method but still respects --dataset/--types.
  --ds K         Downsampling: pass --ds K so the compute script uses every K-th frame
                 (script reads each video dir and takes every K-th frame from disk).
  --num-files N  Max frames per video; output CSV is results_{method}_{dataset}_{type}_N_dsK.csv.

GPU-parallel mode:
  --gpus 0,1,2,3  -> spawns pinned worker processes (CUDA_VISIBLE_DEVICES) and
                     schedules tasks across them.
  --workers-per-gpu K -> spawns K workers per GPU (total workers = len(gpus)*K)

CPU-parallel mode (default when --gpus is not set):
  --workers N -> ProcessPoolExecutor like before.

Graceful shutdown (sbatch / SLURM):
  SIGUSR1 or SIGTERM sets a shutdown flag. Workers finish their current task
  (writing its result CSV), then exit. The main loop exits after the current
  round drains. On re-submission, discover_tasks skips already-written CSVs.
"""

import argparse
import os
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event, Process, Queue
from pathlib import Path
from typing import List, Optional, Tuple

# Must match compute_metrics_parallel.py
EVAL_BASE        = "/datasets/gencamedit/hdreval/evaluations_transfer"
# EVAL_BASE        = "/projects/gencamedit/hdreval/evaluations"
EVAL_OUTPUT_DIR  = "/projects/gencamedit/hdreval/karan_metrics_eval_output"
DATASETS         = ("stuttgart", "ubc")
METHODS = ( "oursapr21", "eilertsen_pytorch_crf", "ltx", "lediff", "x2hdr", "single_crf", "dvitmo_crf",  "zhdrv_crf", "hdrtv_crf", "ablatedebevec", "ablatel2", "ablaterope", "oursmay5")

SCRIPT_NAME = "compute_metrics_parallel.py"
SKIP_TYPES = {"normal", "over20", "under5"}


#  LTM metrics computing (commented out as this was a special config) 
# EVAL_BASE        = "/datasets/gencamedit/hdreval/evaluations_transfer"
# # EVAL_BASE        = "/projects/gencamedit/hdreval/evaluations"
# EVAL_OUTPUT_DIR  = "/projects/gencamedit/hdreval/evaluations_output_ltm"
# DATASETS         = ("stuttgart", "ubc")
# METHODS = ("single_crf", "dvitmo_crf")

# SCRIPT_NAME = "compute_metrics_parallel.py"
# SKIP_TYPES = {"normal", "over20", "under5", "auto", "under", "over" }

# ---------------------------------------------------------------------------
# Graceful shutdown via signal
# ---------------------------------------------------------------------------
_shutdown_event: Optional[Event] = None


def _handle_shutdown_signal(signum, frame):
    sig_name = {signal.SIGUSR1: "SIGUSR1", signal.SIGTERM: "SIGTERM"}.get(signum, str(signum))
    print(
        f"\n[shutdown] Received {sig_name} — letting in-flight tasks finish, then exiting. "
        f"Re-submit to pick up remaining tasks (already-written CSVs will be skipped).",
        flush=True,
    )
    if _shutdown_event is not None:
        _shutdown_event.set()


def _install_signal_handlers():
    try:
        signal.signal(signal.SIGUSR1, _handle_shutdown_signal)
    except (OSError, AttributeError):
        pass  # SIGUSR1 not available on all platforms
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def results_file_for(method: str, dataset: str, type_name: str, num_files: int, ds: int) -> str:
    """Results CSV basename (e.g. results_ours_stuttgart_under_17_ds1.csv)."""
    return f"results_{method}_{dataset}_{type_name}_{num_files}_ds{ds}.csv"


Task = Tuple[str, str, str]  # (dataset, method, type)


def _metrics_dir() -> Path:
    return Path(__file__).resolve().parent


def _script_path() -> Path:
    return _metrics_dir() / SCRIPT_NAME


def _find_eval_dir(base: Path, method: str, dataset: str) -> Optional[Path]:
    """Return the evaluation directory for (method, dataset), trying both name orderings."""
    for name in (f"{method}_{dataset}", f"{dataset}_{method}"):
        p = base / name
        if p.is_dir():
            return p
    return None


def discover_tasks(num_files: int, ds: int) -> List[Task]:
    """Scan EVAL_BASE for (method, dataset, type) dirs; return those missing the output CSV in EVAL_OUTPUT_DIR."""
    tasks: List[Task] = []
    base = Path(EVAL_BASE)
    out_dir = Path(EVAL_OUTPUT_DIR)
    if not base.is_dir():
        return tasks

    for method in METHODS:
        for dataset in DATASETS:
            gt_dir = base / dataset / "hdr"
            if not gt_dir.is_dir():
                continue
            pred_parent = _find_eval_dir(base, method, dataset)
            if pred_parent is None:
                continue
            for type_dir in pred_parent.iterdir():
                if not type_dir.is_dir():
                    continue
                type_name = type_dir.name
                results_csv = out_dir / results_file_for(method, dataset, type_name, num_files, ds)
                if results_csv.is_file():
                    continue
                tasks.append((dataset, method, type_name))
    method_order = {m: i for i, m in enumerate(METHODS)}
    return sorted(tasks, key=lambda t: (method_order.get(t[1], len(METHODS)), t[0], t[2]))


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def run_one_task(
    task: Task,
    num_files: int,
    ds: int,
    gpu_id: Optional[str] = None,
    workers_per_gpu: int = 1,
    videos: Optional[List[str]] = None,
    videos_exclude: Optional[List[str]] = None,
) -> Tuple[Task, bool]:
    """Run compute_metrics_parallel.py for one (dataset, method, type). Returns (task, success)."""
    dataset, method, type_name = task

    out_csv = Path(EVAL_OUTPUT_DIR) / results_file_for(method, dataset, type_name, num_files, ds)
    if out_csv.is_file():
        return (task, True)  # another process already wrote it

    script = _script_path()
    if not script.is_file():
        print(f"Missing script: {script}", file=sys.stderr)
        return (task, False)

    cmd = [sys.executable, str(script), dataset, method, type_name, "--num-files", str(num_files), "--ds", str(ds)]
    if videos:
        cmd += ["--videos", ",".join(videos)]
    if videos_exclude:
        cmd += ["--videos-exclude", ",".join(videos_exclude)]

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["METRICS_MAX_WORKERS"] = "1"
    env["METRICS_MAX_IN_FLIGHT"] = "1" if workers_per_gpu >= 2 else "2"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_metrics_dir()),
            env=env,
            capture_output=False,
            timeout=None,
        )
        return (task, result.returncode == 0)
    except Exception as e:
        print(f"Error running {cmd} (gpu={gpu_id}): {e}", file=sys.stderr)
        return (task, False)


def _gpu_worker(
    gpu_id: str,
    task_q: Queue,
    result_q: Queue,
    num_files: int,
    ds: int,
    workers_per_gpu: int,
    videos: Optional[List[str]],
    videos_exclude: Optional[List[str]],
    shutdown_event: Event,
) -> None:
    """
    One worker pinned to one GPU. Pulls tasks from task_q, reports results to result_q.
    When shutdown_event is set, skips remaining queued tasks (drains queue until None sentinel).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        task = task_q.get()
        if task is None:
            break

        if shutdown_event.is_set():
            # Skip this task — it has no CSV written, so it will be re-discovered on next run.
            print(f"[shutdown] Skipping (gpu {gpu_id}): {task[0]} / {task[1]} / {task[2]}", flush=True)
            result_q.put((task, False, gpu_id, True))  # last bool = skipped_due_to_shutdown
            continue

        t, ok = run_one_task(
            task,
            num_files,
            ds,
            gpu_id=gpu_id,
            workers_per_gpu=workers_per_gpu,
            videos=videos,
            videos_exclude=videos_exclude,
        )
        result_q.put((t, ok, gpu_id, False))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _is_santos_ubc_over(task: Task) -> bool:
    """True if task is santos on ubc with type starting with 'over'."""
    dataset, method, type_name = task
    return dataset == "ubc" and method == "santos" and type_name.startswith("over")


def _apply_filters(tasks: List[Task], args: argparse.Namespace) -> List[Task]:
    """Apply --method, --dataset, --types to task list. --all overrides only --method."""
    out = tasks
    if not getattr(args, "all", False) and args.method is not None:
        out = [t for t in out if t[1] == args.method]
    if args.dataset is not None:
        out = [t for t in out if t[0] == args.dataset]
    if args.types is not None:
        allowed = {s.strip() for s in args.types.split(",") if s.strip()}
        out = [t for t in out if t[2] in allowed]
    if not getattr(args, "santosover", False):
        out = [t for t in out if not _is_santos_ubc_over(t)]
    out = [t for t in out if t[2] not in SKIP_TYPES]
    return out


def _parse_gpus(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    gpus = [x.strip() for x in s.split(",") if x.strip() != ""]
    return gpus if gpus else None


def _parse_videos(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    vids = [x.strip() for x in s.split(",") if x.strip() != ""]
    return vids if vids else None


# ---------------------------------------------------------------------------
# Round runner
# ---------------------------------------------------------------------------

def _run_round(
    tasks: List[Task],
    args: argparse.Namespace,
    num_files: int,
    ds: int,
    videos: Optional[List[str]],
    videos_exclude: Optional[List[str]],
) -> Tuple[List[Task], bool]:
    """
    Run one round of tasks (GPU or CPU).
    Returns (failed_tasks, shutdown_was_triggered).
    failed_tasks excludes tasks skipped due to shutdown (they'll be re-discovered next run).
    """
    failed: List[Task] = []
    shutdown_triggered = False
    gpus = _parse_gpus(args.gpus)

    if gpus is not None:
        task_q: Queue = Queue()
        result_q: Queue = Queue()
        workers_per_gpu = max(1, min(4, int(args.workers_per_gpu)))
        total_workers = len(gpus) * workers_per_gpu
        print(f"GPU mode: {len(gpus)} GPU(s) {gpus}, {workers_per_gpu} process(es) per GPU → {total_workers} total")

        workers: List[Process] = []
        for gpu_id in gpus:
            for _ in range(workers_per_gpu):
                p = Process(
                    target=_gpu_worker,
                    args=(
                        gpu_id,
                        task_q,
                        result_q,
                        num_files,
                        ds,
                        workers_per_gpu,
                        videos,
                        videos_exclude,
                        _shutdown_event,
                    ),
                    daemon=True,
                )
                p.start()
                workers.append(p)

        for t in tasks:
            task_q.put(t)
        for _ in range(total_workers):
            task_q.put(None)

        remaining = len(tasks)
        while remaining > 0:
            t, ok, gpu_id, skipped = result_q.get()
            dataset, method, type_name = t
            if skipped:
                shutdown_triggered = True
            elif ok:
                print(f"Done (gpu {gpu_id}): {dataset} / {method} / {type_name}")
            else:
                failed.append(t)
                print(f"Failed (gpu {gpu_id}): {dataset} / {method} / {type_name}", file=sys.stderr)
            remaining -= 1

        for p in workers:
            p.join()
        return failed, shutdown_triggered

    # CPU mode
    num_workers = max(1, min(args.workers, len(tasks)))
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(run_one_task, t, num_files, ds, None, 1, videos, videos_exclude): t
            for t in tasks
        }
        for fut in as_completed(futures):
            (dataset, method, type_name), ok = fut.result()
            if ok:
                print(f"Done: {dataset} / {method} / {type_name}")
            else:
                failed.append((dataset, method, type_name))
            if _shutdown_event is not None and _shutdown_event.is_set():
                shutdown_triggered = True
                ex.shutdown(wait=False, cancel_futures=True)
                break
    return failed, shutdown_triggered


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run metrics for all (dataset, method, type) missing results.csv. "
        "Use --method/--dataset/--types to limit work across servers."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=list(METHODS),
        default=None,
        help="Only run this method (e.g. lediff on one server, ours on another).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        default=None,
        help="Only run this dataset.",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        metavar="T1,T2,...",
        help="Only run these type subfolders (comma-separated).",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all (method, dataset, type) missing results.csv and exit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU mode: parallel subprocess workers (default 1). Ignored when --gpus is set.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        metavar="0,1,2,...",
        help="GPU mode: comma-separated GPU ids.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="GPU mode: number of concurrent workers per GPU (default 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print tasks that would be run, then exit.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=17,
        metavar="N",
        help="Max frames per video (default 17).",
    )
    parser.add_argument(
        "--ds", "--downsampling",
        type=int,
        default=1,
        dest="ds",
        metavar="K",
        help="Downsample: use every K-th frame (default 1).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all methods. Ignores --method but still respects --dataset/--types.",
    )
    parser.add_argument(
        "--santosover",
        action="store_true",
        help="Include santos ubc over (e.g. over20) in metrics. Excluded by default.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help="Optional comma-separated video folder names to pass to compute script.",
    )
    parser.add_argument(
        "--videos-exclude",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help="Optional comma-separated video folder names to exclude in compute script.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _shutdown_event
    _shutdown_event = Event()
    _install_signal_handlers()

    args = parse_args()
    num_files = args.num_files
    ds = max(1, args.ds)
    videos = _parse_videos(args.videos)
    videos_exclude = _parse_videos(args.videos_exclude)

    if args.types is not None and not getattr(args, "all", False):
        allowed = {s.strip() for s in args.types.split(",") if s.strip()}
        if not allowed:
            print("No valid types in --types", file=sys.stderr)
            return 1

    out_pattern = f"results_*_*_*_{num_files}_ds{ds}.csv"

    all_tasks = discover_tasks(num_files, ds)
    tasks = _apply_filters(all_tasks, args)

    if args.list_types:
        seen = set()
        for t in tasks:
            key = (t[1], t[0], t[2])
            if key not in seen:
                seen.add(key)
                print(f"  {t[0]} / {t[1]} / {t[2]}")
        print(
            f"\nTotal: {len(tasks)} task(s) missing output CSV ({out_pattern} in {EVAL_OUTPUT_DIR}). "
            f"Use --types type1,type2,... to limit types, or --all for all methods."
        )
        return 0

    round_num = 0
    while True:
        if _shutdown_event.is_set():
            print("[shutdown] Exiting before starting new round.", flush=True)
            return 0

        round_num += 1
        all_tasks = discover_tasks(num_files, ds)
        tasks = _apply_filters(all_tasks, args)

        if not tasks:
            if round_num == 1:
                print("Nothing to do.")
            else:
                print("All done; nothing left to run.")
            return 0

        if round_num > 1:
            print(f"Round {round_num}: {len(tasks)} task(s) still missing output CSV.")
        else:
            print(
                f"Discovered {len(tasks)} tasks missing output CSV; "
                f"running {len(tasks)} after filters."
            )

        for t in tasks:
            print(f"  {t[0]} / {t[1]} / {t[2]}")

        if args.dry_run:
            return 0

        failed, shutdown_triggered = _run_round(tasks, args, num_files, ds, videos, videos_exclude)
        if failed:
            print("Some tasks failed (will retry next round):", failed, file=sys.stderr)
        if shutdown_triggered:
            print("[shutdown] Exiting after round — re-submit to continue.", flush=True)
            return 0


if __name__ == "__main__":
    sys.exit(main())