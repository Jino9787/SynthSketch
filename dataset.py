import argparse
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

import main as main_curve


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("npy_outputs")
TOKEN_DECIMAL_PLACES = 3
DEFAULT_FOLDER_SIZE = 10000
DEFAULT_WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )
    parser.add_argument(
        "--folder-size",
        type=int,
        default=DEFAULT_FOLDER_SIZE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def round_token_value(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), TOKEN_DECIMAL_PLACES)
    if isinstance(value, list):
        return [round_token_value(item) for item in value]
    return value


def folder_name_for_index(index, folder_size):
    folder_start = (index // folder_size) * folder_size + 1
    folder_end = folder_start + folder_size - 1
    return f"{folder_start:06d}_{folder_end:06d}"


def output_path_for_sample(index, seed, output_dir, folder_size):
    folder_path = output_dir / folder_name_for_index(index, folder_size)
    return folder_path / f"{seed}.npy"


def generate_one(task):
    index, seed, output_dir, folder_size, overwrite = task
    output_path = output_path_for_sample(index, seed, output_dir, folder_size)
    if output_path.exists() and not overwrite:
        return seed, None, output_path, "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed % (2**32))

    elements = main_curve.generate_mixed_curve()
    token_rows = []
    for loop in main_curve.element_loops(elements):
        token_rows.append(["SOL"])
        for element in loop:
            token_rows.append(round_token_value(main_curve.element_print_token(element)))
        token_rows.append(["EOS"])
    tokens = np.array(token_rows, dtype=object)

    np.save(output_path, tokens)
    return seed, tokens.shape, output_path, "saved"


def validate_sample_count(samples):
    if samples < 1:
        raise ValueError("--samples must be at least 1.")


def iter_tasks(seed, samples, output_dir, folder_size, overwrite):
    for index in range(samples):
        yield index, seed + index, output_dir, folder_size, overwrite


def print_result(seed, shape, output_path, status):
    print(f"seed={seed}")
    if shape is not None:
        print(f"shape={shape}")
    print(f"{status}={output_path.resolve()}")


def run_parallel(tasks, worker_count):
    executor = ProcessPoolExecutor(max_workers=worker_count)
    pending = set()
    task_iter = iter(tasks)
    max_pending = worker_count * 2

    try:
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(generate_one, next(task_iter)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    pending.add(executor.submit(generate_one, next(task_iter)))
                except StopIteration:
                    pass
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()


def main():
    args = parse_args()
    validate_sample_count(args.samples)
    if args.folder_size < 1:
        raise ValueError("--folder-size must be at least 1.")

    worker_count = max(1, min(args.workers, args.samples, os.cpu_count() or 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    saved_count = 0
    skipped_count = 0
    tasks = iter_tasks(
        args.seed,
        args.samples,
        args.output_dir,
        args.folder_size,
        args.overwrite,
    )

    try:
        if worker_count == 1:
            for result in map(generate_one, tasks):
                seed, shape, output_path, status = result
                print_result(seed, shape, output_path, status)
                saved_count += status == "saved"
                skipped_count += status == "skipped"
        else:
            for result in run_parallel(tasks, worker_count):
                seed, shape, output_path, status = result
                print_result(seed, shape, output_path, status)
                saved_count += status == "saved"
                skipped_count += status == "skipped"
    except KeyboardInterrupt:
        print("interrupted=true")

    elapsed = time.perf_counter() - started_at

    print(f"requested_samples={args.samples}")
    print(f"saved_samples={saved_count}")
    print(f"skipped_samples={skipped_count}")
    print(f"workers={worker_count}")
    print(f"folder_size={args.folder_size}")
    print(f"first_seed={args.seed}")
    print(f"last_seed={args.seed + args.samples - 1}")
    print(f"seconds={elapsed:.3f}")
    completed_count = saved_count + skipped_count
    if completed_count:
        print(f"average_seconds_per_completed_sample={elapsed / completed_count:.4f}")


if __name__ == "__main__":
    main()
