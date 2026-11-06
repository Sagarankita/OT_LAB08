"""Submodular coverage and real process-based parallel experiments."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Task:
    task_id: int
    work: int
    block: int
    seed: int


def coverage_value(selected: Iterable[int], sets: Sequence[frozenset[int]]) -> int:
    covered: set[int] = set()
    for index in selected:
        covered.update(sets[index])
    return len(covered)


def greedy_max_coverage(sets: Sequence[frozenset[int]], budget: int) -> tuple[tuple[int, ...], int]:
    if budget < 0 or budget > len(sets):
        raise ValueError("budget must be between zero and the number of sets")
    selected: list[int] = []
    covered: set[int] = set()
    remaining = set(range(len(sets)))
    for _ in range(budget):
        choice = max(remaining, key=lambda i: (len(sets[i] - covered), -i))
        selected.append(choice)
        covered.update(sets[choice])
        remaining.remove(choice)
    return tuple(selected), len(covered)


def exhaustive_max_coverage(sets: Sequence[frozenset[int]], budget: int) -> tuple[tuple[int, ...], int]:
    if budget < 0 or budget > len(sets):
        raise ValueError("budget must be between zero and the number of sets")
    best: tuple[int, ...] = ()
    best_value = -1
    for selected in combinations(range(len(sets)), budget):
        value = coverage_value(selected, sets)
        if value > best_value:
            best, best_value = selected, value
    return best, max(best_value, 0)


def verify_submodular_axioms(sets: Sequence[frozenset[int]]) -> dict[str, bool | int]:
    """Exhaustively check normalized, nonnegative, monotone, diminishing returns."""
    n = len(sets)
    subsets = [frozenset(i for i in range(n) if mask & (1 << i)) for mask in range(1 << n)]
    values = {subset: coverage_value(subset, sets) for subset in subsets}
    normalized = values[frozenset()] == 0
    nonnegative = all(value >= 0 for value in values.values())
    monotone = all(values[a] <= values[b] for a in subsets for b in subsets if a <= b)
    diminishing = True
    comparisons = 0
    for a in subsets:
        for b in subsets:
            if not a <= b:
                continue
            for element in range(n):
                if element in b:
                    continue
                comparisons += 1
                gain_a = values[a | {element}] - values[a]
                gain_b = values[b | {element}] - values[b]
                if gain_a < gain_b:
                    diminishing = False
    return {
        "normalized": normalized,
        "nonnegative": nonnegative,
        "monotone": monotone,
        "diminishing_returns": diminishing,
        "subsets_checked": len(subsets),
        "diminishing_comparisons": comparisons,
    }


def compute_task(task: Task) -> tuple[int, int]:
    """Deterministic irregular CPU work with an auditable integer result."""
    value = (task.seed ^ (task.task_id * 2654435761)) & 0xFFFFFFFF
    checksum = 0
    for index in range(task.work):
        value = (1664525 * value + 1013904223 + index) & 0xFFFFFFFF
        checksum = (checksum + (value ^ (value >> 16))) & 0xFFFFFFFFFFFFFFFF
    return task.task_id, checksum


def _worker_static(worker: int, tasks: list[Task], workers: int, output: mp.Queue) -> None:
    started = time.perf_counter()
    values = [compute_task(task) for task in tasks]
    output.put(
        {
            "worker": worker,
            "values": values,
            "tasks": len(tasks),
            "work": sum(task.work for task in tasks),
            "remote_tasks": sum(task.block % workers != worker for task in tasks),
            "steals": 0,
            "compute_seconds": time.perf_counter() - started,
        }
    )


def _worker_dynamic(worker: int, tasks: mp.Queue, workers: int, output: mp.Queue) -> None:
    started = time.perf_counter()
    values = []
    work = 0
    remote = 0
    while True:
        task = tasks.get()
        if task is None:
            break
        values.append(compute_task(task))
        work += task.work
        remote += task.block % workers != worker
    output.put(
        {
            "worker": worker,
            "values": values,
            "tasks": len(values),
            "work": work,
            "remote_tasks": remote,
            "steals": 0,
            "compute_seconds": time.perf_counter() - started,
        }
    )


def _worker_steal(
    worker: int,
    task_indices: list,
    left_indices: list,
    right_indices: list,
    locks: list,
    tasks: list[Task],
    workers: int,
    output: mp.Queue,
) -> None:
    started = time.perf_counter()
    values = []
    work = 0
    remote = 0
    steals = 0
    while True:
        task = None
        with locks[worker]:
            if left_indices[worker].value < right_indices[worker].value:
                right_indices[worker].value -= 1
                task = tasks[task_indices[worker][right_indices[worker].value]]
        if task is None:
            for offset in range(1, workers):
                victim = (worker + offset) % workers
                with locks[victim]:
                    if left_indices[victim].value < right_indices[victim].value:
                        index = left_indices[victim].value
                        left_indices[victim].value += 1
                        task = tasks[task_indices[victim][index]]
                        steals += 1
                        break
        if task is None:
            break
        values.append(compute_task(task))
        work += task.work
        remote += task.block % workers != worker
    output.put(
        {
            "worker": worker,
            "values": values,
            "tasks": len(values),
            "work": work,
            "remote_tasks": remote,
            "steals": steals,
            "compute_seconds": time.perf_counter() - started,
        }
    )


def _collect(processes: list[mp.Process], output: mp.Queue) -> list[dict]:
    records = [output.get(timeout=180) for _ in processes]
    for process in processes:
        process.join(timeout=180)
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited with code {process.exitcode}")
    return sorted(records, key=lambda record: record["worker"])


def run_tasks(tasks: Sequence[Task], method: str, workers: int) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if method == "serial":
        started = time.perf_counter()
        values = [compute_task(task) for task in tasks]
        elapsed = time.perf_counter() - started
        worker_records = [{
            "worker": 0,
            "values": values,
            "tasks": len(tasks),
            "work": sum(task.work for task in tasks),
            "remote_tasks": 0,
            "steals": 0,
            "compute_seconds": elapsed,
        }]
    else:
        context = mp.get_context("spawn")
        output = context.Queue()
        processes: list[mp.Process] = []
        started = time.perf_counter()
        if method == "static":
            buckets: list[list[Task]] = [[] for _ in range(workers)]
            loads = [0] * workers
            for task in sorted(tasks, key=lambda item: (-item.work, item.task_id)):
                worker = min(range(workers), key=lambda i: (loads[i], i))
                buckets[worker].append(task)
                loads[worker] += task.work
            processes = [
                context.Process(target=_worker_static, args=(worker, buckets[worker], workers, output))
                for worker in range(workers)
            ]
        elif method == "dynamic":
            task_queue = context.Queue()
            for task in tasks:
                task_queue.put(task)
            for _ in range(workers):
                task_queue.put(None)
            processes = [
                context.Process(target=_worker_dynamic, args=(worker, task_queue, workers, output))
                for worker in range(workers)
            ]
        elif method == "work_stealing":
            initial = [[] for _ in range(workers)]
            task_list = list(tasks)
            for index, task in enumerate(task_list):
                initial[task.block % workers].append(index)
            task_indices = [context.Array("i", bucket, lock=False) for bucket in initial]
            left_indices = [context.Value("i", 0, lock=False) for _ in range(workers)]
            right_indices = [context.Value("i", len(bucket), lock=False) for bucket in initial]
            locks = [context.Lock() for _ in range(workers)]
            processes = [
                context.Process(
                    target=_worker_steal,
                    args=(
                        worker,
                        task_indices,
                        left_indices,
                        right_indices,
                        locks,
                        task_list,
                        workers,
                        output,
                    ),
                )
                for worker in range(workers)
            ]
        else:
            raise ValueError(f"unknown scheduling method: {method}")
        for process in processes:
            process.start()
        worker_records = _collect(processes, output)
        elapsed = time.perf_counter() - started

    values = sorted(value for record in worker_records for value in record["values"])
    if len(values) != len(tasks) or len({task_id for task_id, _ in values}) != len(tasks):
        raise AssertionError("tasks were lost or duplicated")
    max_compute = max(record["compute_seconds"] for record in worker_records)
    return {
        "method": method,
        "workers": workers,
        "elapsed_seconds": elapsed,
        "residual_overhead_seconds": max(0.0, elapsed - max_compute),
        "outputs": values,
        "checksum": sum(value for _, value in values) & 0xFFFFFFFFFFFFFFFF,
        "worker_records": worker_records,
        "steals": sum(record["steals"] for record in worker_records),
        "remote_tasks": sum(record["remote_tasks"] for record in worker_records),
        "task_messages": len(tasks),
    }


def data_kernel(values: Sequence[int]) -> int:
    return sum(((value * value + 3 * value + 7) % 1_000_003) for value in values)


def _data_worker(values: Sequence[int]) -> int:
    return data_kernel(values)


def run_data_parallel(values: list[int], workers: int) -> dict:
    started = time.perf_counter()
    if workers == 1:
        result = data_kernel(values)
    else:
        context = mp.get_context("spawn")
        chunks = [values[index::workers] for index in range(workers)]
        with context.Pool(workers) as pool:
            result = sum(pool.map(_data_worker, chunks))
    return {
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "result": result,
        "communicated_bytes": 0 if workers == 1 else len(values) * 8,
    }


def _linear_partial(arguments: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    features, weights = arguments
    return features @ weights


def run_model_parallel(features: np.ndarray, weights: np.ndarray, workers: int) -> dict:
    started = time.perf_counter()
    if workers == 1:
        prediction = features @ weights
    else:
        context = mp.get_context("spawn")
        feature_parts = np.array_split(features, workers, axis=1)
        weight_parts = np.array_split(weights, workers)
        with context.Pool(workers) as pool:
            partials = pool.map(_linear_partial, list(zip(feature_parts, weight_parts)))
        prediction = np.sum(partials, axis=0)
    return {
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "prediction": prediction,
        "checksum": float(np.sum(prediction)),
        "communicated_bytes": 0 if workers == 1 else features.nbytes + weights.nbytes + prediction.nbytes * workers,
    }


def _stage_one(values: list[int]) -> list[int]:
    return [((value * 3) + 1) & 0xFFFFFFFF for value in values]


def _stage_two(values: list[int]) -> list[int]:
    return [((value ^ (value >> 7)) * 5) & 0xFFFFFFFF for value in values]


def _stage_three(values: list[int]) -> int:
    return sum(values) & 0xFFFFFFFFFFFFFFFF


def _pipeline_stage(input_queue: mp.Queue, output_queue: mp.Queue, function) -> None:
    while True:
        item = input_queue.get()
        if item is None:
            output_queue.put(None)
            return
        index, values = item
        output_queue.put((index, function(values)))


def run_pipeline(batches: list[list[int]], parallel: bool) -> dict:
    started = time.perf_counter()
    if not parallel:
        outputs = [(index, _stage_three(_stage_two(_stage_one(batch)))) for index, batch in enumerate(batches)]
    else:
        context = mp.get_context("spawn")
        queues = [context.Queue(maxsize=3) for _ in range(4)]
        processes = [
            context.Process(target=_pipeline_stage, args=(queues[0], queues[1], _stage_one)),
            context.Process(target=_pipeline_stage, args=(queues[1], queues[2], _stage_two)),
            context.Process(target=_pipeline_stage, args=(queues[2], queues[3], _stage_three)),
        ]
        for process in processes:
            process.start()
        for index, batch in enumerate(batches):
            queues[0].put((index, batch))
        queues[0].put(None)
        outputs = []
        while True:
            item = queues[3].get(timeout=180)
            if item is None:
                break
            outputs.append(item)
        for process in processes:
            process.join(timeout=180)
            if process.exitcode != 0:
                raise RuntimeError(f"pipeline stage exited with code {process.exitcode}")
    outputs.sort()
    return {
        "parallel": parallel,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": outputs,
        "checksum": sum(value for _, value in outputs) & 0xFFFFFFFFFFFFFFFF,
        "communicated_bytes": 0 if not parallel else sum(len(batch) for batch in batches) * 8 * 3,
    }
