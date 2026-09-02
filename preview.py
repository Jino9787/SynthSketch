import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

import main as main_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()


def plot_elements_on_axis(axis, main_module, elements):
    plotters = {
        "line": main_curve.plot_line,
        "arc": main_curve.plot_arc,
        "spline": main_curve.plot_spline,
        "circle": main_curve.plot_circle,
    }

    for element in elements:
        plotters[element["type"]](axis, element)

    axis.set_aspect("equal")
    axis.autoscale()
    axis.margins(0.12)
    axis.axis("off")


def generate_shape(seed):
    random.seed(seed)
    np.random.seed(seed)
    elements = main_curve.generate_mixed_curve()
    return seed, elements


def main():
    args = parse_args()
    row_count, column_count = 5, 10
    _, axes = plt.subplots(row_count, column_count, figsize=(20, 10))

    seeds = [
        random.SystemRandom().randrange(2**32)
        for _ in range(row_count * column_count)
    ]
    worker_count = max(1, min(args.workers, len(seeds), os.cpu_count() or 1))
    if worker_count == 1:
        generated_shapes = map(generate_shape, seeds)
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count)
        generated_shapes = executor.map(generate_shape, seeds)

    for shape_index, (axis, result) in enumerate(
        zip(axes.flat, generated_shapes), start=1
    ):
        seed, elements = result
        print(f"shape {shape_index}, seed={seed}")
        main_curve.print_elements(elements)
        plot_elements_on_axis(axis, main, elements)
        axis.set_title(str(shape_index), fontsize=7, pad=1)
        axis.text(
            0.5,
            -0.08,
            f"{seed}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    if worker_count > 1:
        executor.shutdown()

    plt.show()

if __name__ == "__main__":
    main()
