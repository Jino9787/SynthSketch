import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("element_count", "Elemente", "#222222"),
    ("complexity", "Gesamtkomplexit\u00e4t", "#D64550"),
    ("line_count", "Anzahl Linien", "#2F6DB3"),
    ("line_ratio", "Linienanteil", "#8FBCE8"),
    ("arc_count", "Anzahl B\u00f6gen", "#D9791F"),
    ("arc_ratio", "Bogenanteil", "#F2B46D"),
    ("spline_count", "Anzahl Splines", "#2F8F4E"),
    ("spline_ratio", "Spline-Anteil", "#8BCF97"),
    ("control_point_total", "Kontrollpunkte gesamt", "#7A3EA1"),
    ("control_point_max", "Max. Kontrollpunkte", "#B691D1"),
]

METRIC_BINS = {
    "element_count": 60,
    "complexity": 50,
    "line_count": 45,
    "line_ratio": 35,
    "arc_count": 25,
    "arc_ratio": 35,
    "spline_ratio": 35,
    "control_point_total": 35,
    "control_point_max": 35,
}

FIXED_HISTOGRAM_RANGES = {
    "element_count": (0, 60),
    "complexity": (0, 10),
    "line_count": (0, 45),
    "line_ratio": (0, 1),
    "arc_count": (0, 15),
    "arc_ratio": (0, 1),
    "spline_count": (0, 6),
    "spline_ratio": (0, 1),
    "control_point_total": (0, 100),
    "control_point_max": (5, 20),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("complexity_outputs"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()


def read_metric_row(path):
    rows = np.load(path, allow_pickle=True).tolist()
    tokens = [row for row in rows if row and row[0] not in {"SOL", "EOS"}]
    token_types = [row[0] for row in tokens]
    element_count = len(tokens)
    line_count = token_types.count("L")
    arc_count = token_types.count("A")
    spline_count = token_types.count("S")
    control_point_counts = [
        1 + (len(row) - 1) // 2
        for row in tokens
        if row[0] == "S"
    ]
    control_point_total = sum(control_point_counts)
    complexity = (
        element_count / 60
        + control_point_total / 120
        + len(set(token_types)) / 4
    ) * 10 / 3
    safe_element_count = max(element_count, 1)

    return (
        element_count,
        complexity,
        line_count,
        line_count / safe_element_count,
        arc_count,
        arc_count / safe_element_count,
        spline_count,
        spline_count / safe_element_count,
        control_point_total,
        max(control_point_counts, default=0),
    )


def load_metric_rows(paths, worker_count=1):
    result = np.empty((len(paths), len(METRICS)), dtype=np.float64)
    if worker_count <= 1:
        for index, path in enumerate(paths):
            result[index] = read_metric_row(path)
        return result

    with Pool(processes=worker_count) as pool:
        for index, row in enumerate(pool.imap(read_metric_row, paths, chunksize=100)):
            result[index] = row
    return result


def should_use_discrete_bars(values):
    unique_values = np.unique(values)
    return len(unique_values) <= 12 and np.allclose(unique_values, np.round(unique_values))


def metric_range(metric_key, values):
    fixed_min, fixed_max = FIXED_HISTOGRAM_RANGES[metric_key]
    value_min = min(fixed_min, float(values.min()))
    value_max = max(fixed_max, float(values.max()))
    return value_min, value_max


def plot_histogram(axis, metric_key, values, color):
    value_min, value_max = metric_range(metric_key, values)
    if should_use_discrete_bars(values):
        unique_values, counts = np.unique(values, return_counts=True)
        bar_width = 0.8
        axis.bar(
            unique_values,
            counts,
            width=bar_width,
            color=color,
            edgecolor="white",
            alpha=0.85,
        )
        axis.set_xlim(value_min, value_max)
        return

    bin_count = max(5, METRIC_BINS[metric_key])
    bin_edges = np.linspace(value_min, value_max, bin_count + 1)
    axis.hist(
        values,
        bins=bin_edges,
        color=color,
        edgecolor="white",
        alpha=0.85,
    )
    axis.set_xlim(value_min, value_max)


def save_plot(rows, output_path, column_count):
    row_count = int(np.ceil(len(METRICS) / column_count))
    fig_width = 7 * column_count
    fig_height = 3.6 * row_count
    fig, axes = plt.subplots(row_count, column_count, figsize=(fig_width, fig_height))
    axes = axes.ravel()

    for metric_index, (axis, (metric_key, label, color)) in enumerate(
        zip(axes, METRICS)
    ):
        values = rows[:, metric_index]
        value_min = values.min()
        value_max = values.max()

        plot_histogram(axis, metric_key, values, color)
        axis.set_title(label)
        axis.set_ylabel("Anzahl")
        axis.grid(axis="y", alpha=0.25)

        axis.text(
            0.98,
            0.95,
            f"min={value_min:.3g}\nmax={value_max:.3g}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.85),
        )

    for axis in axes[len(METRICS):]:
        axis.axis("off")

    fig.suptitle(f"Geometry Complexity, n={len(rows)}", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input not found: {args.input_dir}")

    paths = list(args.input_dir.rglob("*.npy"))
    if not paths:
        raise SystemExit(f"No .npy files found in: {args.input_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "metrics.png"

    rows = load_metric_rows(paths, max(1, args.workers))
    save_plot(rows, png_path, args.columns)

    print(f"input_dir={args.input_dir.resolve()}")
    print(f"npy_count={len(paths)}")
    print(f"saved_png={png_path}")


if __name__ == "__main__":
    main()
