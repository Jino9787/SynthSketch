import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main as main_curve


def invalid_npy_data():
    raise ValueError("Invalid NPY geometry data")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    return parser.parse_args()


def read_token_rows(path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() != ".npy":
        raise SystemExit(f"Not an .npy file: {path}")

    rows = np.load(path, allow_pickle=True)
    return [row.tolist() if isinstance(row, np.ndarray) else list(row) for row in rows]


def split_token_loops(rows):
    loops = []
    current_loop = None

    for row in rows:
        if not row:
            continue
        token_type = row[0]
        if token_type == "SOL":
            if current_loop is not None:
                invalid_npy_data()
            current_loop = []
        elif token_type == "EOS":
            if current_loop is None:
                invalid_npy_data()
            if current_loop:
                loops.append(current_loop)
            current_loop = None
        elif current_loop is None:
            invalid_npy_data()
        else:
            current_loop.append(row)

    if current_loop is not None:
        invalid_npy_data()
    return loops


def token_endpoint(token):
    token_type = token[0]
    values = [float(value) for value in token[1:]]
    if token_type == "L":
        if len(values) != 2:
            invalid_npy_data()
        return values
    if token_type == "A":
        if len(values) != 4:
            invalid_npy_data()
        return values[2:4]
    if token_type == "S":
        if len(values) < 4 or len(values) % 2:
            invalid_npy_data()
        return values[-2:]
    invalid_npy_data()


def token_to_element(token, start=None):
    token_type = token[0]
    values = [float(value) for value in token[1:]]

    if token_type == "C":
        if len(values) != 3:
            invalid_npy_data()
        return {
            "type": "circle",
            "center": values[:2],
            "radius": values[2],
        }
    if start is None:
        invalid_npy_data()
    if token_type == "L":
        return {"type": "line", "start": start, "end": values}
    if token_type == "A":
        return {
            "type": "arc",
            "start": start,
            "mid": values[:2],
            "end": values[2:4],
        }
    if token_type == "S":
        control_points = [start]
        control_points.extend(
            [values[index:index + 2] for index in range(0, len(values), 2)]
        )
        return {
            "type": "spline",
            "start": start,
            "end": control_points[-1],
            "ctrlpts": control_points,
        }
    invalid_npy_data()


def restore_elements(token_loops):
    elements = []
    for token_loop in token_loops:
        if len(token_loop) == 1 and token_loop[0][0] == "C":
            elements.append(token_to_element(token_loop[0]))
            continue
        if any(token[0] == "C" for token in token_loop):
            invalid_npy_data()

        start = token_endpoint(token_loop[-1])
        for token in token_loop:
            element = token_to_element(token, start)
            elements.append(element)
            start = token_endpoint(token)
    return elements


def plot_elements(elements, path):
    figure, axis = plt.subplots(figsize=(8, 8))
    for element in elements:
        points = main_curve.sample_element_points(element)
        axis.plot(points[:, 0], points[:, 1], color="black", linewidth=1.0)

    axis.set_aspect("equal")
    axis.autoscale()
    axis.margins(0.1)
    axis.axis("off")
    axis.set_title(path.name)
    figure.tight_layout()
    plt.show()


def main():
    args = parse_args()
    rows = read_token_rows(args.file)
    token_loops = split_token_loops(rows)
    elements = restore_elements(token_loops)
    plot_elements(elements, args.file)


if __name__ == "__main__":
    main()
