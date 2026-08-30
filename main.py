import numpy as np
import matplotlib.pyplot as plt
from geomdl import BSpline, utilities
import random


MAX_ELEMENTS = 60

COMPLEXITY_WEIGHTS = {
    "simple": 0.2,
    "medium": 0.3,
    "complex": 0.5,
}


def choose_complexity():
    names, weights = zip(*COMPLEXITY_WEIGHTS.items())
    return random.choices(names, weights=weights, k=1)[0]

def generate_mixed_curve_once():
    complexity = choose_complexity()
    basic_shape_points = generate_basic_shape(complexity)
    points = add_distributed_edge_features(basic_shape_points, complexity)
    elements = build_corner_treated_elements(points, original_points=basic_shape_points)
    edge_lengths = [
        np.linalg.norm(np.array(element["end"]) - np.array(element["start"]))
        if element["type"] in {"line", "arc"}
        else 0.0
        for element in elements
    ]
    drawable_edges = [
        i
        for i, edge_length in enumerate(edge_lengths)
        if edge_length > DRAWABLE_EDGE_MIN_LENGTH
    ]
    if not drawable_edges:
        drawable_edges = list(range(len(elements)))

    arc_candidates = [
        i
        for i in drawable_edges
        if elements[i]["type"] == "line"
    ]
    arc_count = min(random.randint(*ARC_INSERT_COUNT_RANGE), len(arc_candidates))
    for index in random.sample(arc_candidates, arc_count):
        elements[index] = generate_arc(elements[index]["start"], elements[index]["end"])

    elements = insert_local_splines(elements, complexity)
    elements = enforce_spline_ratio_limit(elements)

    return elements, complexity

def generate_mixed_curve(max_elements=MAX_ELEMENTS):
    for _ in range(50):
        outer_elements, complexity = generate_mixed_curve_once()
        if has_self_intersection(outer_elements):
            continue
        if not has_minimum_segment_distance(outer_elements):
            continue

        sampled_outer = elements_to_polygon(outer_elements)
        if len(sampled_outer) < 3:
            continue

        hole_elements = generate_hole_elements(sampled_outer, complexity)
        elements = outer_elements + hole_elements
        if len(elements) > max_elements:
            continue
        if (
            not has_self_intersection(elements)
            and has_minimum_segment_distance(elements)
        ):
            return transform_elements(elements)

    points = generate_rectangle_form(BASIC_SHAPE_FRAME_SIZE, BASIC_SHAPE_FRAME_SIZE)
    elements = build_corner_treated_elements(points, original_points=points)
    elements = insert_local_splines(elements, "simple")
    elements = enforce_spline_ratio_limit(elements)
    return transform_elements(elements)


BASIC_SHAPE_FRAME_SIZE = 2.0

RECTANGLE_SIZE_RATIO_RANGE = (0.2, 1.0)

L_SHAPE_CUTOUT_RATIO_RANGE = (0.2, 0.5)

L_SHAPE_CUTOUT_CORNER_CHOICES = ["top_right", "top_left", "bottom_right", "bottom_left"]

BASIC_SHAPE_ROTATION_CHOICES = [0, 90, 180, 270]

T_SHAPE_BAR_WIDTH_RATIO_RANGE = (0.6, 1.0)

T_SHAPE_BAR_HEIGHT_RATIO_RANGE = (0.2, 0.5)

T_SHAPE_STEM_WIDTH_RATIO_RANGE = (0.2, 0.5)

T_SHAPE_STEM_OFFSET_RATIO_RANGE = (0.0, 0.8)

U_SHAPE_WALL_RATIO_RANGE = (0.1, 0.3)

U_SHAPE_CUTOUT_DEPTH_RATIO_RANGE = (0.3, 0.8)

U_SHAPE_MIN_CUTOUT_WIDTH_RATIO = 0.2

U_SHAPE_SIDE_MARGIN_RATIO = 0.08

STEPPED_SHAPE_STEP_COUNT_RANGE = (1, 5)

RECTANGLE_WITH_CUTOUT_DEPTH_RANGE = (0.2, 0.6)

MULTI_RECTANGLE_ROW_CHOICES = [3, 4, 5]

MULTI_RECTANGLE_COLUMN_CHOICES = [3, 4, 5]

MULTI_RECTANGLE_TARGET_CELL_COUNT_RANGE = (3, 9)

MULTI_RECTANGLE_EVEN_CELL_REMOVAL_PROBABILITY = 0.75

MULTI_RECTANGLE_CELL_SCALE_RANGE = (0.8, 1.2)

MULTI_RECTANGLE_MAX_GENERATION_ATTEMPTS = 30

BASIC_SHAPES = [
    "rectangle",
    "l_shape",
    "t_shape",
    "u_shape",
    "stepped_shape",
    "rectangle_with_cutout",
    "multi_rectangle",
]

BASIC_SHAPE_WEIGHTS = {
    "simple": [
        ("rectangle", 0.35),
        ("l_shape", 0.3),
        ("t_shape", 0.15),
        ("rectangle_with_cutout", 0.15),
        ("u_shape", 0.15),
    ],
    "medium": [
        ("l_shape", 0.15),
        ("t_shape", 0.15),
        ("u_shape", 0.15),
        ("rectangle_with_cutout", 0.15),
        ("stepped_shape", 0.2),
        ("multi_rectangle", 0.2),
    ],
    "complex": [
        ("t_shape", 0.1),
        ("u_shape", 0.1),
        ("stepped_shape", 0.35),
        ("rectangle_with_cutout", 0.1),
        ("multi_rectangle", 0.35),
    ],
}


def add_point(points, x, y):
    point = [round(float(x), 3), round(float(y), 3)]
    if not points or point != points[-1]:
        points.append(point)

def point_on_side(start, end, t):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    return start + (end - start) * t

def polygon_area(points):
    area = 0.0
    for i, point in enumerate(points):
        next_point = points[(i + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area / 2

def clean_points(points):
    cleaned = []
    for point in points:
        add_point(cleaned, point[0], point[1])
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned

def normalize_ccw(points):
    points = clean_points(points)
    if polygon_area(points) < 0:
        points.reverse()
    return points

def generate_rectangle_points(width, height):
    left, right = -width / 2, width / 2
    bottom, top = -height / 2, height / 2
    return [[left, bottom], [right, bottom], [right, top], [left, top]]

def generate_rectangle_form(width, height):
    rectangle_width = width * random.uniform(*RECTANGLE_SIZE_RATIO_RANGE)
    rectangle_height = height * random.uniform(*RECTANGLE_SIZE_RATIO_RANGE)
    return generate_rectangle_points(rectangle_width, rectangle_height)

def generate_l_shape_form(width, height):
    left, right = -width / 2, width / 2
    bottom, top = -height / 2, height / 2
    cutout_width = width * random.uniform(*L_SHAPE_CUTOUT_RATIO_RANGE)
    cutout_height = height * random.uniform(*L_SHAPE_CUTOUT_RATIO_RANGE)
    cutout_corner = random.choice(L_SHAPE_CUTOUT_CORNER_CHOICES)

    if cutout_corner == "top_right":
        points = [
            [left, bottom], [right, bottom], [right, top - cutout_height],
            [right - cutout_width, top - cutout_height], [right - cutout_width, top], [left, top],
        ]
    elif cutout_corner == "top_left":
        points = [
            [left, bottom], [right, bottom], [right, top],
            [left + cutout_width, top], [left + cutout_width, top - cutout_height], [left, top - cutout_height],
        ]
    elif cutout_corner == "bottom_right":
        points = [
            [left, bottom], [right - cutout_width, bottom], [right - cutout_width, bottom + cutout_height],
            [right, bottom + cutout_height], [right, top], [left, top],
        ]
    else:
        points = [
            [left + cutout_width, bottom], [right, bottom], [right, top],
            [left, top], [left, bottom + cutout_height], [left + cutout_width, bottom + cutout_height],
        ]
    return points

def rotate_basic_shape(points, rotation):
    if rotation == 0:
        return points

    rotated = []
    for x, y in points:
        if rotation == 90:
            rotated.append([-y, x])
        elif rotation == 180:
            rotated.append([-x, -y])
        elif rotation == 270:
            rotated.append([y, -x])
    return rotated

def generate_t_shape_form(width, height):
    t_bar_width = width * random.uniform(*T_SHAPE_BAR_WIDTH_RATIO_RANGE)
    t_bar_height = height * random.uniform(*T_SHAPE_BAR_HEIGHT_RATIO_RANGE)
    t_stem_width = width * random.uniform(*T_SHAPE_STEM_WIDTH_RATIO_RANGE)
    bottom, top = -height / 2, height / 2
    max_stem_offset = max(0.0, (t_bar_width - t_stem_width) / 2)
    t_stem_center = random.uniform(-max_stem_offset, max_stem_offset) * random.uniform(*T_SHAPE_STEM_OFFSET_RATIO_RANGE)
    left, right = -t_bar_width / 2, t_bar_width / 2
    stem_left, stem_right = t_stem_center - t_stem_width / 2, t_stem_center + t_stem_width / 2
    bar_bottom = top - t_bar_height
    points = [
        [stem_left, bottom], [stem_right, bottom], [stem_right, bar_bottom],
        [right, bar_bottom], [right, top], [left, top],
        [left, bar_bottom], [stem_left, bar_bottom],
    ]
    return rotate_basic_shape(points, random.choice(BASIC_SHAPE_ROTATION_CHOICES))

def generate_u_shape_form(width, height):
    left, right = -width / 2, width / 2
    bottom, top = -height / 2, height / 2
    u_left_wall = width * random.uniform(*U_SHAPE_WALL_RATIO_RANGE)
    u_right_wall = width * random.uniform(*U_SHAPE_WALL_RATIO_RANGE)
    cutout_depth = height * random.uniform(*U_SHAPE_CUTOUT_DEPTH_RATIO_RANGE)
    minimum_cutout_width = width * U_SHAPE_MIN_CUTOUT_WIDTH_RATIO
    if width - u_left_wall - u_right_wall < minimum_cutout_width:
        total_wall_width = width - minimum_cutout_width
        left_wall_ratio = u_left_wall / (u_left_wall + u_right_wall)
        u_left_wall = total_wall_width * left_wall_ratio
        u_right_wall = total_wall_width * (1 - left_wall_ratio)
    horizontal_slack = width - u_left_wall - u_right_wall - minimum_cutout_width
    cutout_shift = random.uniform(-horizontal_slack / 2, horizontal_slack / 2)
    cutout_left = left + u_left_wall + cutout_shift
    cutout_right = right - u_right_wall + cutout_shift
    cutout_left = max(left + width * U_SHAPE_SIDE_MARGIN_RATIO, cutout_left)
    cutout_right = min(right - width * U_SHAPE_SIDE_MARGIN_RATIO, cutout_right)
    cutout_bottom = top - cutout_depth

    points = [
        [left, bottom], [right, bottom], [right, top],
        [cutout_right, top], [cutout_right, cutout_bottom],
        [cutout_left, cutout_bottom], [cutout_left, top], [left, top],
    ]
    return rotate_basic_shape(points, random.choice(BASIC_SHAPE_ROTATION_CHOICES))

def generate_stepped_shape_form(width, height):
    left, right = -width / 2, width / 2
    bottom, top = -height / 2, height / 2
    step_count = random.randint(*STEPPED_SHAPE_STEP_COUNT_RANGE)
    points = [[left, bottom], [right, bottom], [right, top]]
    for step_index in range(step_count):
        step_x = right - width * (step_index + 1) / (step_count + 1)
        step_y = top - height * (step_index + 1) / (step_count + 1)
        points.append([step_x, top - height * step_index / (step_count + 1)])
        points.append([step_x, step_y])
    points.extend([[left, bottom + height / (step_count + 1)], [left, bottom]])
    return points[:-1]

def generate_rectangle_with_cutout_form(width, height):
    points = generate_rectangle_points(width, height)
    sides = {
        "bottom": (points[0], points[1], [0, 1]),
        "right": (points[1], points[2], [-1, 0]),
        "top": (points[2], points[3], [0, -1]),
        "left": (points[3], points[0], [1, 0]),
    }
    eligible_sides = [
        name
        for name, (start, end, _inward) in sides.items()
        if (
            np.linalg.norm(np.array(end) - np.array(start))
            * EDGE_FEATURE_RELATIVE_WIDTH_RANGE[1]
        )
        >= MIN_EDGE_FEATURE_WIDTH
    ]
    if not eligible_sides:
        return points

    side = random.choice(eligible_sides)
    new_points = []
    for name in ["bottom", "right", "top", "left"]:
        start, end, inward = sides[name]
        add_point(new_points, start[0], start[1])
        if name == side:
            edge_length = np.linalg.norm(np.array(end) - np.array(start))
            feature = random_feature_spec(name, edge_length)
            if feature is not None:
                feature["kind"] = "notch"
                feature["depth"] = random.uniform(*RECTANGLE_WITH_CUTOUT_DEPTH_RANGE)
                add_side_feature(new_points, start, end, inward, feature)
        add_point(new_points, end[0], end[1])
    return new_points

def generate_multi_rectangle_form(width, height):
    return generate_irregular_multi_rectangle_form(width, height)

def generate_irregular_multi_rectangle_form(width, height):
    for _ in range(MULTI_RECTANGLE_MAX_GENERATION_ATTEMPTS):
        points = generate_irregular_multi_rectangle_form_once(width, height)
        if form_points_are_valid(points):
            return points

    return generate_rectangle_points(width, height)

def generate_irregular_multi_rectangle_form_once(width, height):
    grid_row_count = random.choice(MULTI_RECTANGLE_ROW_CHOICES)
    grid_column_count = random.choice(MULTI_RECTANGLE_COLUMN_CHOICES)
    cells = {(grid_column_count // 2, grid_row_count // 2)}

    target_cell_count = random.randint(
        MULTI_RECTANGLE_TARGET_CELL_COUNT_RANGE[0],
        min(grid_row_count * grid_column_count - 1, MULTI_RECTANGLE_TARGET_CELL_COUNT_RANGE[1]),
    )
    while len(cells) < target_cell_count:
        cell = random.choice(list(cells))
        neighbors = [
            (cell[0] + 1, cell[1]),
            (cell[0] - 1, cell[1]),
            (cell[0], cell[1] + 1),
            (cell[0], cell[1] - 1),
        ]
        candidates = [
            neighbor
            for neighbor in neighbors
            if 0 <= neighbor[0] < grid_column_count and 0 <= neighbor[1] < grid_row_count
        ]
        if candidates:
            cells.add(random.choice(candidates))

    if len(cells) % 2 == 0 and random.random() < MULTI_RECTANGLE_EVEN_CELL_REMOVAL_PROBABILITY:
        removable = [
            cell
            for cell in cells
            if cell != (grid_column_count // 2, grid_row_count // 2)
            and any(
                neighbor not in cells
                for neighbor in [
                    (cell[0] + 1, cell[1]),
                    (cell[0] - 1, cell[1]),
                    (cell[0], cell[1] + 1),
                    (cell[0], cell[1] - 1),
                ]
            )
        ]
        if removable:
            cells.remove(random.choice(removable))

    cell_width = width / grid_column_count * random.uniform(*MULTI_RECTANGLE_CELL_SCALE_RANGE)
    cell_height = height / grid_row_count * random.uniform(*MULTI_RECTANGLE_CELL_SCALE_RANGE)
    origin_x = -grid_column_count * cell_width / 2
    origin_y = -grid_row_count * cell_height / 2
    edges = []

    for col, row in cells:
        x0 = origin_x + col * cell_width
        x1 = x0 + cell_width
        y0 = origin_y + row * cell_height
        y1 = y0 + cell_height
        if (col, row - 1) not in cells:
            edges.append(((x0, y0), (x1, y0)))
        if (col + 1, row) not in cells:
            edges.append(((x1, y0), (x1, y1)))
        if (col, row + 1) not in cells:
            edges.append(((x1, y1), (x0, y1)))
        if (col - 1, row) not in cells:
            edges.append(((x0, y1), (x0, y0)))

    return edges_to_ordered_points(edges)

def edges_to_ordered_points(edges):
    edge_map = {}
    for start, end in edges:
        start_key = (round(start[0], 10), round(start[1], 10))
        edge_map[start_key] = [float(end[0]), float(end[1])]

    start = min(edge_map.keys(), key=lambda point: (point[1], point[0]))
    points = [[float(start[0]), float(start[1])]]
    current = start

    for _ in range(len(edges)):
        next_point = edge_map[current]
        next_key = (round(next_point[0], 10), round(next_point[1], 10))
        if next_key == start:
            break
        points.append(next_point)
        current = next_key

    return remove_collinear_points(points)

def remove_collinear_points(points, eps=1e-9):
    if len(points) <= 2:
        return points

    cleaned = []
    count = len(points)
    for index, point in enumerate(points):
        prev_point = np.array(points[index - 1], dtype=float)
        current_point = np.array(point, dtype=float)
        next_point = np.array(points[(index + 1) % count], dtype=float)
        before = current_point - prev_point
        after = next_point - current_point

        if np.linalg.norm(before) < eps or np.linalg.norm(after) < eps:
            continue
        if abs(cross_2d(before, after)) < eps:
            continue

        cleaned.append(point)

    return cleaned

def generate_basic_shape(complexity):
    width = BASIC_SHAPE_FRAME_SIZE
    height = BASIC_SHAPE_FRAME_SIZE
    basic_shape_names, weights = zip(*BASIC_SHAPE_WEIGHTS[complexity])
    basic_shape = random.choices(basic_shape_names, weights=weights, k=1)[0]
    basic_shape_generators = dict(
        zip(
            BASIC_SHAPES,
            [
                generate_rectangle_form,
                generate_l_shape_form,
                generate_t_shape_form,
                generate_u_shape_form,
                generate_stepped_shape_form,
                generate_rectangle_with_cutout_form,
                generate_multi_rectangle_form,
            ],
        )
    )
    return normalize_ccw(basic_shape_generators[basic_shape](width, height))


ARC_MID_OFFSET_RANGE = (-0.2, 0.2)

EDGE_FEATURE_RELATIVE_WIDTH_RANGE = (0.1, 0.4)

MIN_EDGE_FEATURE_WIDTH = 0.05

EDGE_FEATURE_POSITION_MODES = ["near_start", "middle", "near_end"]

EDGE_FEATURE_NEAR_START_RANGE = (0.05, 0.2)

EDGE_FEATURE_MIDDLE_RANGE = (0.2, 0.9)

EDGE_FEATURE_NEAR_END_RANGE = (0.8, 0.9)

EDGE_FEATURE_KINDS = ["notch", "tab"]

EDGE_FEATURE_SHAPE_WEIGHTS = {
    "rect": 0.9,
    "trapezoid": 0.1,
}

EDGE_FEATURE_TRAPEZOID_TOP_RATIO_RANGE = (0.5, 0.9)

EDGE_FEATURE_DEPTH_RANGE = (0.2, 0.4)

EDGE_FEATURE_OVERLAP_GAP = 0.05

CORNER_RIGHT_ANGLE_COS_THRESHOLD = 0.01

CORNER_MIN_EDGE_LENGTH = 0.6

CORNER_TREATMENT_PROBABILITY = 0.33

CORNER_CUT_DISTANCE_RATIO_RANGE = (0.15, 0.25)

CORNER_TREATMENTS = ["chamfer", "round"]

EDGE_FEATURE_COUNTS = {
    "simple": (0, 1),
    "medium": (2, 4),
    "complex": (4, 6),
}

EDGE_FEATURE_MAX_BUDGET = {
    "simple": 1,
    "medium": 3,
    "complex": 4,
}

POCKET_CORNER_RADIUS_RATIO_RANGE = (0.12, 0.25)

POCKET_MIN_CORNER_RADIUS = 0.02

HOLE_SAMPLE_MARGIN = 0.2

HOLE_SAMPLE_ATTEMPTS = 100

HOLE_OUTER_CLEARANCE = 0.05

HOLE_EXISTING_CLEARANCE = 0.1

HOLE_COUNTS = {
    "simple": (0, 1),
    "medium": (2, 3),
    "complex": (4, 5),
}

SYMMETRIC_CIRCLE_HOLE_PROBABILITIES = {
    "simple": 0.2,
    "medium": 0.4,
    "complex": 0.6,
}

SYMMETRIC_CIRCLE_X_OFFSET_RANGE = (0.2, 0.5)

SYMMETRIC_CIRCLE_Y_RANGE = (-0.2, 0.2)

CIRCLE_HOLE_RADIUS_RANGE = (0.05, 0.15)

POCKET_SIZE_RANGE = (0.2, 0.4)

RANDOM_HOLE_CIRCLE_PROBABILITY = 0.5

DRAWABLE_EDGE_MIN_LENGTH = 0.15

ARC_INSERT_COUNT_RANGE = (0, 3)


def generate_line(start, end, role="outer", generation_stage="base", color=None):
    if color is None:
        color = GENERATION_STAGE_COLORS[generation_stage]
    return {
        "type": "line",
        "role": role,
        "start": start,
        "end": end,
        "generation_stage": generation_stage,
        "color": color,
    }

def generate_arc(start, end):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    chord = end - start
    length = np.linalg.norm(chord)
    if length < 1e-6:
        return generate_line(start.tolist(), end.tolist(), generation_stage='arc_replacement')

    normal = np.array([-chord[1], chord[0]]) / length
    mid = (start + end) / 2
    mid += normal * length * random.uniform(*ARC_MID_OFFSET_RANGE)
    return {
        "type": "arc",
        "role": "outer",
        "start": start.tolist(),
        "mid": mid.tolist(),
        "end": end.tolist(),
        "generation_stage": "arc_replacement",
        "color": GENERATION_STAGE_COLORS["arc_replacement"],
    }

def generate_circle(center, radius, role="hole"):
    return {
        "type": "circle",
        "role": role,
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "generation_stage": "hole_or_pocket",
        "color": GENERATION_STAGE_COLORS["hole_or_pocket"],
    }

def element_lies_on_form(start, end, form_points):
    if form_points is None:
        return True
    for index, edge_start in enumerate(form_points):
        edge_end = form_points[(index + 1) % len(form_points)]
        if (
            point_lies_on_segment(start, edge_start, edge_end)
            and point_lies_on_segment(end, edge_start, edge_end)
        ):
            return True
    return False

def add_side_feature(points, start, end, inward, feature):
    length = np.linalg.norm(np.array(end) - np.array(start))
    p0 = point_on_side(start, end, feature["t0"])
    p1 = point_on_side(start, end, feature["t1"])

    direction = np.array(inward, dtype=float)
    if feature["kind"] == "tab":
        direction *= -1

    offset = direction * length * feature["depth"]

    if feature.get("shape") == "trapezoid":
        top_ratio = feature.get("top_ratio", 0.65)
        inset = (p1 - p0) * (1 - top_ratio) / 2
        top0 = p0 + inset + offset
        top1 = p1 - inset + offset

        add_point(points, p0[0], p0[1])
        add_point(points, top0[0], top0[1])
        add_point(points, top1[0], top1[1])
        add_point(points, p1[0], p1[1])
        return

    add_point(points, p0[0], p0[1])
    add_point(points, p0[0] + offset[0], p0[1] + offset[1])
    add_point(points, p1[0] + offset[0], p1[1] + offset[1])
    add_point(points, p1[0], p1[1])

def random_feature_spec(side_name, edge_length):
    min_span, max_span = EDGE_FEATURE_RELATIVE_WIDTH_RANGE
    min_span = max(min_span, MIN_EDGE_FEATURE_WIDTH / edge_length)
    if min_span > max_span:
        return None

    span = random.uniform(min_span, max_span)
    position_mode = random.choice(EDGE_FEATURE_POSITION_MODES)

    if position_mode == "near_start":
        t0 = random.uniform(EDGE_FEATURE_NEAR_START_RANGE[0], min(EDGE_FEATURE_NEAR_START_RANGE[1], 0.9 - span))
    elif position_mode == "near_end":
        t0 = random.uniform(max(0.1, EDGE_FEATURE_NEAR_END_RANGE[0] - span), EDGE_FEATURE_NEAR_END_RANGE[1] - span)
    else:
        t0 = random.uniform(EDGE_FEATURE_MIDDLE_RANGE[0], EDGE_FEATURE_MIDDLE_RANGE[1] - span)

    shape_names, shape_weights = zip(*EDGE_FEATURE_SHAPE_WEIGHTS.items())
    shape = random.choices(shape_names, weights=shape_weights, k=1)[0]
    feature = {
        "side": side_name,
        "t0": t0,
        "t1": t0 + span,
        "kind": random.choice(EDGE_FEATURE_KINDS),
        "shape": shape,
        "depth": random.uniform(*EDGE_FEATURE_DEPTH_RANGE),
    }
    if shape == "trapezoid":
        feature["top_ratio"] = random.uniform(*EDGE_FEATURE_TRAPEZOID_TOP_RATIO_RANGE)

    return feature

def has_feature_overlap(features, new_feature, gap=None):
    if gap is None:
        gap = EDGE_FEATURE_OVERLAP_GAP
    for feature in features:
        if feature["side"] != new_feature["side"]:
            continue
        if (
            new_feature["t0"] < feature["t1"] + gap
            and feature["t0"] < new_feature["t1"] + gap
        ):
            return True
    return False

def add_feature_with_spacing(features, feature):
    if not has_feature_overlap(features, feature):
        features.append(feature)
        return True
    return False

def generate_corner_arc(start, corner, end, center=None):
    start = np.array(start, dtype=float)
    corner = np.array(corner, dtype=float)
    end = np.array(end, dtype=float)
    if center is None:
        center = corner
    else:
        center = np.array(center, dtype=float)

    start_vec = start - center
    end_vec = end - center
    radius = 0.5 * (np.linalg.norm(start_vec) + np.linalg.norm(end_vec))

    if radius < 1e-6:
        return generate_line(start.tolist(), end.tolist(), generation_stage='corner_replacement')

    mid_vec = start_vec + end_vec
    if np.linalg.norm(mid_vec) < 1e-6:
        mid_vec = np.array([-start_vec[1], start_vec[0]])
    mid_vec = mid_vec / np.linalg.norm(mid_vec)
    mid = center + mid_vec * radius

    if np.linalg.norm(corner - mid) > radius:
        mid = center - mid_vec * radius

    return {
        "type": "arc",
        "role": "outer",
        "start": start.tolist(),
        "mid": mid.tolist(),
        "end": end.tolist(),
        "generation_stage": "corner_replacement",
        "color": GENERATION_STAGE_COLORS["corner_replacement"],
    }

def build_corner_treated_elements(points, original_points=None):
    if len(points) < 4:
        return [
            generate_line(
                points[i],
                points[(i + 1) % len(points)],
                generation_stage=(
                    "base"
                    if element_lies_on_form(points[i], points[(i + 1) % len(points)], original_points)
                    else "edge_feature"
                ),
            )
            for i in range(len(points))
        ]

    entries = []
    exits = []
    treatments = []

    for i, point in enumerate(points):
        prev_point = np.array(points[i - 1], dtype=float)
        current = np.array(point, dtype=float)
        next_point = np.array(points[(i + 1) % len(points)], dtype=float)

        incoming = prev_point - current
        outgoing = next_point - current
        incoming_length = np.linalg.norm(incoming)
        outgoing_length = np.linalg.norm(outgoing)

        if incoming_length < 1e-6 or outgoing_length < 1e-6:
            entries.append(current)
            exits.append(current)
            treatments.append("none")
            continue

        cos_angle = np.dot(incoming, outgoing) / (incoming_length * outgoing_length)
        is_right_angle = abs(cos_angle) < CORNER_RIGHT_ANGLE_COS_THRESHOLD
        can_treat_corner = min(incoming_length, outgoing_length) > CORNER_MIN_EDGE_LENGTH

        if is_right_angle and can_treat_corner and random.random() < CORNER_TREATMENT_PROBABILITY:
            distance = min(incoming_length, outgoing_length) * random.uniform(*CORNER_CUT_DISTANCE_RATIO_RANGE)
            entries.append(current + incoming / incoming_length * distance)
            exits.append(current + outgoing / outgoing_length * distance)
            treatments.append(random.choice(CORNER_TREATMENTS))
        else:
            entries.append(current)
            exits.append(current)
            treatments.append("none")

    elements = []
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        side_start = exits[i].tolist()
        side_end = entries[next_i].tolist()
        if np.linalg.norm(np.array(side_end) - np.array(side_start)) > 1e-6:
            side_stage = (
                "base"
                if element_lies_on_form(side_start, side_end, original_points)
                else "edge_feature"
            )
            elements.append(generate_line(side_start, side_end, generation_stage=side_stage))

        if treatments[next_i] == "chamfer":
            elements.append(
                generate_line(entries[next_i].tolist(), exits[next_i].tolist(), generation_stage='corner_replacement')
            )
        elif treatments[next_i] == "round":
            corner = np.array(points[next_i], dtype=float)
            round_center = entries[next_i] + exits[next_i] - corner
            elements.append(
                generate_corner_arc(
                    entries[next_i].tolist(),
                    corner.tolist(),
                    exits[next_i].tolist(),
                    center=round_center.tolist(),
                )
            )

    return elements

def add_distributed_edge_features(points, complexity):
    count_spec = EDGE_FEATURE_COUNTS[complexity]
    if isinstance(count_spec, tuple):
        feature_count = random.randint(*count_spec)
    else:
        feature_count = random.choice(count_spec)
    if feature_count == 0:
        return points

    side_features = {edge_index: [] for edge_index in range(len(points))}
    edge_indices = [
        edge_index
        for edge_index in range(len(points))
        if np.linalg.norm(
            np.array(points[(edge_index + 1) % len(points)], dtype=float)
            - np.array(points[edge_index], dtype=float)
        )
        * EDGE_FEATURE_RELATIVE_WIDTH_RANGE[1]
        >= MIN_EDGE_FEATURE_WIDTH
    ]
    if not edge_indices:
        return points

    active_edge_count = min(len(edge_indices), random.randint(1, max(1, min(len(edge_indices), feature_count))))
    active_edges = random.sample(edge_indices, active_edge_count)

    remaining = feature_count
    for edge_index in active_edges:
        if remaining <= 0:
            break
        edge_budget = random.randint(1, min(remaining, EDGE_FEATURE_MAX_BUDGET[complexity]))
        for feature_attempt in range(edge_budget):
            edge_length = np.linalg.norm(
                np.array(points[(edge_index + 1) % len(points)], dtype=float)
                - np.array(points[edge_index], dtype=float)
            )
            feature = random_feature_spec(str(edge_index), edge_length)
            if feature is not None and add_feature_with_spacing(side_features[edge_index], feature):
                remaining -= 1
            if remaining <= 0:
                break

    enhanced = []
    orientation = 1 if polygon_area(points) > 0 else -1
    for edge_index, start in enumerate(points):
        end = points[(edge_index + 1) % len(points)]
        start_np = np.array(start, dtype=float)
        end_np = np.array(end, dtype=float)
        chord = end_np - start_np
        length = np.linalg.norm(chord)
        if length < 1e-6:
            continue
        inward = np.array([-chord[1], chord[0]]) / length
        if orientation < 0:
            inward *= -1

        add_point(enhanced, start[0], start[1])
        for feature in sorted(side_features[edge_index], key=lambda item: item["t0"]):
            add_side_feature(enhanced, start, end, inward, feature)

    return normalize_ccw(enhanced)

def create_circle_hole(center, radius):
    return [generate_circle(center, radius, role="hole")]

def create_pocket(center, width, height):
    cx, cy = center
    left = cx - width / 2
    right = cx + width / 2
    bottom = cy - height / 2
    top = cy + height / 2
    radius = max(
        min(width, height) * random.uniform(*POCKET_CORNER_RADIUS_RATIO_RANGE),
        POCKET_MIN_CORNER_RADIUS,
    )
    diagonal = radius / np.sqrt(2)

    def pocket_line(start, end):
        return generate_line(start, end, role='pocket', generation_stage='hole_or_pocket')

    def pocket_arc(start, mid, end):
        return {
            "type": "arc",
            "role": "pocket",
            "start": [float(start[0]), float(start[1])],
            "mid": [float(mid[0]), float(mid[1])],
            "end": [float(end[0]), float(end[1])],
            "generation_stage": "hole_or_pocket",
            "color": GENERATION_STAGE_COLORS["hole_or_pocket"],
        }

    bottom_right_center = [right - radius, bottom + radius]
    top_right_center = [right - radius, top - radius]
    top_left_center = [left + radius, top - radius]
    bottom_left_center = [left + radius, bottom + radius]

    return [
        pocket_line([left + radius, bottom], [right - radius, bottom]),
        pocket_arc(
            [right - radius, bottom],
            [bottom_right_center[0] + diagonal, bottom_right_center[1] - diagonal],
            [right, bottom + radius],
        ),
        pocket_line([right, bottom + radius], [right, top - radius]),
        pocket_arc(
            [right, top - radius],
            [top_right_center[0] + diagonal, top_right_center[1] + diagonal],
            [right - radius, top],
        ),
        pocket_line([right - radius, top], [left + radius, top]),
        pocket_arc(
            [left + radius, top],
            [top_left_center[0] - diagonal, top_left_center[1] + diagonal],
            [left, top - radius],
        ),
        pocket_line([left, top - radius], [left, bottom + radius]),
        pocket_arc(
            [left, bottom + radius],
            [bottom_left_center[0] - diagonal, bottom_left_center[1] - diagonal],
            [left + radius, bottom],
        ),
    ]

def sample_point_inside(points, margin=None):
    if margin is None:
        margin = HOLE_SAMPLE_MARGIN
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    for _ in range(HOLE_SAMPLE_ATTEMPTS):
        candidate = [
            random.uniform(min(xs) + margin, max(xs) - margin),
            random.uniform(min(ys) + margin, max(ys) - margin),
        ]
        if point_in_polygon(candidate, points):
            return candidate
    return [
        sum(xs) / len(xs),
        sum(ys) / len(ys),
    ]

def generate_hole_elements(points, complexity):
    count_spec = HOLE_COUNTS[complexity]
    if isinstance(count_spec, tuple):
        hole_count = random.randint(*count_spec)
    else:
        hole_count = random.choice(count_spec)
    holes = []
    accepted_holes = []

    if random.random() < SYMMETRIC_CIRCLE_HOLE_PROBABILITIES[complexity]:
        x_offset = random.uniform(*SYMMETRIC_CIRCLE_X_OFFSET_RANGE)
        y = random.uniform(*SYMMETRIC_CIRCLE_Y_RANGE)
        radius = random.uniform(*CIRCLE_HOLE_RADIUS_RANGE)
        centers = [[-x_offset, y], [x_offset, y]]
        if all(point_in_polygon(center, points) for center in centers):
            for center in centers:
                hole = create_circle_hole(center, radius)
                if (
                    hole_elements_inside(hole, points)
                    and hole_clear_of_existing(hole, accepted_holes)
                ):
                    holes.extend(hole)
                    accepted_holes.append(sample_hole_points(hole))
            hole_count = max(0, hole_count - 2)

    for _ in range(hole_count):
        center = sample_point_inside(points)
        if random.random() < RANDOM_HOLE_CIRCLE_PROBABILITY:
            hole = create_circle_hole(center, random.uniform(*CIRCLE_HOLE_RADIUS_RANGE))
        else:
            hole = create_pocket(
                center,
                random.uniform(*POCKET_SIZE_RANGE),
                random.uniform(*POCKET_SIZE_RANGE),
            )
        if (
            hole_elements_inside(hole, points)
            and hole_clear_of_existing(hole, accepted_holes)
        ):
            holes.extend(hole)
            accepted_holes.append(sample_hole_points(hole))

    return holes


SPLINE_DEGREE = 3

SPLINE_CTRLPT_COUNT_RANGE = (5, 20)

SPLINE_MODES = ["single_bend", "wave", "irregular"]

SPLINE_SINGLE_BEND_AMPLITUDE_RANGE = (0.1, 0.2)

SPLINE_WAVE_AMPLITUDE_RANGE = (0.1, 0.3)

SPLINE_IRREGULAR_AMPLITUDE_RANGE = (0.08, 0.30)

SPLINE_WAVE_COUNT_RANGE = (1, 3)

SPLINE_INTERIOR_T_RANGE = (0.1, 0.9)

SPLINE_INSERT_MIN_LINE_LENGTH = 0.5

SPLINE_INSERT_COUNTS = {
    "simple": (1, 2),
    "medium": (1, 4),
    "complex": (2, 6),
}

SPLINE_MAX_RATIO = 0.5


def generate_spline(start, end, source_element=None):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    chord = end - start
    length = np.linalg.norm(chord)

    if length < 1e-6:
        return generate_line(start.tolist(), end.tolist(), generation_stage='spline_replacement')

    tangent = chord / length
    normal = np.array([-tangent[1], tangent[0]])
    degree = SPLINE_DEGREE
    ctrlpt_count = random.randint(*SPLINE_CTRLPT_COUNT_RANGE)
    mode = random.choice(SPLINE_MODES)
    ctrlpts = [start]

    if mode == "single_bend":
        sign = random.choice([-1, 1])
        amplitude = length * random.uniform(*SPLINE_SINGLE_BEND_AMPLITUDE_RANGE)
    elif mode == "wave":
        amplitude = length * random.uniform(*SPLINE_WAVE_AMPLITUDE_RANGE)
        wave_count = random.randint(*SPLINE_WAVE_COUNT_RANGE)
        phase = random.uniform(0, 2 * np.pi)
    else:
        amplitude = length * random.uniform(*SPLINE_IRREGULAR_AMPLITUDE_RANGE)

    interior_ts = sorted((random.uniform(*SPLINE_INTERIOR_T_RANGE) for _ in range(ctrlpt_count - 2)))
    for t in interior_ts:
        envelope = np.sin(np.pi * t)

        if mode == "single_bend":
            offset = sign * amplitude * envelope
        elif mode == "wave":
            offset = amplitude * envelope * np.sin(2 * np.pi * wave_count * t + phase)
        else:
            offset = amplitude * envelope * random.uniform(-1, 1)

        point = start + chord * t + normal * offset
        ctrlpts.append(point)

    ctrlpts.append(end)
    spline = {
        "type": "spline",
        "role": "outer",
        "degree": degree,
        "start": start.tolist(),
        "end": end.tolist(),
        "ctrlpts": [point.tolist() for point in ctrlpts],
        "weights": [1.0] * len(ctrlpts),
        "knotvector_type": "open_uniform",
        "generation_stage": "spline_replacement",
        "color": GENERATION_STAGE_COLORS["spline_replacement"],
    }
    if source_element is not None:
        spline["replaced_generation_stage"] = source_element.get('generation_stage', 'base')
        spline["replaced_color"] = source_element.get('color', GENERATION_STAGE_COLORS['base'])
    return spline

def insert_local_splines(elements, complexity):
    line_indices = [
        i for i, element in enumerate(elements)
        if element["type"] == "line"
        and np.linalg.norm(np.array(element["end"]) - np.array(element["start"])) > SPLINE_INSERT_MIN_LINE_LENGTH
    ]
    if not line_indices:
        return elements

    max_count = max(1, int(np.floor(len(elements) * SPLINE_MAX_RATIO)))
    count_spec = SPLINE_INSERT_COUNTS[complexity]
    if isinstance(count_spec, tuple):
        low, high = count_spec
        spline_count = random.randint(low, max(low, min(max_count, high)))
    else:
        spline_count = count_spec
    spline_count = min(spline_count, len(line_indices))
    if spline_count == 0:
        return elements

    selected = set(random.sample(line_indices, spline_count))
    updated = []
    for i, element in enumerate(elements):
        if i not in selected:
            updated.append(element)
            continue

        updated.append(generate_spline(element['start'], element['end'], source_element=element))

    return updated

def enforce_spline_ratio_limit(elements, max_ratio=None):
    if max_ratio is None:
        max_ratio = SPLINE_MAX_RATIO
    total = len(elements)
    if total == 0:
        return elements

    max_count = max(1, int(np.floor(total * max_ratio)))
    updated = list(elements)
    spline_indices = [
        i
        for i, element in enumerate(updated)
        if element["type"] == "spline"
    ]

    if len(spline_indices) > max_count:
        for index in random.sample(spline_indices, len(spline_indices) - max_count):
            element = updated[index]
            restored_stage = element.get("replaced_generation_stage", "base")
            restored_color = element.get('replaced_color', GENERATION_STAGE_COLORS[restored_stage])
            updated[index] = generate_line(
                element["start"],
                element["end"],
                role=element.get("role", "outer"),
                generation_stage=restored_stage,
                color=restored_color,
            )
        spline_indices = [
            i
            for i, element in enumerate(updated)
            if element["type"] == "spline"
        ]

    return updated


MIN_VALID_SEGMENT_DISTANCE = 0.05


def point_lies_on_segment(point, start, end, eps=1e-7):
    point = np.array(point, dtype=float)
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < eps:
        return False
    offset = point - start
    cross = abs(float(segment[0] * offset[1] - segment[1] * offset[0]))
    if cross > eps * max(1.0, float(np.linalg.norm(segment))):
        return False
    projection = float(np.dot(offset, segment) / length_sq)
    return -eps <= projection <= 1.0 + eps

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (
                (xj - xi)
                * (y - yi)
                / (yj - yi)
                + xi
            )
            if x < x_cross:
                inside = not inside
        j = i
    return inside

def form_points_are_valid(points, eps=1e-8):
    if len(points) < 3:
        return False

    point_keys = [
        (round(point[0], 8), round(point[1], 8))
        for point in points
    ]
    if len(set(point_keys)) != len(point_keys):
        return False

    edge_keys = set()
    for index, start in enumerate(point_keys):
        end = point_keys[(index + 1) % len(point_keys)]
        if start == end:
            return False
        edge_key = tuple(sorted((start, end)))
        if edge_key in edge_keys:
            return False
        edge_keys.add(edge_key)

    def points_close_enough(a, b):
        return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)) < eps

    for i, start_a in enumerate(points):
        end_a = points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if abs(i - j) == 1 or (i == 0 and j == len(points) - 1):
                continue

            start_b = points[j]
            end_b = points[(j + 1) % len(points)]
            if (
                points_close_enough(start_a, start_b)
                or points_close_enough(start_a, end_b)
                or points_close_enough(end_a, start_b)
                or points_close_enough(end_a, end_b)
            ):
                return False
            if line_segments_overlap_or_intersect(start_a, end_a, start_b, end_b, eps):
                return False

    return True

def line_segments_overlap_or_intersect(a, b, c, d, eps=1e-8):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)
    ab = b - a
    cd = d - c
    denom = ab[0] * cd[1] - ab[1] * cd[0]
    ca = c - a

    if abs(denom) < eps:
        if abs(ca[0] * ab[1] - ca[1] * ab[0]) > eps:
            return False
        axis = 0 if abs(ab[0]) >= abs(ab[1]) else 1
        a0, b0 = sorted([a[axis], b[axis]])
        c0, d0 = sorted([c[axis], d[axis]])
        return max(a0, c0) < min(b0, d0) - eps

    t = (ca[0] * cd[1] - ca[1] * cd[0]) / denom
    u = (ca[0] * ab[1] - ca[1] * ab[0]) / denom
    return eps < t < 1 - eps and eps < u < 1 - eps

def distance_point_to_segment(point, start, end):
    point = np.array(point, dtype=float)
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    segment = end - start
    length_sq = np.dot(segment, segment)
    if length_sq < 1e-9:
        return np.linalg.norm(point - start)
    t = np.clip(np.dot(point - start, segment) / length_sq, 0, 1)
    projection = start + t * segment
    return np.linalg.norm(point - projection)

def distance_to_polygon_edges(point, polygon):
    return min(
        distance_point_to_segment(point, polygon[edge_index], polygon[(edge_index + 1) % len(polygon)])
        for edge_index in range(len(polygon))
    )

def hole_elements_inside(hole_elements, outer_points, clearance=None):
    if clearance is None:
        clearance = HOLE_OUTER_CLEARANCE
    for element in hole_elements:
        sampled = sample_element_points(element)
        for point in sampled[:: max(1, len(sampled) // 12)]:
            point = point.tolist()
            if not point_in_polygon(point, outer_points):
                return False
            if distance_to_polygon_edges(point, outer_points) < clearance:
                return False
    return True

def sampled_min_distance(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return float(np.min(distances))

def sample_hole_points(hole_elements):
    samples = []
    for element in hole_elements:
        sampled = sample_element_points(element)
        samples.append(sampled[:: max(1, len(sampled) // 16)])
    if not samples:
        return np.empty((0, 2))
    return np.vstack(samples)

def hole_clear_of_existing(hole_elements, accepted_holes, clearance=None):
    if clearance is None:
        clearance = HOLE_EXISTING_CLEARANCE
    current = sample_hole_points(hole_elements)
    for accepted in accepted_holes:
        if sampled_min_distance(current, accepted) < clearance:
            return False
    return True

def sample_element_points(element):
    if element["type"] == "line":
        return np.array([element["start"], element["end"]], dtype=float)

    if element["type"] == "arc":
        pts = sample_arc_points(
            np.array(element["start"], dtype=float),
            np.array(element["mid"], dtype=float),
            np.array(element["end"], dtype=float),
            sample_size=24,
        )
        if pts is not None:
            return pts
        return np.array([element["start"], element["end"]], dtype=float)

    if element["type"] == "spline":
        curve = BSpline.Curve()
        curve.degree = 3
        curve.ctrlpts = element["ctrlpts"]
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.sample_size = 32
        return np.array(curve.evalpts, dtype=float)

    if element["type"] == "circle":
        center = np.array(element["center"], dtype=float)
        radius = element["radius"]
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=True)
        return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))

    return np.empty((0, 2))

def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]

def points_close(a, b, eps=1e-6):
    return np.linalg.norm(np.array(a) - np.array(b)) < eps

def sampled_segments_intersect(a,b,c,d,eps=1e-8,):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    if (
        points_close(a, c)
        or points_close(a, d)
        or points_close(b, c)
        or points_close(b, d)
    ):
        return False

    ab = b - a
    cd = d - c
    denom = cross_2d(ab, cd)
    ca = c - a

    if abs(denom) < eps:
        if abs(cross_2d(ca, ab)) > eps:
            return False
        axis = 0 if abs(ab[0]) >= abs(ab[1]) else 1
        a0, b0 = sorted([a[axis], b[axis]])
        c0, d0 = sorted([c[axis], d[axis]])
        return (
            max(a0, c0)
            < min(b0, d0) - eps
        )

    t = cross_2d(ca, cd) / denom
    u = cross_2d(ca, ab) / denom
    return (
        eps < t < 1 - eps
        and eps < u < 1 - eps
    )

def distance_between_sampled_segments(a, b, c, d):
    return min(
        distance_point_to_segment(a, c, d),
        distance_point_to_segment(b, c, d),
        distance_point_to_segment(c, a, b),
        distance_point_to_segment(d, a, b),
    )

def collect_sampled_segments(elements):
    sampled_segments = []
    for element_index, element in enumerate(elements):
        points = sample_element_points(element)
        for segment_index in range(len(points) - 1):
            if not points_close(points[segment_index], points[segment_index + 1]):
                sampled_segments.append(
                    (
                        element_index,
                        segment_index,
                        points[segment_index],
                        points[segment_index + 1],
                    )
                )
    return sampled_segments

def should_skip_sampled_segment_pair(
    first_element,
    second_element,
    a,
    b,
    c,
    d,
    element_count,
):
    if first_element == second_element:
        return True

    if abs(first_element - second_element) == 1:
        if (
            points_close(b, c)
            or points_close(a, d)
        ):
            return True

    if first_element == 0 and second_element == element_count - 1:
        if (
            points_close(a, d)
            or points_close(b, c)
        ):
            return True

    return False

def has_self_intersection(elements):
    sampled_segments = collect_sampled_segments(elements)

    for i, first in enumerate(sampled_segments):
        for second in sampled_segments[i + 1:]:
            first_element, _, a, b = first
            second_element, _, c, d = second

            if should_skip_sampled_segment_pair(first_element, second_element, a, b, c, d, len(elements)):
                continue

            if sampled_segments_intersect(a, b, c, d):
                return True

    return False

def should_skip_clearance_segment_pair(first_element, second_element, elements):
    if first_element == second_element:
        return True

    first_element_data = elements[first_element]
    second_element_data = elements[second_element]
    first_endpoints = [
        first_element_data[key]
        for key in ("start", "end")
        if key in first_element_data
    ]
    second_endpoints = [
        second_element_data[key]
        for key in ("start", "end")
        if key in second_element_data
    ]
    if any(
        points_close(first_point, second_point)
        for first_point in first_endpoints
        for second_point in second_endpoints
    ):
        return True

    if abs(first_element - second_element) == 1:
        previous_element = elements[min(first_element, second_element)]
        next_element = elements[max(first_element, second_element)]
        if (
            "end" in previous_element
            and "start" in next_element
            and points_close(previous_element["end"], next_element["start"])
        ):
            return True

    if first_element == 0 and second_element == len(elements) - 1:
        first_element_data = elements[first_element]
        last_element = elements[second_element]
        if (
            "start" in first_element_data
            and "end" in last_element
            and points_close(first_element_data["start"], last_element["end"])
        ):
            return True

    return False

def has_minimum_segment_distance(elements, clearance=MIN_VALID_SEGMENT_DISTANCE):
    sampled_segments = collect_sampled_segments(elements)

    for i, first in enumerate(sampled_segments):
        for second in sampled_segments[i + 1:]:
            first_element, _, a, b = first
            second_element, _, c, d = second

            if should_skip_clearance_segment_pair(first_element, second_element, elements):
                continue

            if distance_between_sampled_segments(a, b, c, d) < clearance:
                return False

    return True

def circle_from_three_points(a, b, c):
    d = b - a
    e = c - a
    det = np.linalg.det([d, e])
    if abs(det) < 1e-6:
        return None, None
    b = np.dot(d, d) / 2
    c = np.dot(e, e) / 2
    offset = np.array([b * e[1] - c * d[1], c * d[0] - b * e[0]]) / det
    center = a + offset
    return center, np.linalg.norm(a - center)

def angle_between_ccw(theta1, theta_mid, theta2):
    theta1 = theta1 % (2 * np.pi)
    theta_mid = theta_mid % (2 * np.pi)
    theta2 = theta2 % (2 * np.pi)
    if theta2 < theta1:
        theta2 += 2 * np.pi
    if theta_mid < theta1:
        theta_mid += 2 * np.pi
    return theta1 <= theta_mid <= theta2

def sample_arc_points(p1, p2, p3, sample_size=80):
    center, radius = circle_from_three_points(p1, p2, p3)
    if center is None:
        return None

    theta1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    theta_mid = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    theta2 = np.arctan2(p3[1] - center[1], p3[0] - center[0])

    if angle_between_ccw(theta1, theta_mid, theta2):
        end = theta2
        if end < theta1:
            end += 2 * np.pi
        angles = np.linspace(theta1, end, sample_size)
    else:
        end = theta2
        if end > theta1:
            end -= 2 * np.pi
        angles = np.linspace(theta1, end, sample_size)

    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


ELEMENT_ROTATION_CHOICES = [0, 90, 180, 270]

MIRROR_X_CHOICES = [False, True]

MIRROR_Y_CHOICES = [False, True]


def transform_point(point, rotation, mirror_x, mirror_y):
    x, y = point
    if rotation == 90:
        x, y = -y, x
    elif rotation == 180:
        x, y = -x, -y
    elif rotation == 270:
        x, y = y, -x

    if mirror_x:
        y = -y
    if mirror_y:
        x = -x

    return [float(x), float(y)]

def orient_elements(elements):
    rotation = random.choice(ELEMENT_ROTATION_CHOICES)
    mirror_x = random.choice(MIRROR_X_CHOICES)
    mirror_y = random.choice(MIRROR_Y_CHOICES)
    transformed = []

    for element in elements:
        new_element = {**element}
        if "start" in new_element:
            new_element["start"] = transform_point(new_element['start'], rotation, mirror_x, mirror_y)
        if "mid" in new_element:
            new_element["mid"] = transform_point(new_element['mid'], rotation, mirror_x, mirror_y)
        if "end" in new_element:
            new_element["end"] = transform_point(new_element['end'], rotation, mirror_x, mirror_y)
        if "ctrlpts" in new_element:
            new_element["ctrlpts"] = [
                transform_point(point, rotation, mirror_x, mirror_y)
                for point in new_element["ctrlpts"]
            ]
        if "center" in new_element:
            new_element["center"] = transform_point(new_element['center'], rotation, mirror_x, mirror_y)
        transformed.append(new_element)

    return transformed

def reverse_element_direction(element):
    reversed_element = {**element}
    if "start" in reversed_element and "end" in reversed_element:
        reversed_element["start"], reversed_element["end"] = reversed_element["end"], reversed_element["start"]
    if reversed_element["type"] == "spline":
        reversed_element["ctrlpts"] = list(reversed(reversed_element["ctrlpts"]))
    return reversed_element

def chain_polygon_area(elements):
    points = []
    for element in elements:
        sampled = sample_element_points(element)
        if len(sampled) == 0:
            continue
        for point in sampled[:-1]:
            add_point(points, point[0], point[1])
    if len(points) < 3:
        return 0.0
    return polygon_area(points)

def force_ccw_element_chain(elements):
    if chain_polygon_area(elements) >= 0:
        return elements
    return [
        reverse_element_direction(element)
        for element in reversed(elements)
    ]

def force_ccw_element_order(elements):
    result = []
    chain = []

    def is_connected(previous_element, next_element, eps=1e-6):
        if "end" not in previous_element or "start" not in next_element:
            return False
        return (
            np.linalg.norm(
                np.array(previous_element["end"], dtype=float)
                - np.array(next_element["start"], dtype=float)
            )
            < eps
        )

    def flush_chain():
        nonlocal chain
        if chain:
            result.extend(force_ccw_element_chain(chain))
            chain = []

    for element in elements:
        if element["type"] == "circle":
            flush_chain()
            result.append(element)
            continue

        if chain and not is_connected(chain[-1], element):
            flush_chain()
        chain.append(element)

    flush_chain()
    return result

def transform_elements(elements):
    return force_ccw_element_order(orient_elements(elements))

def elements_to_polygon(elements):
    points = []
    for element in elements:
        if element.get("role") in {"hole", "pocket"}:
            continue
        pts = sample_element_points(element)
        if len(pts) == 0:
            continue
        for point in pts[:-1]:
            add_point(points, point[0], point[1])
    return normalize_ccw(points)


GENERATION_STAGE_COLORS = {
    "base": "#000000",
    "edge_feature": "#FF0000",
    "corner_replacement": "#0000FF",
    "arc_replacement": "#9423D1",
    "hole_or_pocket": "#46D908",
    "spline_replacement": "#FF9C2B",
}

ELEMENT_LINE_WIDTH = 1.0

ELEMENT_TYPE_IDS = {
    "line": "L",
    "circle": "C",
    "arc": "A",
    "spline": "S",
}


def plot_line(ax, element):
    ax.plot(
        [element["start"][0], element["end"][0]],
        [element["start"][1], element["end"][1]],
        color=element["color"],
        linewidth=ELEMENT_LINE_WIDTH,
    )

def plot_spline(ax, element):
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = element["ctrlpts"]
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.sample_size = 1000
    pts = np.array(curve.evalpts)
    ax.plot(pts[:, 0], pts[:, 1], color=element['color'], linewidth=ELEMENT_LINE_WIDTH)

def plot_arc(ax, element):
    p1, p2, p3 = map(np.array, [element['start'], element['mid'], element['end']])
    pts = sample_arc_points(p1, p2, p3)
    if pts is None:
        return
    ax.plot(pts[:, 0], pts[:, 1], color=element['color'], linewidth=ELEMENT_LINE_WIDTH)

def plot_circle(ax, element):
    center = np.array(element["center"], dtype=float)
    radius = element["radius"]
    angles = np.linspace(0, 2 * np.pi, 120)
    pts = np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
    ax.plot(pts[:, 0], pts[:, 1], color=element['color'], linewidth=ELEMENT_LINE_WIDTH)

def plot_elements(elements):
    plotters = {"line": plot_line, "arc": plot_arc, "spline": plot_spline, "circle": plot_circle}
    fig, ax = plt.subplots()
    for element in elements:
        plotters[element["type"]](ax, element)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.1)
    fig.tight_layout()
    plt.show()

def elements_are_connected(previous_element, next_element, eps=1e-6):
    if "end" not in previous_element or "start" not in next_element:
        return False
    return (
        np.linalg.norm(np.array(previous_element['end'], dtype=float) - np.array(next_element['start'], dtype=float))
        < eps
    )

def element_loops(elements):
    loops = []
    current_loop = []

    def flush_loop():
        nonlocal current_loop
        if current_loop:
            loops.append(current_loop)
            current_loop = []

    for element in elements:
        if element["type"] == "circle":
            flush_loop()
            loops.append([element])
            continue

        if (
            current_loop
            and current_loop[-1].get("role") in {"hole", "pocket"}
            and element.get("role") not in {"hole", "pocket"}
        ):
            flush_loop()

        if current_loop and not elements_are_connected(current_loop[-1], element):
            flush_loop()

        current_loop.append(element)

    flush_loop()
    return loops

def element_print_token(element):
    element_type = element["type"]
    values = [ELEMENT_TYPE_IDS[element_type]]

    if element_type == "line":
        values.extend(element["end"])
    elif element_type == "circle":
        values.extend(element["center"])
        values.append(float(element["radius"]))
    elif element_type == "arc":
        values.extend(element["mid"])
        values.extend(element["end"])
    elif element_type == "spline":
        for point in element["ctrlpts"][1:]:
            values.extend(point)
    else:
        raise ValueError(f"unsupported element type: {element_type}")

    return values

def round_printable_numbers(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    if isinstance(value, list):
        return [round_printable_numbers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(round_printable_numbers(item) for item in value)
    if isinstance(value, dict):
        return {
            key: round_printable_numbers(item)
            for key, item in value.items()
        }
    return value

def print_elements(elements):
    for loop in element_loops(elements):
        print("SOL")
        for element in loop:
            print(round_printable_numbers(element_print_token(element)))
        print("EOS")


if __name__ == "__main__":
    elements = generate_mixed_curve()
    print_elements(elements)
    plot_elements(elements)
