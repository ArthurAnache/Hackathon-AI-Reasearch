import numpy as np
import itertools
from scipy.spatial import ConvexHull

def evaluate(data):
    points = np.array(data["points"], dtype=np.float64)
    if points.shape != (14, 2):
        return -float("inf")
    if not np.isfinite(points).all():
        return -float("inf")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins

    if np.any(spans <= 0):
        return -float("inf")

    points = (points - mins) / spans

    def tri_area(p1, p2, p3):
        return abs(
            p1[0] * (p2[1] - p3[1])
            + p2[0] * (p3[1] - p1[1])
            + p3[0] * (p1[1] - p2[1])
        ) / 2.0

    try:
        hull_area = ConvexHull(points).volume
    except Exception:
        return -float("inf")

    if not np.isfinite(hull_area) or hull_area < 1e-12:
        return -float("inf")

    min_area = min(
        tri_area(points[i], points[j], points[k])
        for i, j, k in itertools.combinations(range(14), 3)
    )

    return float(min_area / hull_area)