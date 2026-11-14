import math

import pytest

from problem import N_POINTS, InvalidSolution, verify


def grid_points():
    # 4x3 grid on [0,1]^2: min distance = 1/3
    return [[i / 3, j / 2] for i in range(4) for j in range(3)]


def test_verify_known_layout():
    assert verify({"points": grid_points()}) == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad", [
    None,
    {"points": [[0.5, 0.5]] * (N_POINTS - 1)},
    {"points": [[0.5, 0.5]] * (N_POINTS - 1) + [[1.5, 0.5]]},
    {"points": [[0.5, 0.5]] * (N_POINTS - 1) + [[math.nan, 0.5]]},
])
def test_verify_rejects_invalid(bad):
    with pytest.raises(InvalidSolution):
        verify(bad)


def test_baseline_strategy_returns_valid_solution():
    from strategy.search import search
    sol = search({"method": "random", "n_samples": 50}, seed=0, budget_s=1)
    assert 0 < verify(sol) < 0.5
