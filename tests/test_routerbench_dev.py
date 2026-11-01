import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "routerbench_dev", Path(__file__).resolve().parents[1] / "scripts/analysis/routerbench_dev.py"
)
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)


def test_ndch_drops_points_under_the_hull_and_falling_points():
    # RouterBench Figure 2 (middle): D lies under the hull, E costs more and scores less.
    pts = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.5, 0.3), (0.9, 0.6)]
    assert rb.ndch(pts) == [(0.2, 0.2), (0.8, 0.8)]


def test_aiq_of_a_line_is_its_mean_and_flat_after_the_last_point():
    assert abs(rb.aiq([(0.0, 0.0), (1.0, 1.0)], 0.0, 1.0) - 0.5) < 1e-6
    assert abs(rb.aiq([(0.0, 0.4), (0.5, 0.6)], 0.0, 1.0) - 0.55) < 1e-6


def test_bootstrap_macro_zero_difference():
    q = np.array([1.0, 0.0] * 8)
    buckets = np.array([b for b in rb.BUCKETS for _ in range(4)])
    out = rb.bootstrap_macro(q, q, buckets, n_boot=200, seed=1)
    assert out["diff"] == 0 and out["ci95"] == [0.0, 0.0] and not out["distinguishable"]
