import random

from scripts.analysis.v7_compare import CLAIM_BUCKETS, compare, verdict


def _suite(n: int = 40) -> dict[str, str]:
    return {f"{bucket}-{i}": bucket for bucket in CLAIM_BUCKETS for i in range(n)}


def test_identical_systems_are_not_distinguishable():
    bucket_of = _suite()
    scores = {i: float(k % 2) for k, i in enumerate(bucket_of)}
    rows = compare(scores, dict(scores), bucket_of, seed=42, n_boot=200)
    assert rows["macro4"]["diff"] == 0 and rows["macro4"]["verdict"] == "not distinguishable"


def test_clear_gap_in_one_bucket_is_reported_there_and_in_the_macro():
    bucket_of = _suite()
    rng = random.Random(1)
    b = {i: float(rng.random() < 0.5) for i in bucket_of}
    a = {i: (1.0 if bucket_of[i] == "instruct" else b[i]) for i in bucket_of}
    rows = compare(a, b, bucket_of, seed=42, n_boot=400)
    assert rows["instruct"]["verdict"] == "A higher"
    assert rows["knowledge"]["verdict"] == "not distinguishable"
    assert rows["macro4"]["verdict"] == "A higher"
    assert rows["macro4"]["n"] == len(bucket_of)


def test_verdict_needs_an_interval_that_excludes_zero():
    assert verdict([-0.01, 0.02]) == "not distinguishable"
    assert verdict([0.0, 0.02]) == "not distinguishable"
    assert verdict([-0.05, -0.01]) == "A lower"
