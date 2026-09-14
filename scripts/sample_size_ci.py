#!/usr/bin/env python3
"""Binomial sample-size and Wald CI helpers for accuracy-like scores."""

from __future__ import annotations

import argparse
import math


def binomial_n(*, p: float = 0.8, epsilon: float = 0.03, z: float = 1.96) -> int:
    """N ≥ (z · s / ε)² with s = √[p(1-p)]."""
    p = min(1.0, max(0.0, p))
    s = math.sqrt(p * (1.0 - p))
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    return math.ceil((z * s / epsilon) ** 2)


def accuracy_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = min(1.0, max(0.0, p))
    se = math.sqrt(p * (1.0 - p) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    lo, hi = accuracy_ci(p, n, z=z)
    return 0.5 * (hi - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--z", type=float, default=1.96)
    parser.add_argument("--n", type=int, default=0, help="If set, print CI around --p")
    args = parser.parse_args()
    n_star = binomial_n(p=args.p, epsilon=args.epsilon, z=args.z)
    payload = {
        "p": args.p,
        "epsilon": args.epsilon,
        "z": args.z,
        "n_star": n_star,
        "formula": "N >= (z * sqrt(p*(1-p)) / epsilon)^2",
    }
    if args.n:
        lo, hi = accuracy_ci(args.p, args.n, z=args.z)
        payload.update({"n": args.n, "ci95": [round(lo, 4), round(hi, 4)]})
    print(payload)


if __name__ == "__main__":
    main()
