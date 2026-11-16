"""MLflow logging helpers: run parameters from system names and metrics from score summaries."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("run_composite_router")
scores = _load("log_scores_mlflow")


def test_system_params():
    assert runner.system_params("mamay27") == {"planner": "none", "planner_model": "-", "router": "none (one model)"}
    assert runner.system_params("router_mlp_nodes_only")["router"] == "mlp_nodes_only"
    assert runner.system_params("oracle+mlp_nodes_only") == {"planner": "gold plan", "planner_model": "-",
                                                           "router": "mlp_nodes_only"}
    assert runner.system_params("hgpt_mamay27+mlp_nodes_only") == {"planner": "hgpt", "planner_model": "mamay27",
                                                                 "router": "mlp_nodes_only"}
    assert runner.system_params("hybrid_gated") == {"planner": "hybrid_gated", "planner_model": "mamay4",
                                                  "router": "v2"}
    assert runner.system_params("hybrid_gated_lapa")["planner_model"] == "lapa"
    assert runner.system_params("hybrid_mamay12")["planner_model"] == "mamay12"


def test_flat_metrics():
    summary = {"n": 92, "repeats": 1, "final_score_mean": 0.7, "plan_valid_rate": None, "workflow_errors": {"x": 1},
               "by_family": {"chain": {"final_score": 0.5}}}
    assert scores.flat_metrics(summary) == {"final_score_mean": 0.7, "family.chain.final_score": 0.5}
