# router/archive

Router code of experiments that were dropped. Nothing in the live router or the eval runner imports these files; they are kept so the old results can be traced to their code.

| File | Experiment | Result | Removed |
|---|---|---|---|
| `cascade.py` | first model answers, escalate to Mamay-12B (or retry Mamay-4B on social alignment) when the answer looks unsure | `week_6_cascade_micro/`: 0.842 on v4, 1.56% of items escalated; no gain | 2 Nov 2026 |
| `ensemble.py` | majority vote of Lapa, Mamay-4B and Aya on alignment digits and ZNO letters | `week_7_ensemble/`: 0.843 on v4; no gain | 2 Nov 2026 |
| `litellm/` | LiteLLM proxy configs; the router now calls the vLLM ports directly through `router/backends.py` | — | 2 Nov 2026 |

The runner modes `router_cascade`, `router_cascade_micro` and `router_ensemble` were removed from `scripts/run_small_router_cluster.py` at the same time. To rerun them, check out a commit from before 2 Nov 2026.
