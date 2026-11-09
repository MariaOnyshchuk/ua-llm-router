"""One-line "why" comment written at the top of every benchmark file.

File names follow <unit>_<origin>_<number of tasks>[_<split>]:
  unit    tasks (one request, one skill) | graphs (a request made of several nodes) | nodes (one step)
  origin  synthetic (written by us) | public (from existing datasets) | mixed
Builders call `why_line(path.name)`; files without a builder got the same line once, by hand.
The full old -> new name table is in docs/benchmarks.md.
"""

from __future__ import annotations

WHY = {
    "graphs_mixed_100_dev.jsonl": "100 graph tasks (250 nodes), all dev; written by us + STaD, ZNO, Belebele "
                                  "(was mixed_ua_composite_graph100); math steps intent math, not chat, since 9 Nov 2026",
    "graphs_mixed_100_test.jsonl": "same build as graphs_mixed_100_dev, held out, frozen; run once "
                                   "(was mixed_ua_composite_test100); math steps intent math, not chat, since 9 Nov 2026 (label only)",
    "nodes_mixed_600_train.jsonl": "600 single nodes, training labels for the learned routers; "
                                   "545 written + 40 ZNO + 15 UAlign (was mixed_ua_nodes_train); label math, not chat, since 9 Nov 2026",
    "nodes_mixed_600_test.jsonl": "600 single nodes, held out for testing the routers; same build as nodes_mixed_600_train; "
                                  "label math, not chat, since 9 Nov 2026 (label only)",
    "tasks_public_4401.jsonl": "4401 single-skill tasks, claim suite with dev 1542 / test 2859 inside; "
                               "all from public datasets except 32 hand-written chat items (was mixed_ua_v7)",
    "tasks_mixed_192.jsonl": "192 single-skill tasks, development suite for wiring checks; "
                             "hand-written + ZNO, FLORES, UAlign (was mixed_ua_v4_balanced)",
    "intent_requests_72.jsonl": "72 hand-written everyday requests, diagnostic for intent detection; "
                                "not a quality benchmark (was intent_natural_v1)",
    "graphs_agentcoma_200.jsonl": "AgentCoMa-UK, 200 commonsense-then-arithmetic tasks, dev 50 / test 150; "
                                  "gated source, local only (was agentcoma_uk_200_composite)",
    "graphs_agentcoma_50_dev.jsonl": "AgentCoMa-UK dev 50 in composite form; gated source, local only "
                                     "(was agentcoma_uk_50_composite)",
    "graphs_agentcoma_200_source.jsonl": "AgentCoMa-UK, 200 translated source items, dev 50 + test 150; "
                                         "gated source, local only (was agentcoma_uk_200_merged)",
    "graphs_agentcoma_50_dev_source.jsonl": "AgentCoMa-UK dev 50 translated source items; gated source, local only "
                                            "(was agentcoma_uk_50items_merged)",
    "graphs_agentcoma_150_test_source.jsonl": "AgentCoMa-UK test 150 translated source items; gated source, local only "
                                              "(was agentcoma_uk_150_test_translated)",
    # benchmarks/archive/: names kept so that old results still point at them
    "mixed_ua_v0.jsonl": "archived 1 Nov 2026: historical, used 25 Jul 2026",
    "mixed_ua_v1.jsonl": "archived 1 Nov 2026: historical, used 26-27 Jul 2026",
    "mixed_ua_v2.jsonl": "archived 1 Nov 2026: historical, used 27-29 Jul 2026",
    "mixed_ua_v3.jsonl": "archived 1 Nov 2026: historical, used 29 Jul - 3 Aug 2026",
    "mixed_ua_v5.jsonl": "archived 1 Nov 2026: historical, used 21 Aug 2026; not used for claims",
    "mixed_ua_v6.jsonl": "archived 1 Nov 2026: withdrawn 12 Oct 2026, its selection step was inverted; do not cite",
    "mixed_ua_v6_screen.jsonl": "archived 1 Nov 2026: the screening pool that tasks_public_4401 is split from",
    "external_ua_v2.jsonl": "archived 1 Nov 2026: historical, used 27 Jul 2026",
    "mixed_ua_composite_v1.jsonl": "archived 1 Nov 2026: historical composite suite (week 10), replaced by graphs_mixed_100_*",
    "mixed_ua_composite_v2.jsonl": "archived 1 Nov 2026: historical composite suite (week 12), replaced by graphs_mixed_100_*",
    "mixed_ua_composite_probe100.jsonl": "archived 1 Nov 2026: retired 28 Oct 2026, linear chains from 5 templates only",
}


def why_line(name: str) -> str:
    """'# why: ...\\n' for a known benchmark file name, else ''."""
    why = WHY.get(name)
    return f"# why: {why}\n" if why else ""
