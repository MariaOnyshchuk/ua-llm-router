from __future__ import annotations

import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from router.intent_rules import route_intent
from scripts.curate_discriminative_suite import difficulty_balanced, proxy_stats
from scripts.extract_ua_leaderboard import (
    rows_from_arc,
    rows_from_belebele,
    rows_from_ifeval,
    rows_from_mmlu,
    rows_from_wmt,
)
from scripts.ifeval_check import score_ifeval
from scripts.mcq_format import letter_from_answer
from scripts.sample_size_ci import accuracy_ci, binomial_n
from scripts.score_results import score_alignment, score_knowledge, score_row


class SampleSizeTests(unittest.TestCase):
    def test_n_star_around_683(self) -> None:
        self.assertEqual(binomial_n(p=0.8, epsilon=0.03, z=1.96), 683)

    def test_ci_narrows_with_n(self) -> None:
        lo_s, hi_s = accuracy_ci(0.8, 50)
        lo_l, hi_l = accuracy_ci(0.8, 700)
        self.assertLess(hi_l - lo_l, hi_s - lo_s)


class McqAndIfevalTests(unittest.TestCase):
    def test_letter_from_index(self) -> None:
        self.assertEqual(letter_from_answer(1, 4), "B")
        self.assertEqual(letter_from_answer(0, 4), "A")
        self.assertEqual(letter_from_answer(1, 4, one_indexed=True), "A")
        self.assertEqual(letter_from_answer(2, 4, one_indexed=True), "B")
        self.assertEqual(letter_from_answer("C", 4), "C")
        self.assertEqual(letter_from_answer("Г", 4), "D")

    def test_knowledge_accepts_latin_or_cyrillic(self) -> None:
        self.assertEqual(score_knowledge("Відповідь: B", "B")["score"], 1.0)
        self.assertEqual(score_knowledge("Б", "Б")["score"], 1.0)
        self.assertEqual(score_knowledge("C", "A")["score"], 0.0)

    def test_alignment_ignores_echoed_scale_prefix(self) -> None:
        echoed = "0 - погано, 1 - нормально, 2 - добре\nВідповідь: 2"
        got = score_alignment(echoed, "2")
        self.assertEqual(got["score"], 1.0)
        self.assertIn("method=anchored", got["detail"])
        first_digit_would_fail = score_alignment("0 - погано, 1 - нормально, 2 - добре\n2", "2")
        self.assertEqual(first_digit_would_fail["score"], 1.0)
        self.assertIn("method=last_digit", first_digit_would_fail["detail"])
        self.assertEqual(score_alignment("1", "1")["score"], 1.0)

    def test_ifeval_word_count_and_no_comma(self) -> None:
        text = "one two three four five six seven eight nine ten"
        got = score_ifeval(
            text,
            ["length_constraints:number_words", "punctuation:no_comma"],
            [{"num_words": 8, "relation": "at least"}, {}],
        )
        self.assertEqual(got["score"], 1.0)

    def test_score_row_ifeval_item(self) -> None:
        row = {
            "id": "ifeval-0001",
            "bucket": "instruct",
            "prompt": "Say hello",
            "content": "HELLO WORLD",
            "http_status": 200,
            "notes": "source=ifeval_ukr score_mode=ifeval",
            "ifeval_instruction_id_list": ["change_case:english_capital"],
            "ifeval_kwargs": [{}],
        }
        self.assertEqual(score_row(row)["score"], 1.0)


class ExtractorShapeTests(unittest.TestCase):
    def test_ifeval_rows(self) -> None:
        rows = rows_from_ifeval(
            [
                {
                    "prompt": "Write JSON",
                    "instruction_id_list": ["detectable_format:json_format"],
                    "kwargs": [{}],
                }
            ]
        )
        self.assertEqual(rows[0]["bucket"], "instruct")
        self.assertIn("ifeval_instruction_id_list", rows[0])

    def test_belebele_and_mmlu_and_arc(self) -> None:
        bel = rows_from_belebele(
            [
                {
                    "question": "Хто?",
                    "flores_passage": "Текст.",
                    "mc_answer1": "Аня",
                    "mc_answer2": "Богдан",
                    "mc_answer3": "Віра",
                    "mc_answer4": "Данило",
                    "correct_answer_num": 2,
                }
            ]
        )
        self.assertEqual(bel[0]["reference"], "B")
        self.assertEqual(bel[0]["bucket"], "knowledge")
        mmlu = rows_from_mmlu(
            [
                {
                    "question": "2+2?",
                    "choices": ["1", "4", "3", "0"],
                    "answer": 1,
                    "subject": "math",
                }
            ]
        )
        self.assertEqual(mmlu[0]["reference"], "B")
        arc = rows_from_arc(
            [{"question": "Q?", "choices": {"text": ["x", "y"]}, "answerKey": "B"}]
        )
        self.assertEqual(arc[0]["reference"], "B")

    def test_wmt_both_directions(self) -> None:
        rows = rows_from_wmt([{"translation": {"en": "Hi", "uk": "Привіт"}, "id": 1}])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["bucket"] == "translate" for r in rows))

    def test_mmlu_cap_stratified(self) -> None:
        records = []
        for subj in ("a", "b"):
            for i in range(10):
                records.append(
                    {
                        "question": f"{subj}{i}",
                        "choices": ["1", "2"],
                        "answer": 0,
                        "subject": subj,
                        "id": f"{subj}-{i}",
                    }
                )
        rows = rows_from_mmlu(records, cap=6, seed=0)
        self.assertEqual(len(rows), 6)


class CuratorTests(unittest.TestCase):
    def test_variance_and_terciles(self) -> None:
        var, mean = proxy_stats({"a": 1.0, "b": 0.0, "c": 0.0})
        self.assertGreater(var, 0.0)
        self.assertAlmostEqual(mean, 1 / 3)
        rows = [{"id": str(i), "_mean": i / 9} for i in range(9)]
        picked = difficulty_balanced(rows, 3, __import__("random").Random(0))
        self.assertEqual(len(picked), 3)


class LearnedRouterTests(unittest.TestCase):
    def test_v2_unchanged_default(self) -> None:
        d = route_intent("Переклади з англійської: hello", profile="v2")
        self.assertEqual(d.model, "aya")
        self.assertEqual(d.intent, "translate")

    def test_knn_profile_with_hash_artifacts(self) -> None:
        from sklearn.neighbors import KNeighborsClassifier

        from router import learned as learned_mod
        from router.learned import HashEncoder

        encoder = HashEncoder(32)
        prompts = [
            "Переклади з англійської українською. Hello",
            "Дай відповідь на тестове завдання ЗНО. Варіанти:",
            "Напиши функцію python add",
        ]
        labels = ["aya", "lapa", "mamay4"]
        X = encoder.encode(prompts)
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X, labels)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "knn"
            dest.mkdir()
            (dest / "model.pkl").write_bytes(pickle.dumps(model))
            (dest / "config.json").write_text(
                json.dumps({"encoder": "hash", "encoder_dim": 32, "default": "aya"})
            )
            with patch.object(learned_mod, "ARTIFACT_ROOT", Path(tmp)):
                d = route_intent(prompts[0], profile="knn")
            self.assertEqual(d.model, "aya")
            self.assertIn("learned", d.intent)

    def test_knn_missing_artifacts_falls_back(self) -> None:
        from router import learned as learned_mod

        with patch.object(learned_mod, "ARTIFACT_ROOT", Path("/tmp/diploma-no-artifacts")):
            d = route_intent("Привіт, як справи?", profile="knn")
        self.assertEqual(d.model, "aya")
        self.assertIn("fallback", d.reason)


class ComparisonTests(unittest.TestCase):
    def test_oracle_and_best_single(self) -> None:
        from scripts.build_router_comparison import means, oracle_rows

        solos = {
            "mamay4": {
                "1": {"id": "1", "bucket": "knowledge", "score": 0.0},
                "2": {"id": "2", "bucket": "translate", "score": 1.0},
            },
            "lapa": {
                "1": {"id": "1", "bucket": "knowledge", "score": 1.0},
                "2": {"id": "2", "bucket": "translate", "score": 0.0},
            },
        }
        orows = oracle_rows(solos)
        stats = means(orows, ("knowledge", "translate"))
        self.assertEqual(stats["quality_micro"], 1.0)


class EvalRunnerFailureTests(unittest.TestCase):
    def test_backend_failure_stops_run_and_keeps_partial_out_of_jsonl_glob(self) -> None:
        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                pass

        fake_httpx = types.SimpleNamespace(Client=FakeClient)
        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            from scripts.run_small_router_cluster import run_system

        class FakeSampler:
            def __init__(self, interval_s: float) -> None:
                self.samples = [[]]
                self.stopped = False

            def start(self) -> None:
                pass

            def stop(self) -> dict:
                self.stopped = True
                return {"n_samples": 0}

        failed = {
            "ok": False,
            "content": "",
            "latency_ms": 1.0,
            "gpu_seconds": 0.001,
            "http_status": 500,
            "error": {"message": "engine stopped"},
        }

        def pick_backend(item: dict) -> tuple[str, str, str, dict]:
            return "mamay12", "http://localhost/v1", "checkpoint", {"model": "mamay12"}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("scripts.run_small_router_cluster.VramSampler", FakeSampler),
                patch("scripts.run_small_router_cluster.chat", return_value=failed),
            ):
                with self.assertRaisesRegex(RuntimeError, "eval backend failed"):
                    run_system(
                        "mamay12",
                        pick_backend,
                        [{"id": "item-1", "bucket": "chat", "prompt": "hello"}],
                        "test",
                        out_dir=Path(tmp),
                    )

            self.assertEqual(list(Path(tmp).glob("*.jsonl")), [])
            partials = list(Path(tmp).glob("*.jsonl.partial"))
            self.assertEqual(len(partials), 1)
            row = json.loads(partials[0].read_text(encoding="utf-8"))
            self.assertFalse(row["ok"])


if __name__ == "__main__":
    unittest.main()
