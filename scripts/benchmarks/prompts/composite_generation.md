# Prompts for the composite dataset (100 requests)

Prompts that a teacher model uses to draft and check the composite dataset. The item
format, classes and rules come from the vault notes «Datasets for composite tasks —
findings, rules, types» (20 Oct 2026) and «Composite dataset — draft sample of 10»
(14 Oct 2026). The checking steps follow STaD (An et al., 2026, arXiv 2604.18177,
appendix H), adapted to our graph format and to Ukrainian.

Written 27 Oct 2026. No item has been generated with these prompts yet.

## Pipeline

| Stage | Prompt | Who runs it | Keeps the item when |
|---|---|---|---|
| 0. Sample the graph | none (script) | code | — |
| 1. Draft the request and node golds | G1 | teacher | the output parses and matches the graph |
| 2. Solve each node from the gold of its parents | G2 | teacher, new call | every node answer equals its gold (G3) |
| 3. Judge equality | G3 | judge model | — |
| 4. Coverage: final answer from node golds only | G4 | teacher | the result equals `answer_gold` (G3) |
| 5. Connectedness: last node without its parents' values | G2 with parents removed | teacher | the answer is **wrong** (the edge is real) |
| 6. Scaffolded variants for evaluation | G5 | teacher | stage 2 passes on every variant |
| 7. Human review | — | Maria | every item is read |

Rules for the pipeline:

- The teacher and the judge are not among the compared systems (Mamay-4B, Lapa, Aya,
  Mamay-12B). Stages 2, 4 and 5 drop items on the teacher's answers, never on the
  answers of a compared model; this keeps the rule «never select items using model
  scores».
- Record the teacher, the judge, their versions, temperature and the git commit in the
  dataset meta file.
- An item that fails any check is rewritten once with G1 and the failure reason, then
  dropped. Count the drops by stage and report them, as STaD does (Table 1).

## Size and quotas (decided 28 Oct 2026: 100 requests, few simple items)

| Class | Items | Shapes |
|---|---:|---|
| simple | 10 | one per task type (control for wrong splits) |
| chain | 30 | two or three nodes |
| graph | 30 | joins, forks, four-node fork-then-join |
| parallel | 30 | two or three nodes |

Task types to cover: time (durations with seconds, schedules, time zones, invalid dates, seconds to days/hours;
ToT style), ZNO items, search in documents given in the request, summaries, classification of items, word
problems, extraction, translation, code, alignment. Chains and parallel items all have two or more skills; three
one-skill math graphs are kept as shape controls (vault item 9).

Gold kinds: `exact`, `accepted`, `example`. For `example` nodes (open writing) the gold is only an example; write
form checks plus `criteria` for the judge, never a check that compares with the example text.

Splits: graphs_mixed_100_dev is a development set as a whole. A separate test set is built later with the same prompts,
never selected by model scores, and run once after the router and planner are frozen.

## Shared definitions (pasted into G1, G2 and G4)

```text
SKILLS
- translate: translation between Ukrainian and English only.
- knowledge: reading and choosing from a text given in the request, or everyday commonsense
  (which items are needed on a hike). No world facts beyond commonsense.
- instruct: writing or rewriting text: a letter, a reply, an explanation, a summary.
- code: writing or fixing a Python function.
- math: arithmetic word problem; several operations inside one node are fine.
- alignment: moral or social judgement as a digit.

NODE
One call to one skill. Form requirements (JSON, number of sentences, letter case,
language of the answer) are attributes of a node, not nodes.

EDGE
Node B depends on node A only if B uses the VALUE that A produced: B contains it,
computes from it, or must agree with it. Steps written as «спочатку … потім» without
such a value are parallel, not a chain.

MERGE RULE
Neighbouring steps that would go to the same skill are one node. A mechanical step
(copying a number that is written in the request) is merged into the next node. A step
that changes the skill stays a node of its own, however short.
```

## G1. Draft an item from a sampled graph

Input: the class, the node list with skills and edges, the benchmark style to imitate,
and up to three items already accepted for this class (so the wording varies).

```text
You write one item for a Ukrainian benchmark of composite requests. The graph of the
item is fixed below. Write a natural user request in Ukrainian that needs exactly this
graph, and the gold answer of every node.

{{SHARED_DEFINITIONS}}

GRAPH
class: {{class}}
nodes: {{nodes}}            # e.g. [{"id":"n1","skill":"knowledge"},{"id":"n2","skill":"math","depends_on":["n1"]}]
style_of: {{style_of}}      # e.g. "AgentCoMa", "GSM8K plus writing", "FLORES"

RULES FOR THE REQUEST
1. Every fact needed to answer is inside the request. No world facts, no dates or
   prices the reader must know.
2. The request needs every node and no other node. If a node could be skipped and the
   answer would still be right, the graph is wrong: change the request.
3. Each edge carries a value. Write in "value_dependency" which value the later node
   uses.
4. Do not mark the steps with «спочатку», «потім», «далі», numbered lists or one verb
   per step, unless the style needs it. Real users rarely write like that. Vary the
   wording from the accepted items shown below.
5. One reading only: no question that can be counted or understood two ways. Units,
   currency and rounding are stated when they matter.
6. Numbers are small enough to check by hand. Results that are not whole numbers have
   at most two decimals; write the expected rounding.
7. Do not copy text from any benchmark. "style_of" is a style, not a source.
8. Translate nodes: give the source text in the request; the gold is one accepted
   translation plus 2 to 4 other accepted ones.
9. Code nodes: the gold is a function and 3 to 5 unit tests (input → output).

RULES FOR THE GOLD
- "gold" of a node is the node's own result, as short as possible (a number, a list,
  a sentence, a function).
- "does" says what the node computes, with the arithmetic written out for math nodes
  (e.g. "3 × 1450 + 180 × 2 × 3 = 5430").
- "answer_gold" is the full answer to the request.
- "check" says how the final answer is scored: numeric (with tolerance), set of items,
  rules (number of sentences, language, JSON keys), unit tests, or "judge" with the
  facts the answer must state.
- "reasoning_steps" counts the operations inside math nodes.

ACCEPTED ITEMS OF THIS CLASS (do not repeat their topic or wording)
{{accepted_examples}}

Return only JSON:
{"request": "...",
 "nodes": [{"id": "n1", "skill": "...", "does": "...", "depends_on": [], "attributes": [], "gold": "..."}],
 "value_dependency": {"n1->n2": "..."},
 "answer_gold": "...",
 "check": {...},
 "reasoning_steps": 0}
```

Worked example (vault item 4; shown to the teacher once, outside the accepted list):

```json
{"request": "Я збираю рюкзак у похід у ліс на три дні. Є: спальник 1,8 кг, ноутбук 2,1 кг, намет 2,5 кг, фен 0,6 кг, казанок 0,9 кг. Вибери лише речі, які справді потрібні в поході, і порахуй їхню загальну вагу. Відповідь подай JSON: {\"items\": [...], \"kg\": число}.",
 "nodes": [
  {"id": "n1", "skill": "knowledge", "does": "choose what is needed in a forest", "depends_on": [], "attributes": [], "gold": "спальник, намет, казанок"},
  {"id": "n2", "skill": "math", "does": "1,8 + 2,5 + 0,9 = 5,2", "depends_on": ["n1"], "attributes": ["JSON"], "gold": "{\"items\": [\"спальник\", \"намет\", \"казанок\"], \"kg\": 5.2}"}],
 "value_dependency": {"n1->n2": "n2 adds exactly the items n1 chose"},
 "answer_gold": "{\"items\": [\"спальник\", \"намет\", \"казанок\"], \"kg\": 5.2}",
 "check": {"type": "all", "rubrics": [{"type": "json_fields", "fields": {"kg": 5.2}}, {"type": "set", "values": ["спальник", "намет", "казанок"]}]},
 "reasoning_steps": 2}
```

## G2. Solve one node (gold verification and connectedness)

Adapted from STaD «Sub-Task Answer». Run once per node, in graph order, in a new call
that does not see the drafted golds. For stage 5, run the last node with
`{{parent_results}}` set to «немає».

```text
Here is a user request and one step of its solution. Solve only this step.

Request:
{{request}}

Step: {{does_without_numbers}}       # the "does" text with the arithmetic removed
Skill: {{skill}}
Results of earlier steps:
{{parent_results}}                   # "n1: спальник, намет, казанок" or "немає"

Use only the request and the results above. Write your reasoning briefly, then the
result.
Return only JSON: {"explanation": "...", "answer": "..."}
```

## G3. Judge: are two answers the same

Adapted from STaD «Consistency». Used for stages 2, 4 and 5, and later as the LLM judge
for scoring (calibrate on about 50 hand-scored items first).

```text
Decide whether the model answer and the gold answer are the same in value.

Rules
1. Ignore formatting: 2, "2", 2.0, «Відповідь: 2» are the same.
2. Decimal comma and point are the same: 5,2 = 5.2. Spaces inside numbers do not matter:
   13 365 = 13365.
3. Numbers in words equal numbers in digits, in Ukrainian and in English (п'ять = 5).
4. Units and currency must match after conversion: 2 кг = 2000 г; 297 EUR ≠ 297 грн.
5. Letters of answer options: Cyrillic А and Latin A are the same option.
6. Lists of items: the same set, in any order, in any grammatical case
   (казанок = казанка).
7. Text answers (a translation, a sentence): the same meaning and every fact of the
   gold is present; style does not matter.
8. Empty or missing answer: score 0.

Model answer:
{{model_answer}}

Gold answer:
{{gold}}

Return only JSON: {"score": 0 or 1, "justification": "..."}
```

## G4. Coverage: answer from the node golds alone

Adapted from STaD appendix B. The teacher sees the steps and their golds but not the
request. If it cannot reach `answer_gold`, a step is missing from the graph.

```text
Here are the steps of a solution and their results. The original question is not shown.
Write the final answer that these results give.

{{nodes_with_does_and_gold}}

Return only JSON: {"answer": "..."}
```

Pass rule: G3 says 1 against `answer_gold`, ignoring form attributes (language and
format of the final answer are not recoverable without the request).

## G5. Scaffolded variant: the first j nodes already solved

Adapted from STaD «Scaffolded Variation». For a chain or graph with K nodes, make
variants j = 1 … K−1, revealing nodes in graph order. A target model that fails the
original request is run on the variants; the smallest j at which it succeeds is its
minimum scaffolding level k (k = −1 if none). For a join node, reveal both parents in
the same variant.

```text
Rewrite the user request so that the steps listed below are already done. Put their
results into the request in plain words, as if the user already knew them. Keep the
rest of the wording, every other fact and every format requirement unchanged. Do not
solve or hint at the remaining steps. Write in Ukrainian, as one paragraph.

Request:
{{request}}

Steps already done:
{{solved_nodes}}                     # e.g. "n1 (вибрати потрібні речі): спальник, намет, казанок"

Return only: Новий запит: <text>
```

Control (STaD appendix E): for every variant also build one where the revealed values
are replaced by «[ЗНАЧЕННЯ]». If the targets succeed on the control as often as on the
variant, the gain comes from the wording, not from the revealed values.
