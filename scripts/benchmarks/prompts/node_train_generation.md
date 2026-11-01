You write items for a Ukrainian training set. The set teaches a router which of three small
Ukrainian models to send a request to. Each item is ONE step that a model answers on its own:
one skill, one call. The items look like the steps of composite requests that a planner splits
into parts.

TASK
Write {{n}} items of skill "{{skill}}", subtype "{{subtype}}", form "{{form}}".

SKILLS AND SUBTYPES
- math: money word problems (GSM8K style); durations and clock times (ToT style); calendars
  and schedules; percent; unit conversion; "count the price of only these items".
- translate: Ukrainian → English and English → Ukrainian sentences; one word; a list; a short
  paragraph; a sentence with numbers and times.
- instruct: a notice with exact constraints; a summary of a given text; a fixed line format;
  an acrostic; JSON; a change of register (casual → official); a list with line prefixes; a
  length limit.
- knowledge: search in a timetable, notice or log given in the prompt; commonsense selection
  (what to take on a trip); classification of a given list; day of the week; reading a short
  text given in the prompt and choosing an option by letter.
- alignment: social acceptability of an everyday action on the scale
  "0 — погано, 1 — очікувано (звичайна поведінка), 2 — добре (явно похвально)",
  answered as a digit, as "Оцінка: <цифра>", or as two lines "Дія: …" and "Оцінка: …".
- code: a Python function from a Ukrainian description; fixing a bug in a given function; a
  function whose docstring or mapping table is given in the prompt; parsing a Ukrainian string;
  time arithmetic.

FORMS
- root: the prompt is written from scratch.
- with previous value: the prompt already contains a value that an earlier step produced, put
  in as that step would write it: a list of chosen items, a number, an action to judge, a
  translated sentence. Example: "Порахуй, скільки коштують лише ці речі: лінійка, гумка, клей".

RULES
1. Write the prompt in Ukrainian, as a user or a planner step would. Use Ukrainian settings:
   hryvnias, Ukrainian names, cities and holidays.
2. Every fact needed to answer is in the prompt. Commonsense only beyond that. No world
   facts, no dates, prices or rules the reader must know. Nothing about other countries' law
   or history.
3. One reading only. State units, currency and rounding when they matter. Numbers are small
   enough to check by hand; non-whole results have at most two decimals.
4. Do not copy text from any benchmark (GSM8K, STaD, FLORES, IFEval, ZNO, UAlign, HumanEval).
   The subtype is a style, not a source.
5. Vary topics and wording. Do not repeat the topics or phrasing of the accepted items below
   or of each other. Do not number steps or write «спочатку … потім».
6. math: end the prompt with "Наприкінці напиши: Відповідь: <число>." unless the answer is a
   time or a date in a stated format. Write the arithmetic in "does".
7. translate: the gold is one accepted translation. The check lists 2 to 4 groups of key
   words that any right translation must contain, each group with its accepted variants.
8. instruct: state the constraints exactly (number of sentences or lines, words that must or
   must not appear, format, length). The check tests exactly those constraints.
9. knowledge: the answer follows from the given text or from commonsense most adults share.
10. alignment: only actions whose rating most adults would agree on.
11. code: a function with a named entry point, 3 to 5 unit tests, and a gold that passes them.
12. For every item also write "wrong": a plausible wrong answer that the check must reject.

CHECK TYPES (use only these)
numeric {"value", "tolerance"}; has_number {"value", "tolerance"}; regex {"pattern",
"ignore_case"}; contains_any {"values"}; contains_all {"values"}; not_contains {"values"};
sentence_count {"min", "max"}; line_prefix_count {"prefix", "count"}; json_fields {"fields"};
python_function {"entry_point", "only_code", "cases": [{"args": [...], "expected": ...}]};
all {"rubrics": [...]}; any {"rubrics": [...]}.
Text checks ignore letter case. A gold must pass its check; the wrong answer must fail it.

EXAMPLE
{"skill": "math", "subtype": "count the price of only these items", "form": "with previous value",
 "prompt": "Ціни в канцелярській крамниці: лінійка 35 грн, гумка 12 грн, степлер 140 грн, клей 48 грн, глобус 520 грн, пенал 95 грн. Порахуй, скільки коштують лише ці речі: лінійка, гумка, клей, пенал\nНаприкінці напиши: Відповідь: <число>.",
 "does": "35 + 12 + 48 + 95 = 190",
 "gold": "Відповідь: 190",
 "wrong": "Відповідь: 850",
 "check": {"type": "numeric", "value": 190, "tolerance": 0.01}}

ACCEPTED ITEMS OF THIS SUBTYPE (do not repeat their topics or wording)
{{accepted_examples}}

Return only a JSON list of {{n}} objects with the keys
"skill", "subtype", "form", "prompt", "does", "gold", "wrong", "check".
