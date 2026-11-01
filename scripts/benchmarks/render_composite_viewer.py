#!/usr/bin/env python3
"""Render a composite benchmark (and, if given, its per-item scores) as one HTML page.

The page is self-contained: the data is embedded, there are no network calls. Open it
in a browser. Left: items with filters; right: the request, the gold plan step by step
with rubrics, and the score of every system on that item.

    python scripts/benchmarks/render_composite_viewer.py \\
        benchmarks/archive/mixed_ua_composite_probe100.jsonl \\
        --scores results/composite_probe100/scores_detail.jsonl \\
        --out benchmarks/viewers/mixed_ua_composite_probe100.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_LABELS = {
    "router_direct": "One call",
    "hybrid_gated": "Planner (gated)",
    "template": "Template",
    "oracle": "Gold plan",
}
WORDING_LABELS = {
    "A": "A · as in composite v2",
    "B": "B · two verbs",
    "C": "C · спершу / потім",
    "D": "D · no sequence words",
}
STAGE_KEYS = ("id", "intent", "actual_intent", "score", "detail")
ROW_KEYS = ("final_score", "final_success", "final_detail", "stage_mean", "all_stages_pass", "plan_valid",
            "plan_match", "intent_f1", "superfluous_steps", "missing_steps", "workflow_error", "calls",
            "latency_ms", "selected_template")


def load_jsonl(path: Path) -> tuple[str, list[dict]]:
    header, rows = "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header = header or line.lstrip("# ").strip()
        elif line.strip():
            rows.append(json.loads(line))
    return header, rows


def load_scores(path: Path | None) -> dict[str, dict[str, dict]]:
    if not path or not path.is_file():
        return {}
    out: dict[str, dict[str, dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        row = {k: r.get(k) for k in ROW_KEYS}
        row["stages"] = [{k: s.get(k) for k in STAGE_KEYS} for s in r.get("stage_scores") or []]
        out.setdefault(str(r["id"]), {})[str(r["system"])] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bench", type=Path)
    p.add_argument("--scores", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    header, items = load_jsonl(args.bench)
    scores = load_scores(args.scores)
    systems = [s for s in SYSTEM_LABELS if any(s in v for v in scores.values())]
    systems += sorted({s for v in scores.values() for s in v} - set(systems))
    data = {
        "title": args.bench.stem,
        "header": header,
        "source": str(args.bench.resolve().relative_to(ROOT) if args.bench.resolve().is_relative_to(ROOT) else args.bench),
        "scores_source": str(args.scores) if scores else "",
        "systems": [{"key": s, "label": SYSTEM_LABELS.get(s, s)} for s in systems],
        "wordings": WORDING_LABELS,
        "items": [{**it, "scores": scores.get(str(it["id"]), {})} for it in items],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = args.out or ROOT / "benchmarks/viewers" / f"{args.bench.stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__TITLE__", args.bench.stem).replace("__DATA__", payload), encoding="utf-8")
    print(f"wrote {out} ({len(items)} items, {len(systems)} systems)")


TEMPLATE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f5f1; --surface: #fcfcfb; --surface-2: #efeee9; --border: #dedcd5;
  --ink: #1a1a19; --ink-2: #52514e; --ink-3: #8a8984; --accent: #2a78d6;
  --good: #1f7a3a; --good-bg: #e3f2e7; --bad: #b3261e; --bad-bg: #fbe7e5; --mid: #8a5a00; --mid-bg: #fbf0d9;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --surface-2: #242422; --border: #353532;
    --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86; --accent: #3987e5;
    --good: #7fd197; --good-bg: #1d3324; --bad: #f19a92; --bad-bg: #3a1f1c; --mid: #e7c06b; --mid-bg: #352b16;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --surface-2: #242422; --border: #353532;
  --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86; --accent: #3987e5;
  --good: #7fd197; --good-bg: #1d3324; --bad: #f19a92; --bad-bg: #3a1f1c; --mid: #e7c06b; --mid-bg: #352b16;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; }
header { padding: 14px 20px 10px; border-bottom: 1px solid var(--border); background: var(--surface); }
header h1 { font-size: 17px; margin: 0 0 2px; }
header .sub { color: var(--ink-3); font-size: 12px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 20px; border-bottom: 1px solid var(--border); background: var(--surface); }
.filters input, .filters select { font: inherit; color: var(--ink); background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; }
.filters input[type=search] { flex: 1 1 220px; min-width: 0; }
.filters .count { color: var(--ink-3); font-size: 12px; margin-left: auto; }
main { display: grid; grid-template-columns: minmax(260px, 360px) 1fr; height: calc(100vh - 112px); }
#list { overflow-y: auto; border-right: 1px solid var(--border); background: var(--surface); }
.row { padding: 9px 14px; border-bottom: 1px solid var(--border); cursor: pointer; }
.row:hover { background: var(--surface-2); }
.row.active { background: var(--surface-2); box-shadow: inset 3px 0 0 var(--accent); }
.row .top { display: flex; gap: 6px; align-items: baseline; font-size: 12px; color: var(--ink-3); }
.row .id { font-family: var(--mono); color: var(--ink-2); }
.row .text { margin-top: 2px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.row .marks { display: flex; gap: 4px; margin-top: 5px; }
.mark { font: 11px var(--mono); padding: 1px 5px; border-radius: 4px; background: var(--surface-2); color: var(--ink-2); }
.s-good { background: var(--good-bg); color: var(--good); }
.s-mid { background: var(--mid-bg); color: var(--mid); }
.s-bad { background: var(--bad-bg); color: var(--bad); }
#detail { overflow-y: auto; padding: 18px 24px 40px; }
#detail h2 { font-size: 16px; margin: 0 0 4px; font-family: var(--mono); }
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.chip { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); color: var(--ink-2); background: var(--surface); }
section { margin-bottom: 20px; }
section > h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-3); margin: 0 0 6px; }
.box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
pre.text { margin: 0; white-space: pre-wrap; word-break: break-word; font: 13.5px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif; }
.steps { display: grid; gap: 10px; }
.step { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
.step .head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px; }
.step .sid { font-family: var(--mono); font-weight: 600; }
.intent { font-size: 12px; padding: 1px 8px; border-radius: 4px; background: var(--surface-2); border: 1px solid var(--border); }
.dep { font-size: 12px; color: var(--ink-3); }
.final { font-size: 11px; color: var(--accent); border: 1px solid var(--accent); border-radius: 4px; padding: 0 6px; }
.step pre.text { font-size: 13px; color: var(--ink-2); }
.rubric { margin-top: 8px; font-size: 12.5px; color: var(--ink-2); border-top: 1px dashed var(--border); padding-top: 6px; }
.rubric code, td code { font-family: var(--mono); font-size: 12px; }
table { border-collapse: collapse; width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: top; font-size: 13px; }
th { color: var(--ink-3); font-weight: 600; font-size: 12px; background: var(--surface-2); }
td.num { font-family: var(--mono); white-space: nowrap; }
.stages { display: flex; flex-wrap: wrap; gap: 4px; }
.empty { color: var(--ink-3); padding: 30px; }
.hint { color: var(--ink-3); font-size: 12px; }
@media (max-width: 760px) {
  main { grid-template-columns: 1fr; height: auto; }
  #list { max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--border); }
  #detail { padding: 14px 16px 40px; }
  .filters, header { padding-left: 16px; padding-right: 16px; }
  .table-wrap { overflow-x: auto; }
}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="sub" id="sub"></div>
</header>
<div class="filters">
  <input type="search" id="q" placeholder="Пошук у запиті, id, кроках…" aria-label="Search">
  <select id="family" aria-label="Family"></select>
  <select id="wording" aria-label="Wording"></select>
  <select id="depth" aria-label="Depth"></select>
  <select id="split" aria-label="Split"></select>
  <select id="sys" aria-label="Filter by system result"></select>
  <span class="count" id="count"></span>
</div>
<main>
  <div id="list" role="listbox" aria-label="Items"></div>
  <div id="detail"></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (v) => typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : (v ?? "—");
const cls = (v) => typeof v !== "number" ? "" : v >= 0.999 ? "s-good" : v <= 0.001 ? "s-bad" : "s-mid";
const yes = (v) => v === true ? "✓ так" : v === false ? "✗ ні" : "—";

$("title").textContent = D.title;
$("sub").textContent = `${D.items.length} items · ${D.source}` + (D.scores_source ? ` · scores: ${D.scores_source}` : "") + (D.header ? ` · ${D.header}` : "");

function options(sel, label, values, names = {}) {
  sel.innerHTML = `<option value="">${label}: all</option>` + values.map((v) => `<option value="${esc(v)}">${esc(names[v] || v)}</option>`).join("");
}
const uniq = (k) => [...new Set(D.items.map((i) => i[k]).filter((v) => v !== undefined && v !== null))].sort();
const hasNodes = D.items.some((i) => Array.isArray(i.nodes));
options($("family"), hasNodes ? "Class" : "Family", uniq("family"));
options($("wording"), "Wording", uniq("wording"), D.wordings);
if (!uniq("wording").length) $("wording").hidden = true;
options($("depth"), "Depth", uniq("depth").map(String));
options($("split"), "Split", uniq("split"));
$("sys").innerHTML = `<option value="">Result: any</option>` + D.systems.flatMap((s) => [
  `<option value="${s.key}|fail">${esc(s.label)}: not full score</option>`,
  `<option value="${s.key}|pass">${esc(s.label)}: full score</option>`]).join("");
if (!D.systems.length) $("sys").hidden = true;

function haystack(it) {
  const steps = (it.oracle_plan?.steps || []).map((s) => `${s.id} ${s.intent} ${s.prompt}`).join(" ");
  const nodes = (it.nodes || []).map((n) => `${n.skill} ${n.does} ${n.gold}`).join(" ");
  return `${it.id} ${it.prompt} ${steps} ${nodes} ${it.family} ${it.style_of || ""} ${it.source_key || ""}`.toLowerCase();
}
D.items.forEach((it) => (it._h = haystack(it)));

let shown = [], current = null;
function apply() {
  const q = $("q").value.trim().toLowerCase(), fam = $("family").value, w = $("wording").value, dep = $("depth").value, sp = $("split").value;
  const [sys, want] = $("sys").value.split("|");
  shown = D.items.filter((it) =>
    (!q || it._h.includes(q)) && (!fam || it.family === fam) && (!w || it.wording === w) &&
    (!dep || String(it.depth) === dep) && (!sp || it.split === sp) &&
    (!sys || (it.scores[sys] && ((it.scores[sys].final_score >= 0.999) === (want === "pass")))));
  $("count").textContent = `${shown.length} / ${D.items.length}`;
  $("list").innerHTML = shown.map((it, n) => `
    <div class="row${current === it.id ? " active" : ""}" data-n="${n}" role="option">
      <div class="top"><span class="id">${esc(it.id)}</span><span>${esc(it.family)}</span><span>· ${esc(it.wording || it.subclass || "")}</span></div>
      <div class="text">${esc(it.prompt)}</div>
      ${D.systems.length ? `<div class="marks">${D.systems.map((s) => {
        const r = it.scores[s.key]; const v = r ? r.final_score : null;
        return `<span class="mark ${cls(v)}" title="${esc(s.label)}">${esc(s.label.split(" ")[0])} ${fmt(v)}</span>`;
      }).join("")}</div>` : ""}
    </div>`).join("") || `<div class="empty">Нічого не знайдено.</div>`;
  if (!shown.some((it) => it.id === current)) show(shown[0]);
}

function rubricText(r) {
  if (!r) return "";
  const t = r.type;
  if (t === "python_function") return `Python <code>${esc(r.entry_point)}</code>${r.only_code ? ", лише код" : ""}; тести: ` +
    (r.cases || []).map((c) => `<code>${esc(JSON.stringify(c.args))} → ${esc(JSON.stringify(c.expected))}</code>`).join(", ");
  if (t === "all" || t === "any") return (t === "all" ? "усе з: " : "будь-що з: ") + (r.rubrics || []).map(rubricText).join("; ");
  if (t === "has_number" || t === "numeric") return `${t === "numeric" ? "число" : "є число"} <code>${esc(r.value)}</code>`;
  if (t === "sentence_count") return `речень ${r.min === r.max ? r.min : r.min + "–" + r.max}`;
  if (t === "regex") return `шаблон <code>${esc(r.pattern)}</code>`;
  if (t === "line_prefix_count") return `${r.count} рядк. з <code>${esc(r.prefix)}</code>`;
  if (Array.isArray(r.values)) return `${esc(t)}: ` + r.values.map((v) => `<code>${esc(v)}</code>`).join(", ");
  const rest = Object.fromEntries(Object.entries(r).filter(([k]) => k !== "type"));
  return `${esc(t)} <code>${esc(JSON.stringify(rest))}</code>`;
}

function show(it) {
  if (!it) { $("detail").innerHTML = `<div class="empty">Вибери приклад зліва.</div>`; current = null; return; }
  current = it.id;
  document.querySelectorAll(".row").forEach((r) => r.classList.toggle("active", shown[r.dataset.n]?.id === it.id));
  const steps = it.oracle_plan?.steps || [];
  const graph = (it.nodes || []).map((n) => `
    <div class="step">
      <div class="head"><span class="sid">${esc(n.id)}</span><span class="intent">${esc(n.skill)}</span>
        <span class="dep">${n.depends_on?.length ? "← " + n.depends_on.map(esc).join(", ") : "без залежностей"}</span>
        ${(n.attributes || []).map((a) => `<span class="chip">${esc(a)}</span>`).join("")}
        ${n.id === it.final_step ? `<span class="final">фінальний</span>` : ""}</div>
      <div class="hint">${esc(n.does)}</div>
      <pre class="text">${esc(n.prompt)}</pre>
      <div class="rubric"><b>${{exact: "Gold (точний)", accepted: "Gold (один із прийнятних)", example: "Приклад, не еталон"}[n.gold_kind] || "Gold"}:</b>
        <pre class="text" style="display:inline">${esc(n.gold)}</pre></div>
      <div class="rubric">Перевірка: ${rubricText(n.check)}</div>
      ${(n.criteria || []).length ? `<div class="rubric">Критерії для судді: ${n.criteria.map(esc).join("; ")}</div>` : ""}
    </div>`).join("");
  const asm = steps.find((s) => s.id === "asm");
  const edges = Object.entries(it.value_dependency || {}).map(([k, v]) => `<div><code>${esc(k)}</code> — ${esc(v)}</div>`).join("");
  const plan = steps.map((s) => `
    <div class="step">
      <div class="head"><span class="sid">${esc(s.id)}</span><span class="intent">${esc(s.intent)}</span>
        <span class="dep">${s.depends_on?.length ? "← " + s.depends_on.map(esc).join(", ") : "без залежностей"}</span>
        ${s.id === it.final_step ? `<span class="final">фінальний</span>` : ""}</div>
      <pre class="text">${esc(s.prompt)}</pre>
      ${s.rubric ? `<div class="rubric">Перевірка: ${rubricText(s.rubric)}</div>` : ""}
    </div>`).join("");
  const res = D.systems.map((s) => {
    const r = it.scores[s.key];
    if (!r) return `<tr><td>${esc(s.label)}</td><td colspan="6" class="hint">немає прогону</td></tr>`;
    const stages = r.stages.map((st) => `<span class="mark ${cls(st.score)}" title="${esc(st.detail)}">${esc(st.id)}${st.actual_intent && st.actual_intent !== st.intent ? " (" + esc(st.actual_intent) + ")" : ""} ${fmt(st.score)}</span>`).join("");
    const err = [r.workflow_error, r.missing_steps ? `пропущено кроків: ${r.missing_steps}` : "", r.superfluous_steps ? `зайвих кроків: ${r.superfluous_steps}` : ""].filter(Boolean).join("; ");
    return `<tr><td>${esc(s.label)}</td><td class="num"><span class="mark ${cls(r.final_score)}">${fmt(r.final_score)}</span></td>
      <td>${esc(r.final_detail)}</td><td>${yes(r.plan_match)}</td><td class="num">${fmt(r.intent_f1)}</td>
      <td class="num">${fmt(r.calls)} · ${r.latency_ms != null ? (r.latency_ms / 1000).toFixed(1) + " с" : "—"}</td>
      <td><div class="stages">${stages || "—"}</div>${err ? `<div class="hint">${esc(err)}</div>` : ""}</td></tr>`;
  }).join("");
  $("detail").innerHTML = `
    <h2>${esc(it.id)}</h2>
    <div class="meta">
      <span class="chip">${esc(it.family)}</span>${it.wording ? `<span class="chip">${esc(D.wordings[it.wording] || it.wording)}</span>` : ""}
      ${it.subclass ? `<span class="chip">${esc(it.subclass)}</span>` : ""}${it.split ? `<span class="chip">${esc(it.split)}</span>` : ""}
      ${it.n_nodes ? `<span class="chip">вузлів ${it.n_nodes} · глибина ${it.depth} · ширина ${it.width} · вмінь ${it.distinct_skills}</span>` : `<span class="chip">глибина ${esc(it.depth)}</span><span class="chip">${steps.length} кроки</span>`}
      ${it.style_of ? `<span class="chip">${esc(it.style_of)}</span>` : ""}${it.stad_id ? `<span class="chip">STaD ${esc(it.stad_id.slice(0, 8))}</span>` : ""}${it.source_id ? `<span class="chip">${esc(it.source_id)}</span>` : ""}
      ${it.shape_control ? `<span class="chip">контроль форми</span>` : ""}
      ${it.source_key ? `<span class="chip">${esc(it.source_key)}</span>` : ""}${it.provenance ? `<span class="chip">${esc(it.provenance)}</span>` : ""}
    </div>
    <section><h3>Запит</h3><div class="box"><pre class="text">${esc(it.prompt)}</pre></div></section>
    ${graph ? `<section><h3>Граф (gold)</h3><div class="steps">${graph}
      ${asm ? `<div class="step"><div class="head"><span class="sid">asm</span><span class="intent">instruct</span><span class="dep">← ${asm.depends_on.map(esc).join(", ")}</span><span class="final">фінальний</span></div><div class="hint">лише з'єднує частини</div></div>` : ""}</div></section>
      ${edges ? `<section><h3>Ребра: яке значення передається</h3><div class="box">${edges}</div></section>` : ""}
      <section><h3>Фінальна відповідь (gold)</h3><div class="box"><pre class="text">${esc(it.answer_gold)}</pre><div class="rubric">Перевірка: ${rubricText(it.check)}</div></div></section>`
      : `<section><h3>Еталонний план</h3><div class="steps">${plan || '<div class="hint">немає</div>'}</div></section>`}
    ${D.systems.length ? `<section><h3>Результати систем</h3><div class="table-wrap"><table>
      <tr><th>Система</th><th>Бал</th><th>Деталі фіналу</th><th>План = еталон</th><th>Intent F1</th><th>Виклики · час</th><th>Кроки</th></tr>${res}</table></div></section>` : ""}
    <p class="hint">Клавіші: ↑/↓ або j/k — попередній / наступний, / — пошук.</p>`;
  $("detail").scrollTop = 0;
}

$("list").addEventListener("click", (e) => { const r = e.target.closest(".row"); if (r) show(shown[+r.dataset.n]); });
["q", "family", "wording", "depth", "split", "sys"].forEach((id) => $(id).addEventListener("input", apply));
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select")) { if (e.key === "Escape") e.target.blur(); return; }
  const n = shown.findIndex((it) => it.id === current);
  if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); show(shown[Math.min(n + 1, shown.length - 1)]); }
  else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); show(shown[Math.max(n - 1, 0)]); }
  else if (e.key === "/") { e.preventDefault(); $("q").focus(); }
  else return;
  document.querySelector(".row.active")?.scrollIntoView({block: "nearest"});
});
apply();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
