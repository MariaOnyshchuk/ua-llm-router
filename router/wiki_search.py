"""Retrieval from Ukrainian Wikipedia for knowledge requests (no model training).

Flow for one request:
  gate (knowledge intent, no passage in the prompt, not a calculation)
  -> short keyword queries from the question and its options
  -> MediaWiki search API -> plain text of the top pages
  -> paragraphs scored with BM25 against the question and options
  -> the best paragraphs are put in front of the unchanged prompt.

Every API answer is cached as JSON on disk (one file per request). A run with
`offline=True` reads only the cache, so a frozen cache makes a run repeatable and
lets the cluster run without internet. Off by default everywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "ua-llm-router/0.1 (https://github.com/MariaOnyshchuk/ua-llm-router)"

# Lines like "Дай відповідь на тестове завдання ..." that only describe the answer format.
_INSTRUCTION_RE = re.compile(
    r"^(дай відповідь|прочитай|відповідай|відповідь\s*[—-]|answer\b|read\b)", re.IGNORECASE
)
_FORMAT_SENTENCE_RE = re.compile(r"(відповід[ьа][^.?!]*|відповідай[^.?!]*)[.!]?\s*$", re.IGNORECASE)
_OPTION_RE = re.compile(r"^\s*([A-EА-ДҐЄІЇ])[\)\.]\s*(.+)$")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яҐґЄєІіЇї'’ʼ-]+")
_ROMAN_RE = re.compile(r"^[IVXLCХІ]+$")  # Latin or Cyrillic look-alike Roman numerals
_SKIP_SECTIONS = ("джерела", "посилання", "література", "примітки", "див. також", "виноски")

_STOP = frozenset(
    """що який яка яке які якого якої яких якому якій якими укажіть вкажіть визначте оберіть
    виберіть назвіть позначте чи як у в на з із зі до від для за по про та і й а але або не ні
    це цей ця ці цього цієї того той та те ті був була було були є має мали можна слід під над
    при між без через після перед один одна одне всі усі всіх якщо коли де хто чому правильно
    написано рядку рядок варіант варіанти твердження наведених наведено поданих подано вказано
    такий така таке такі так його її їх він вона воно вони ви ми я ти себе свій своє своїх дай
    відповідь відповідай тестове завдання зно літера лише тільки одним коротким реченням
    якою яким більш менш найбільш відповідно the a an of to in on for and or is are was were which what""".split()
)
# Nouns that name the kind of question, not its topic.
_GENERIC = frozenset(
    """вид тип приклад наслідок причина ознака особливість твердження рядок варіант відповідь
    завдання ст рік століття частина положення характеристика""".split()
)


@dataclass
class WikiConfig:
    lang: str = "uk"
    cache_dir: Path = field(default_factory=lambda: ROOT / ".cache" / "wiki")
    offline: bool = False  # True: never call the API, use only cached answers
    max_queries: int = 6
    hits_per_query: int = 3
    max_pages: int = 6
    page_chars: int = 30000
    chunk_words: int = 120
    top_k: int = 3
    max_context_chars: int = 3000
    min_coverage: float = 0.25  # share of question lemmas a paragraph must contain
    intro_boost: float = 1.3
    max_stem_chars: int = 700  # a longer question carries its own passage (Belebele)
    timeout: float = 15.0


# ---------------------------------------------------------------- prompt parsing


def split_question(prompt: str) -> tuple[str, list[str]]:
    """Return (question text without format instructions, option texts)."""
    head, _, tail = prompt.strip().partition("Варіанти:")
    paras = [p.strip() for p in re.split(r"\n\s*\n", head) if p.strip()]
    while len(paras) > 1 and _INSTRUCTION_RE.match(paras[0]):
        paras = paras[1:]
    stem = "\n\n".join(paras)
    stem = _FORMAT_SENTENCE_RE.sub("", stem).strip() or stem
    options = []
    for line in tail.splitlines():
        m = _OPTION_RE.match(line)
        if m:
            options.append(m.group(2).strip())
    return stem, options


def needs_retrieval(prompt: str, intent: str, cfg: WikiConfig | None = None) -> tuple[bool, str]:
    cfg = cfg or WikiConfig()
    if intent != "knowledge":
        return False, f"intent {intent}"
    stem, _ = split_question(prompt)
    if re.match(r"\s*прочитай", prompt, re.IGNORECASE) or re.match(r"\s*прочитай", stem, re.IGNORECASE):
        return False, "passage given"
    if len(stem) > cfg.max_stem_chars:
        return False, "passage given"
    # years and dates are fine; arithmetic signs between numbers or symbols mean a calculation
    if len(re.findall(r"\d\s*[+*/=^×÷]\s*[\d(a-zA-Z]|[=√∫≤≥]", stem)) >= 2:
        return False, "calculation"
    return True, "knowledge question"


# ---------------------------------------------------------------- text helpers


def _words(text: str) -> list[str]:
    return [w.strip("'’ʼ-") for w in _WORD_RE.findall(text) if w.strip("'’ʼ-")]


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer(lang="uk")
    except Exception:  # library or Ukrainian dictionary missing: fall back to prefixes
        return None


@lru_cache(maxsize=200_000)
def _parse(word: str) -> tuple[str, str]:
    """(lemma, part of speech) for one word; without pymorphy3 a 5-letter prefix and ''."""
    morph = _morph()
    low = word.lower()
    if morph is None:
        return (low[:5] if len(low) > 5 else low), ""
    p = morph.parse(low)[0]
    return p.normal_form, str(p.tag.POS or "")


def tokens(text: str) -> list[str]:
    """Lemmas of content words, used for paragraph scoring."""
    out = []
    for w in _words(text):
        lemma, _pos = _parse(w)
        if w.lower() not in _STOP and lemma not in _STOP and len(w) > 2:
            out.append(lemma)
    return out


def noun_groups(text: str) -> list[list[str]]:
    """Runs of adjectives/participles ending in nouns, e.g. ["стахановського", "руху"]."""
    groups: list[list[str]] = []
    cur: list[str] = []
    has_noun = False
    for w in _words(text):
        lemma, pos = _parse(w)
        keep = len(w) > 2 and w.lower() not in _STOP and lemma not in _STOP and not _ROMAN_RE.match(w)
        if keep and pos in ("ADJF", "PRTF") and not has_noun:
            cur.append(w)
        elif keep and (pos == "NOUN" or (not pos and w[:1].isupper())) and lemma not in _GENERIC:
            cur.append(w)
            has_noun = True
        else:
            if has_noun:
                groups.append(cur)
            cur, has_noun = [], False
            if keep and pos in ("ADJF", "PRTF"):
                cur = [w]
    if has_noun:
        groups.append(cur)
    return groups


def _group_rank(group: list[str], first_word: str) -> int:
    named = any(w[:1].isupper() and w != first_word for w in group)
    return 2 * named + (len(group) >= 2)


def build_queries(prompt: str, cfg: WikiConfig | None = None) -> list[str]:
    """Short noun-phrase queries: Wikipedia search ANDs every word, so long queries find nothing."""
    cfg = cfg or WikiConfig()
    stem_text, options = split_question(prompt)
    words = _words(stem_text)
    first = words[0] if words else ""
    groups = noun_groups(stem_text)
    ranked = sorted(range(len(groups)), key=lambda i: -_group_rank(groups[i], first))
    queries: list[str] = []
    if ranked:
        top = sorted(ranked[:2])
        queries.append(" ".join(w for i in top for w in groups[i])[:120])
        if len(groups[ranked[0]]) >= 2:
            queries.append(" ".join(groups[ranked[0]]))
    topic = groups[ranked[0]] if ranked else []
    for opt in options:
        if len(_words(opt)) > 6:
            continue
        opt_groups = noun_groups(opt)
        if opt_groups:
            q = [w for g in opt_groups for w in g]
            if len(q) == 1 and not q[0][:1].isupper():
                q += [w for w in topic if w not in q][:2]
        else:  # an option without a noun, e.g. "невичерпні": attach it to the question topic
            q = [w for w in _words(opt) if len(w) > 2 and w.lower() not in _STOP] + topic
        if q:
            queries.append(" ".join(q[:5]))
    out: list[str] = []
    for q in queries:
        if q and q not in out:
            out.append(q)
    return out[: cfg.max_queries]


def chunk_page(text: str, words_per_chunk: int) -> list[str]:
    """Paragraph chunks of about `words_per_chunk` words; reference sections dropped."""
    keep: list[str] = []
    for line in text.splitlines():
        header = re.match(r"^=+\s*(.*?)\s*=+$", line.strip())
        if header:
            if header.group(1).lower() in _SKIP_SECTIONS:
                break
            continue
        if line.strip():
            keep.append(line.strip())
    chunks: list[str] = []
    buf: list[str] = []
    for para in keep:
        buf.append(para)
        if sum(len(p.split()) for p in buf) >= words_per_chunk:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not docs:
        return []
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    q = set(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


# ---------------------------------------------------------------- API client with cache


class WikiClient:
    def __init__(self, cfg: WikiConfig | None = None, http: httpx.Client | None = None):
        self.cfg = cfg or WikiConfig()
        self._http = http
        self.api_calls = 0
        self.cache_hits = 0

    @property
    def api(self) -> str:
        return f"https://{self.cfg.lang}.wikipedia.org/w/api.php"

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "format": "json", "formatversion": 2}
        key = hashlib.sha1(json.dumps([self.cfg.lang, params], sort_keys=True).encode()).hexdigest()
        path = Path(self.cfg.cache_dir) / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.cfg.offline:
            raise LookupError(f"offline and not cached: {params}")
        if self._http is None:
            self._http = httpx.Client(timeout=self.cfg.timeout, headers={"User-Agent": USER_AGENT})
        resp = self._http.get(self.api, params=params)
        resp.raise_for_status()
        data = resp.json()
        self.api_calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"lang": self.cfg.lang, "params": params, "response": data}, ensure_ascii=False),
            encoding="utf-8",
        )
        return data

    def search(self, query: str) -> list[str]:
        data = self._get(
            {"action": "query", "list": "search", "srsearch": query,
             "srlimit": self.cfg.hits_per_query, "srprop": "", "srnamespace": 0}
        )
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def page_text(self, title: str) -> str:
        data = self._get(
            {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title}
        )
        pages = data.get("query", {}).get("pages", [])
        return (pages[0].get("extract") or "")[: self.cfg.page_chars] if pages else ""

    def close(self) -> None:
        if self._http is not None:
            self._http.close()


# ---------------------------------------------------------------- retrieval and prompt


def retrieve(prompt: str, client: WikiClient) -> dict[str, Any]:
    cfg = client.cfg
    queries = build_queries(prompt, cfg)
    errors: list[str] = []
    hits: list[list[str]] = []
    for q in queries:
        try:
            hits.append(client.search(q))
        except (httpx.HTTPError, LookupError) as exc:
            errors.append(f"search {q!r}: {exc}")
    # round robin: the best hit of every query first, so option queries are not crowded out
    titles: list[str] = []
    for rank in range(cfg.hits_per_query):
        for found in hits:
            if rank < len(found) and found[rank] not in titles:
                titles.append(found[rank])
    titles = titles[: cfg.max_pages]

    chunks: list[tuple[str, str]] = []
    intro: set[int] = set()  # index of each page's first paragraph chunk
    for t in titles:
        try:
            page_chunks = chunk_page(client.page_text(t), cfg.chunk_words)
            if page_chunks:
                intro.add(len(chunks))
            chunks += [(t, c) for c in page_chunks]
        except (httpx.HTTPError, LookupError) as exc:
            errors.append(f"page {t!r}: {exc}")

    stem_text, options = split_question(prompt)
    query_tokens = tokens(stem_text + " " + " ".join(options))
    chunk_tokens = [tokens(c) for _, c in chunks]
    # the opening paragraph usually defines the topic, so it wins close ties
    scores = [s * (cfg.intro_boost if i in intro else 1.0)
              for i, s in enumerate(bm25_scores(query_tokens, chunk_tokens))]
    wanted = set(tokens(stem_text))  # coverage of the question itself, not of all options
    coverage = [len(wanted & set(ct)) / max(len(wanted), 1) for ct in chunk_tokens]
    ranked = sorted(zip(scores, range(len(chunks))), key=lambda x: (-x[0], x[1]))

    passages: list[dict[str, Any]] = []
    used = 0
    for score, idx in ranked:
        if len(passages) >= cfg.top_k:
            break
        if coverage[idx] < cfg.min_coverage:
            continue
        title, text = chunks[idx]
        if used + len(text) > cfg.max_context_chars:
            text = text[: max(cfg.max_context_chars - used, 0)].rsplit(" ", 1)[0]
            if len(text) < 200:
                break
        used += len(text)
        passages.append(
            {"title": title, "url": f"https://{cfg.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
             "score": round(score, 3), "coverage": round(coverage[idx], 3), "text": text}
        )
    return {"queries": queries, "titles": titles, "n_chunks": len(chunks),
            "passages": passages, "errors": errors}


def augment_prompt(prompt: str, passages: list[dict[str, Any]]) -> str:
    if not passages:
        return prompt
    blocks = "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))
    return (
        "Нижче — фрагменти з української Вікіпедії, знайдені автоматично. Вони можуть не "
        "стосуватися питання; тоді відповідай зі своїх знань.\n\n"
        f"{blocks}\n\n---\n\n{prompt}"
    )


def wiki_augment(
    prompt: str,
    intent: str,
    client: WikiClient,
    *,
    query_prompt: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (prompt to send, metadata). `query_prompt` is the item text used for the gate and
    the queries when `prompt` carries extra wrapping such as few-shot examples."""
    source = query_prompt if query_prompt is not None else prompt
    ok, reason = needs_retrieval(source, intent, client.cfg)
    meta: dict[str, Any] = {"used": False, "gate": reason, "config": _config_meta(client.cfg)}
    if not ok:
        return prompt, meta
    found = retrieve(source, client)
    meta.update(found)
    meta["used"] = bool(found["passages"])
    return augment_prompt(prompt, found["passages"]), meta


def _config_meta(cfg: WikiConfig) -> dict[str, Any]:
    raw = asdict(cfg)
    raw.pop("cache_dir", None)
    return raw
