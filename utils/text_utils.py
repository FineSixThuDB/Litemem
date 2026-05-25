"""Text utilities — lemmatization, entity extraction, JSON parsing, message formatting.

Ported from mem0:
- ``lemmatize_for_bm25``  ← mem0/utils/lemmatization.py
- ``extract_entities``    ← mem0/utils/entity_extraction.py
- ``remove_code_blocks``  ← mem0/memory/utils.py
- ``extract_json``        ← mem0/memory/utils.py
- ``parse_messages``      ← mem0/memory/utils.py

spaCy is loaded lazily; if unavailable, lemmatization falls back to the
original text and entity extraction returns an empty list (mem0 has the
same behavior).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared spaCy loader (full + lemma-only). Mirrors mem0/utils/spacy_models.py.
# ---------------------------------------------------------------------------

_nlp_full = None
_nlp_lemma = None
_load_failed_full = False
_load_failed_lemma = False
_lock = threading.Lock()


def _ensure_model_available() -> None:
    """Download en_core_web_sm if spaCy is installed but model is missing."""
    import spacy  # imported here so spaCy is truly optional

    if not spacy.util.is_package("en_core_web_sm"):
        logger.info("Downloading spaCy model en_core_web_sm ...")
        from spacy.cli import download

        download("en_core_web_sm")


def get_nlp_full():
    """Full spaCy pipeline (used for entity extraction)."""
    global _nlp_full, _load_failed_full
    if _load_failed_full:
        return None
    if _nlp_full is not None:
        return _nlp_full
    with _lock:
        if _nlp_full is not None:
            return _nlp_full
        if _load_failed_full:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_full = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy unavailable for entity extraction: {e}")
            _load_failed_full = True
            return None
    return _nlp_full


def get_nlp_lemma():
    """Lemma-only spaCy pipeline (used for BM25 preprocessing)."""
    global _nlp_lemma, _load_failed_lemma
    if _load_failed_lemma:
        return None
    if _nlp_lemma is not None:
        return _nlp_lemma
    with _lock:
        if _nlp_lemma is not None:
            return _nlp_lemma
        if _load_failed_lemma:
            return None
        try:
            _ensure_model_available()
            import spacy

            _nlp_lemma = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception as e:
            logger.warning(f"spaCy unavailable for lemmatization: {e}")
            _load_failed_lemma = True
            return None
    return _nlp_lemma


# ---------------------------------------------------------------------------
# BM25 lemmatization
# ---------------------------------------------------------------------------

def lemmatize_for_bm25(text: str) -> str:
    """Return space-joined lemmas suitable for BM25 keyword matching.

    Falls back to lowercased original text when spaCy is unavailable.
    Behavior mirrors mem0 — including the trick of also emitting the
    original -ing form when it differs from the lemma (handles noun/verb
    ambiguity like "meeting" vs "meet").
    """
    if not text:
        return ""

    nlp = get_nlp_lemma()
    if nlp is None:
        # Graceful degradation: lowercase + simple tokenization.
        return " ".join(re.findall(r"\b[\w]+\b", text.lower()))

    doc = nlp(text.lower())
    tokens: List[str] = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Entity extraction — verbatim port of mem0/utils/entity_extraction.py
# ---------------------------------------------------------------------------

_GENERIC_HEADS = {
    "thing", "stuff", "way", "time", "experience", "situation", "case",
    "fact", "matter", "issue", "idea", "thought", "feeling", "place",
    "area", "part", "kind", "type", "sort", "lot", "bit", "day", "year",
    "week", "month", "moment", "instance", "example", "technique",
    "method", "approach", "process", "step", "tool", "result", "outcome",
    "goal", "task", "item", "topic", "scale", "size", "level", "degree",
    "amount", "number", "style", "look", "color", "colour", "shape",
    "form", "piece", "section", "side", "end", "edge", "surface", "point",
}

_CIRCUMSTANTIAL_MODS = {
    "solo", "individual", "team", "group", "joint", "collaborative",
    "first", "last", "next", "previous", "final", "initial", "main", "side",
}

_NON_SPECIFIC_ADJ = {
    "many", "few", "several", "some", "any", "all", "most", "more", "less",
    "much", "little", "enough", "various", "numerous", "multiple", "countless",
    "great", "good", "bad", "nice", "terrible", "awful", "awesome", "amazing",
    "wonderful", "horrible", "excellent", "poor", "best", "worst", "fine",
    "okay", "new", "old", "recent", "past", "future", "current", "previous",
    "next", "last", "first", "latest", "early", "late", "former", "modern",
    "ancient", "big", "small", "large", "tiny", "huge", "enormous", "long",
    "short", "tall", "high", "low", "wide", "narrow", "thick", "thin", "deep",
    "shallow", "similar", "different", "same", "other", "another", "such",
    "certain", "important", "main", "major", "minor", "key", "primary",
    "real", "actual", "true", "whole", "entire", "full", "complete", "total",
    "basic", "simple", "interesting", "boring", "exciting", "special",
    "particular", "general", "common", "unique", "rare", "typical", "usual",
    "normal", "regular", "possible", "likely", "potential", "available",
    "necessary", "only", "solo", "individual", "team", "group", "joint",
    "collaborative", "final", "initial", "side",
}

_GENERIC_ENDINGS = {
    "work", "works", "job", "jobs", "task", "tasks", "stuff", "things",
    "thing", "info", "information", "details", "data", "content", "material",
    "materials", "activities", "activity", "efforts", "effort", "options",
    "option", "choices", "choice", "results", "result", "output", "outputs",
    "products", "product", "items", "item",
}

_GENERIC_CAPS = {
    "works", "items", "things", "stuff", "resources", "options", "tips",
    "ideas", "steps", "ways", "methods", "tools", "features", "benefits",
    "examples", "details", "notes", "instructions", "guidelines",
    "recommendations", "suggestions", "overview", "summary", "conclusion",
    "introduction", "pros", "cons", "advantages", "disadvantages",
}

_FORMATTING_MARKERS = {"*", "-", "+", "•", "–", "—", "#", "##", "###", "**", "__"}


def _is_sentence_start(tokens: list, idx: int) -> bool:
    if idx == 0:
        return True
    tok = tokens[idx]
    if tok.is_sent_start:
        return True
    prev = tokens[idx - 1].text
    return prev in ".!?:" or prev in _FORMATTING_MARKERS or "\n" in prev


def _strip_generic_ending(toks: list) -> list:
    if len(toks) <= 1:
        return toks
    last = toks[-1].lemma_.lower() if hasattr(toks[-1], "lemma_") else toks[-1].lower()
    return toks[:-1] if last in _GENERIC_ENDINGS and len(toks) > 2 else toks


def _lemmatize_compound(toks: list) -> str:
    return " ".join(t.lemma_ if t.pos_ == "NOUN" else t.text for t in toks)


def _has_artifacts(txt: str) -> bool:
    return any(
        [
            "**" in txt or "__" in txt or ":*" in txt,
            re.search(r"\s\*\s|\s\*$|^\*\s", txt),
            "  " in txt or "\n" in txt or "\t" in txt,
            len(txt) > 100,
            txt.startswith(("•", "-", "+", "–", "—")),
        ]
    )


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract proper nouns / quoted strings / noun compounds from text.

    Returns ``[]`` when spaCy is unavailable. Output is a deduplicated list
    of ``(entity_type, entity_text)`` tuples.
    """
    if not text:
        return []
    nlp = get_nlp_full()
    if nlp is None:
        return []
    return _extract_entities_from_doc(nlp(text))


def extract_entities_batch(texts: List[str], batch_size: int = 32) -> List[List[Tuple[str, str]]]:
    """Batched ``extract_entities`` via spaCy's ``nlp.pipe``."""
    if not texts:
        return []
    nlp = get_nlp_full()
    if nlp is None:
        return [[] for _ in texts]
    return [_extract_entities_from_doc(doc) for doc in nlp.pipe(texts, batch_size=batch_size)]


def _extract_entities_from_doc(doc) -> List[Tuple[str, str]]:
    """Extract entities from a spaCy Doc — verbatim port of mem0's algorithm."""
    entities: List[Tuple[str, str]] = []
    text = doc.text
    tokens = list(doc)

    # === Proper noun sequences ===
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.text in _FORMATTING_MARKERS:
            i += 1
            continue
        is_cap = tok.text and tok.text[0].isupper()
        is_label = i + 1 < len(tokens) and tokens[i + 1].text == ":"
        if is_cap and not is_label and tok.pos_ in {"PROPN", "NOUN", "ADJ"}:
            seq = [(tok, i)]
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if (t.text and t.text[0].isupper()) or t.text.lower() in {
                    "'s", "of", "the", "in", "and", "for", "at", "is",
                }:
                    seq.append((t, j))
                    j += 1
                else:
                    break
            while seq and seq[-1][0].text.lower() in {"of", "the", "in", "and", "for", "at", "is", "'s"}:
                seq.pop()
            if seq:
                has_mid_cap = any(
                    not _is_sentence_start(tokens, idx)
                    for (t, idx) in seq
                    if t.text[0].isupper()
                    and t.text.lower() not in {"'s", "of", "the", "in", "and", "for", "at", "is"}
                )
                if has_mid_cap:
                    phrase = "".join(t.text_with_ws for (t, idx) in seq).strip()
                    if len(phrase) > 2:
                        entities.append(("PROPER", phrase))
            i = j
        else:
            i += 1

    # === Quoted text ===
    for m in re.finditer(r'"([^"]+)"', text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))
    for m in re.finditer(r"(?:^|[\s\(\[{,;])'([^']+)'(?=[\s\.,;:!?\)\]]|$)", text):
        if len(m.group(1).strip()) > 2:
            entities.append(("QUOTED", m.group(1).strip()))

    # === Noun-noun compounds ===
    for chunk in doc.noun_chunks:
        chunk_tokens = list(chunk)
        split_indices: list = []
        poss_splits: list = []
        for idx, tok in enumerate(chunk_tokens):
            if tok.dep_ == "case" and tok.text in {"'s", "’s", "'"}:
                split_indices.append(idx)
                poss_splits.append(idx)
            elif tok.pos_ == "PUNCT" and tok.text in {"'", '"', "‘", "’", "“", "”"}:
                split_indices.append(idx)
        if split_indices:
            groups: list = []
            prev = 0
            for split_idx in split_indices:
                if split_idx > prev:
                    groups.append(chunk_tokens[prev:split_idx])
                if split_idx in poss_splits:
                    next_split = next((s for s in split_indices if s > split_idx), None)
                    owned = chunk_tokens[split_idx + 1: next_split if next_split else len(chunk_tokens)]
                    if owned:
                        first_content = next((t for t in owned if t.pos_ not in {"PUNCT", "PART"}), None)
                        if not (first_content and first_content.text and first_content.text[0].isupper()):
                            prev = next_split if next_split else len(chunk_tokens)
                            continue
                prev = split_idx + 1
            if prev < len(chunk_tokens):
                groups.append(chunk_tokens[prev:])
        else:
            groups = [chunk_tokens]

        for group in groups:
            if not group:
                continue
            head = next((t for t in reversed(group) if t.pos_ in {"NOUN", "PROPN"}), None)
            if not head:
                continue
            head_generic = head.lemma_.lower() in _GENERIC_HEADS
            content = [
                t for t in group
                if t.pos_ not in {"DET", "PRON", "PUNCT", "PART", "ADP", "SCONJ", "NUM"}
                and (t.pos_ == "ADJ" or not t.is_stop)
            ]
            if not content:
                continue
            compound_toks = [t for t in content if t.dep_ == "compound"]
            adj_toks = [t for t in content if t.pos_ == "ADJ" or t.dep_ == "amod"]
            has_spec_adj = any(t.lemma_.lower() not in _NON_SPECIFIC_ADJ for t in adj_toks)
            if head_generic and not has_spec_adj and not compound_toks:
                continue
            if compound_toks:
                is_circ = any(t.lemma_.lower() in _CIRCUMSTANTIAL_MODS for t in compound_toks)
                if is_circ:
                    val = head.lemma_ if head.pos_ == "NOUN" else head.text
                    if len(val) > 2:
                        entities.append(("NOUN", val))
                else:
                    filtered = _strip_generic_ending(
                        [t for t in content if not (t.pos_ == "ADJ" and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                    )
                    if filtered:
                        phrase = _lemmatize_compound(filtered)
                        if len(phrase) > 3 and " " in phrase:
                            entities.append(("COMPOUND", phrase))
            elif len(content) > 1 and has_spec_adj:
                filtered = _strip_generic_ending(
                    [t for t in content if not ((t.pos_ == "ADJ" or t.dep_ == "amod") and t.lemma_.lower() in _NON_SPECIFIC_ADJ)]
                )
                if filtered:
                    phrase = _lemmatize_compound(filtered)
                    if len(phrase) > 3 and " " in phrase:
                        entities.append(("COMPOUND", phrase))

    # === Fallback: mis-tagged VERB heads ===
    processed = {e[1].lower() for e in entities if e[0] == "COMPOUND"}
    generic_verb_heads = _GENERIC_HEADS | {"find", "buy", "purchase", "sale", "deal", "trip", "visit"}

    def collect_compounds(head):
        return [t for t in doc if t.head == head and t.dep_ == "compound"]

    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ in {"pobj", "dobj", "nsubj"}:
            comps = sorted(collect_compounds(tok), key=lambda t: t.i)
            if comps:
                phrase_toks = comps if tok.lemma_.lower() in generic_verb_heads else comps + [tok]
                phrase = " ".join(t.text for t in phrase_toks)
                if phrase.lower() not in processed and len(phrase) > 3 and " " in phrase:
                    entities.append(("COMPOUND", phrase))
                    processed.add(phrase.lower())

    # === Cleanup ===
    seen: set = set()
    deduped: List[Tuple[str, str]] = []
    for t, e in entities:
        k = e.lower().strip()
        if k not in seen and len(k) > 2:
            seen.add(k)
            deduped.append((t, e))

    cleaned: List[Tuple[str, str]] = []
    for etype, etext in deduped:
        txt = re.sub(r"^\*+\s*|\s*\*+$", "", etext.strip())
        txt = re.sub(r"\s*:+$", "", txt)
        txt = re.sub(r"^\d+\s*\.\s*", "", txt)
        if not txt or len(txt) <= 2 or _has_artifacts(txt):
            continue
        if etype == "PROPER" and " " not in txt and txt.lower() in _GENERIC_CAPS:
            continue
        cleaned.append((etype, txt))

    type_pri = {"PROPER": 0, "COMPOUND": 1, "QUOTED": 2, "NOUN": 3, "VERB": 4}
    best: dict = {}
    for t, e in cleaned:
        k = e.lower()
        if k not in best or type_pri.get(t, 99) < type_pri.get(best[k][0], 99):
            best[k] = (t, e)
    deduped = list(best.values())

    all_lower = [e[1].lower() for e in deduped]
    return [(t, e) for t, e in deduped if not any(e.lower() != o and e.lower() in o for o in all_lower)]


# ---------------------------------------------------------------------------
# JSON + message helpers
# ---------------------------------------------------------------------------

def remove_code_blocks(content: str) -> str:
    """Strip enclosing ```fence``` and any ``<think>...</think>`` blocks."""
    if not content:
        return ""
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    inner = match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", inner, flags=re.DOTALL).strip()


def extract_json(text: str) -> str:
    """Extract a JSON-looking substring from a possibly-noisy LLM response."""
    if not text:
        return ""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx: end_idx + 1]
    return text


def parse_messages(messages) -> str:
    """Flatten a chat message list to ``role: content`` lines (mem0's format)."""
    parts = []
    for msg in messages:
        role = msg.get("role")
        if role in {"system", "user", "assistant"}:
            parts.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(parts) + ("\n" if parts else "")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
