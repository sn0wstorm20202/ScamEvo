from __future__ import annotations

import random
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.generator import GeneratorAction


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_anchor_cta(s: str) -> bool:
    s = s.lower()
    patterns = [
        r"\bclick\b",
        r"\bcall\b",
        r"\breply\b",
        r"\bverify\b",
        r"\bconfirm\b",
        r"\bupdate\b",
        r"\blogin\b",
        r"\bclaim\b",
        r"\bpay\b",
        r"\brecharge\b",
        r"\bopen\b",
        r"\btap\b",
        r"\blink\b",
    ]
    return any(re.search(p, s) for p in patterns)


def _has_anchor_urgency_or_reward_or_authority(s: str) -> bool:
    s = s.lower()

    urgency = [
        r"\burgent\b",
        r"\bimmediate\b",
        r"\bimmediately\b",
        r"\bnow\b",
        r"\btoday\b",
        r"\bwithin\b",
        r"\bfinal\b",
        r"\blast\b",
        r"\blimited\b",
        r"\bexpires\b",
    ]
    reward = [
        r"\bwin\b",
        r"\bprize\b",
        r"\breward\b",
        r"\bbonus\b",
        r"\bfree\b",
        r"\bcash\b",
        r"\boffer\b",
        r"\bdiscount\b",
    ]
    authority = [
        r"\bbank\b",
        r"\bupi\b",
        r"\bpolice\b",
        r"\bgov\b",
        r"\bgovt\b",
        r"\bofficial\b",
        r"\baccount\b",
        r"\bsecurity\b",
    ]

    return any(re.search(p, s) for p in urgency + reward + authority)


def anchors_pass(text: str) -> bool:
    return _has_anchor_cta(text) and _has_anchor_urgency_or_reward_or_authority(text)


def _lexical_swap(text: str, rnd: random.Random) -> str:
    mapping: dict[str, list[str]] = {
        "urgent": ["important", "time-sensitive"],
        "verify": ["confirm", "validate"],
        "account": ["profile", "account"],
        "bank": ["bank", "financial provider"],
        "blocked": ["restricted", "blocked"],
        "click": ["tap", "open"],
        "reply": ["respond", "reply"],
        "winner": ["selected", "winner"],
    }

    def repl(match: re.Match[str]) -> str:
        w = match.group(0)
        lw = w.lower()
        if lw in mapping and rnd.random() < 0.45:
            rep = rnd.choice(mapping[lw])
            if w[:1].isupper():
                rep = rep[:1].upper() + rep[1:]
            return rep
        return w

    return re.sub(r"\b[\w']+\b", repl, text)


_URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)


def _obfuscate(text: str, rnd: random.Random) -> str:
    def url_repl(match: re.Match[str]) -> str:
        url = match.group(0)
        url = re.sub(r"^https", "hxxps", url, flags=re.IGNORECASE)
        url = re.sub(r"^http", "hxxp", url, flags=re.IGNORECASE)
        url = url.replace("://", "://")
        url = url.replace(".", "[.]")
        return url

    text = _URL_RE.sub(url_repl, text)

    def digits_repl(match: re.Match[str]) -> str:
        s = match.group(0)
        if len(s) <= 1:
            return s
        if rnd.random() < 0.5:
            return " ".join(list(s))
        return s

    text = re.sub(r"\d+", digits_repl, text)
    return text


def _urgency_modulate(text: str, rnd: random.Random) -> str:
    lower = text.lower()
    increase = rnd.random() < 0.6

    if increase:
        if not lower.startswith("urgent"):
            text = "URGENT: " + text
        if "today" not in lower and rnd.random() < 0.5:
            text = text + " Today."
        return text

    text = re.sub(r"\burgent\b", "important", text, flags=re.IGNORECASE)
    text = re.sub(r"\bimmediately\b", "soon", text, flags=re.IGNORECASE)
    return text


def _tfidf_similarities(base: str, candidates: Iterable[str]) -> list[float]:
    candidates_list = list(candidates)
    if not candidates_list:
        return []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    x = vectorizer.fit_transform([base] + candidates_list)
    sims = cosine_similarity(x[0:1], x[1:]).reshape(-1)
    return [float(v) for v in sims]


def mutate_text(
    *,
    base_text: str,
    num_candidates: int,
    seed: int,
    actions: list[GeneratorAction],
    similarity_threshold: float,
    require_anchors: bool,
) -> list[dict[str, Any]]:
    base_text = _normalize_ws(base_text)
    if not base_text:
        return []

    allowed = [a for a in actions if a in {"lexical_swap", "obfuscate", "urgency"}]
    if not allowed:
        return []

    rnd = random.Random(seed)
    raw: list[tuple[str, list[GeneratorAction]]] = []
    seen: set[str] = {base_text}

    max_attempts = max(20, num_candidates * 20)
    for _ in range(max_attempts):
        k = 1
        if len(allowed) >= 2 and rnd.random() < 0.5:
            k = 2

        chosen = rnd.sample(allowed, k=k)
        candidate = base_text

        for action in chosen:
            if action == "lexical_swap":
                candidate = _lexical_swap(candidate, rnd)
            elif action == "obfuscate":
                candidate = _obfuscate(candidate, rnd)
            elif action == "urgency":
                candidate = _urgency_modulate(candidate, rnd)

        candidate = _normalize_ws(candidate)
        if not candidate:
            continue
        if candidate in seen:
            continue

        seen.add(candidate)

        if require_anchors and not anchors_pass(candidate):
            continue

        raw.append((candidate, chosen))

        if len(raw) >= num_candidates * 5:
            break

    if not raw:
        return []

    texts = [t for t, _ in raw]
    sims = _tfidf_similarities(base_text, texts)

    kept: list[dict[str, Any]] = []
    for (cand_text, cand_actions), sim in zip(raw, sims, strict=False):
        if sim < similarity_threshold:
            continue
        kept.append(
            {
                "text": cand_text,
                "similarity": float(sim),
                "actions": list(cand_actions),
                "metadata": {
                    "synthetic": True,
                    "watermark": "SCAMEVO_SYNTH_v1",
                    "created_at": _utc_now_iso(),
                    "generator": "mvp",
                },
            }
        )

    kept.sort(key=lambda x: x["similarity"], reverse=True)
    return kept[:num_candidates]
