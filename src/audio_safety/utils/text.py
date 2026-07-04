"""Small text normalization and WER helpers."""

import re
import unicodedata

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "for",
    "can",
    "could",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "would",
    "with",
}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def word_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def content_tokens(text: str) -> set[str]:
    return {token for token in word_tokens(text) if len(token) > 2 and token not in _STOPWORDS}


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1] / len(ref)


def token_overlap(reference: str, hypothesis: str) -> float:
    ref_tokens = content_tokens(reference)
    if not ref_tokens:
        return 1.0
    hyp_tokens = set(word_tokens(hypothesis))
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)
