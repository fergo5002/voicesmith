"""Text normalisation, error rates and chunking."""

from __future__ import annotations

import re
import unicodedata

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + number_words(rest))
    for div, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            rest = n % div
            tail = "" if rest == 0 else (" and " if rest < 100 else " ") + number_words(rest)
            return number_words(n // div) + " " + word + tail
    return str(n)


def _expand_numbers(text: str) -> str:
    def year(m: re.Match[str]) -> str:
        y = int(m.group(0))
        if 1100 <= y <= 1999 or 2010 <= y <= 2099:
            hi, lo = divmod(y, 100)
            return number_words(hi) + " " + ("hundred" if lo == 0 else ("oh " + _ONES[lo] if lo < 10 else number_words(lo)))
        return number_words(y)

    # "1,200" is a quantity, never a year, so expand grouped numbers before the year rule sees them.
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: number_words(int(m.group(0).replace(",", ""))), text)
    text = re.sub(r"(\d+)%", r"\1 percent", text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", year, text)
    text = re.sub(r"\d+", lambda m: number_words(int(m.group(0))) if len(m.group(0)) < 10 else m.group(0), text)
    return text


def normalise(text: str) -> str:
    """Lower-case words only, so a transcript can be compared with its script."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("&", " and ")
    text = _expand_numbers(text)
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)  # paralinguistic tags like [laugh]
    text = re.sub(r"[^\w' ]+", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref).split(), normalise(hyp).split()
    return _edits(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r, h = list(normalise(ref).replace(" ", "")), list(normalise(hyp).replace(" ", ""))
    return _edits(r, h) / max(1, len(r))


def repeated_ngram(ref: str, hyp: str, n: int = 3) -> str | None:
    """A run of ``n`` words that the transcript repeats more often than the script does."""
    r, h = normalise(ref).split(), normalise(hyp).split()

    def counts(words: list[str]) -> dict[tuple[str, ...], int]:
        out: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - n + 1):
            key = tuple(words[i : i + n])
            out[key] = out.get(key, 0) + 1
        return out

    rc = counts(r)
    for gram, c in counts(h).items():
        if c > rc.get(gram, 0) and c >= 2:
            return " ".join(gram)
    return None


_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e|No|approx)\.$", re.I)


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text)
    out: list[str] = []
    for part in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]


def chunk(text: str, max_chars: int) -> list[str]:
    """Group whole sentences into chunks of at most ``max_chars``.

    Fewer, longer chunks keep one continuous performance; splitting is a
    fallback for engines that cannot take the whole text, because every join is
    a chance for the pitch and energy to reset.
    """
    out: list[str] = []
    cur = ""
    for s in sentences(text):
        while len(s) > max_chars:
            cut = s.rfind(", ", 0, max_chars)
            cut = cut if cut > max_chars // 3 else s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            head, s = s[: cut + 1].strip(), s[cut + 1 :].strip()
            if cur:
                out.append(cur)
                cur = ""
            out.append(head)
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur += " " + s
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out
