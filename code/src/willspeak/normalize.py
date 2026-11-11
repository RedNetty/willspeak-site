from __future__ import annotations

import json
import re
from pathlib import Path

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "shan't": "shall not",
    "y'all": "you all", "ain't": "is not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "what's": "what is", "who's": "who is", "he's": "he is",
    "she's": "she is", "let's": "let us", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "mustn't": "must not", "we're": "we are", "they're": "they are",
    "you're": "you are", "we've": "we have", "they've": "they have",
    "you've": "you have", "we'll": "we will", "they'll": "they will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
}

CONTRACTIONS.update({
    "im": "i am", "ive": "i have", "ill": "i will", "id": "i would",
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not",
    "werent": "were not", "havent": "have not", "hasnt": "has not",
    "hadnt": "had not", "wont": "will not", "cant": "cannot",
    "couldnt": "could not", "wouldnt": "would not", "shouldnt": "should not",
    "youre": "you are", "youve": "you have", "youll": "you will",
    "theyre": "they are", "theyve": "they have", "weve": "we have", "thats": "that is", "whats": "what is",
    "lets": "let us",
})

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]


def _n2w(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        r = ONES[n // 100] + " hundred"
        return r if n % 100 == 0 else r + " " + _n2w(n % 100)
    if n < 1_000_000:
        r = _n2w(n // 1000) + " thousand"
        return r if n % 1000 == 0 else r + " " + _n2w(n % 1000)
    return str(n)


US_SPELLING: dict[str, str] = json.loads(
    Path(__file__).with_name("us_spelling.json").read_text(encoding="utf-8"))

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


class EnglishNormalizer:
    def __call__(self, text: str, us_spelling: bool = True) -> str:
        t = text.lower().strip()
        t = t.replace("’", "'").replace("‘", "'").replace("–", " ").replace("—", " ")
        t = _PUNCT.sub(" ", t)
        toks = []
        for w in t.split():
            w = w.strip("'")
            if w in CONTRACTIONS:
                toks.extend(CONTRACTIONS[w].split())
            elif w.isdigit() and len(w) <= 7:
                toks.extend(_n2w(int(w)).split())
            elif w:
                toks.append(US_SPELLING.get(w, w) if us_spelling else w)
        return _WS.sub(" ", " ".join(toks)).strip()


normalize = EnglishNormalizer()

_NOT_IN_VOCAB = re.compile(r'[;"]')


def stock_form(text: str) -> str:
    t = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-")
    t = _NOT_IN_VOCAB.sub("", t)
    t = " ".join(t.split())
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if t[-1] not in ".?!":
        t += "."
    return t
