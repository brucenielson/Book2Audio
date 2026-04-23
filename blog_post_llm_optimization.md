# Why Are You Calling the LLM?

I've been building a Book-to-Audio pipeline. The idea is simple: take a PDF or
EPUB, parse it into clean paragraphs, and feed those paragraphs to a
text-to-speech engine. Simple enough. But scanned books are messy. OCR artifacts
like `Whenin` (a word join from "When in"), hyphenated line breaks carried over
from print, footnote markers embedded mid-sentence — the raw text often needs
cleaning before it's speakable.

So I added an LLM-based cleaner. You pass it a paragraph, it returns a cleaned
version and a classification: body, footnote, or drop. It works well. The
problem is I was calling it on every paragraph.

## Was That Actually a Problem?

At first glance, no. The LLM produces good results. Why complicate things?

Here's why: a typical chapter has hundreds of paragraphs. The LLM takes roughly
five seconds per call. Most paragraphs — the ones that say "He has refused to
pass Laws of immediate importance" — are already perfectly clean. Calling the
LLM on those is waste, pure and simple.

The more interesting question is: *when do we actually need the LLM?* The answer
is when the text contains something the parser couldn't fix — OCR artifacts,
broken word joins, corrupted characters. Not when it contains the word "justice."

So I built a gate.

## The Gate: `_all_words_valid`

The idea is straightforward. Before calling the LLM, check whether every word in
the paragraph is already valid English. If they all are, skip the LLM entirely.
The paragraph is clean enough.

```python
def _all_words_valid(text: str) -> bool:
    for token in text.split():
        stripped = re.sub(r"[,;:.!?()'\"—–]", '', token.lower())
        if not stripped or not word_validator.is_valid_word(stripped):
            return False
    return True
```

`word_validator.is_valid_word()` is doing real work here: it checks the NLTK
English words corpus, then falls back to Porter stemming and WordNet
lemmatization. So "running", "armies", "endeavoured" — these all pass. Just
checking a raw dictionary wouldn't cut it. English inflection is too irregular.

The result: the Declaration of Independence, which has clean prose, now sends
only a handful of paragraphs to the LLM — the ones with genuine OCR artifacts —
instead of all of them.

## What Was Actually Slow?

Once I had the gate in place and added timing instrumentation, the results were
revealing:

```
[TIMING] validation=6.24s (1 skipped) | llm=9.26s (1 calls) | total_timed=15.50s
```

Six seconds on word validation for *one paragraph*? That can't be right. The
validation logic is fast. What was happening is that NLTK loads its word corpus,
stemmers, and WordNet lemmatizer lazily — on first use. The first call to
`is_valid_word()` was triggering the entire NLTK initialization chain mid-paragraph,
and that takes six seconds.

The fix is a warm-up call at the start of processing:

```python
if self._cleaner is not None:
    word_validator.is_valid_word('warm')
```

After that, the timing looked quite different:

```
[TIMING] validation=0.01s (14 skipped) | llm=31.11s (7 calls) | total_timed=31.12s
```

Validation: effectively free. LLM: the clear bottleneck, as expected.

## What the Remaining LLM Calls Reveal

Of the seven paragraphs still going to the LLM, only two were genuine OCR
artifacts. The other five were false positives — cases where the validator
correctly identified a token it didn't recognise, but the token was actually fine:

- `self-evident` — the hyphen confused the token stripper
- `endeavoured` (twice) — valid British spelling, not in the American NLTK corpus
- `offences` — same problem
- `compleat` — archaic spelling, borderline

This is worth sitting with for a moment. The gate is working correctly in the
sense that it's catching tokens it genuinely can't validate. The problem is that
"can't validate" and "needs LLM cleaning" are not the same thing. A British
spelling is not an OCR artifact. We're using the wrong signal.

A better gate would know about British English. A simpler gate would split
hyphenated words and check each part. Neither fix is difficult. I haven't made
them yet.

## What I'd Do Differently

The timing instrumentation taught me something I should have measured from the
start: *where is the time actually going?* I had assumed the LLM was the
bottleneck. It was — but only after fixing a completely separate problem (NLTK
lazy loading) that was masking everything else.

This is a mundane lesson dressed up as a profound one: don't optimise until you
measure. But there's a sharper version of it. The six-second NLTK startup cost
was invisible in normal use because it happened once per run. It only became
visible when we added the gate — because the gate called `is_valid_word()` on the
*first paragraph* instead of waiting for the LLM to warm up later. The
optimisation revealed the bottleneck it was supposed to solve.

That's not irony. That's just what measurement does. You don't know what's slow
until something makes the slow thing visible.

## Where This Leaves Things

The pipeline is meaningfully faster. Most paragraphs skip the LLM entirely.
The remaining LLM calls are concentrated on the paragraphs that actually need
attention. The test suite now runs in about 30 seconds instead of... considerably
longer.

What remains: British spellings, hyphenated words, and the general question of
what "valid English" really means for a validator that's supposed to be catching
OCR artifacts, not judging orthographic conventions. That's a problem worth
thinking about carefully before reaching for a fix.

However, some caution is warranted. The gate is a heuristic. It will miss things.
A paragraph full of correctly-spelled words can still be semantically broken in
ways no word list will catch. The LLM is still there for the hard cases. That's
probably the right architecture — use it as a last resort, not a default.
