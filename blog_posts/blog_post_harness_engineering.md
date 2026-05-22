# The Harness Is the Hard Part

In Part 2, I built a layer that audits the LLM's output and silently undoes
the changes it wasn't supposed to make. That felt like a one-time fix. Patch
the obvious failure modes, move on.

It wasn't a one-time fix. It was the beginning of a pattern.

Over the next major development push, almost no time went into improving the
LLM itself. No prompt tuning. No model swaps. No clever chain-of-thought
tricks. Instead, nearly all the engineering effort went into the machinery
*around* the LLM: the code that prepares input before the model sees it, the
code that validates output after it returns, and the code that catches the
cases where the model can't even produce a parseable response.

There's a name for this pattern now. It's called harness engineering.

## What a Harness Actually Is

In February 2026, OpenAI published a post describing how a small team shipped
a million lines of production code without writing a single line by hand. The
agents wrote the code. The engineers designed the environment the agents worked
inside — the constraints, the feedback loops, the documentation structure, the
rules about what the agent was and wasn't allowed to do.

That environment is the harness.

Anthropic, ThoughtWorks, and others published similar frameworks within weeks.
The consensus: an AI agent without a harness is a raw language model guessing
its way through your problem. The harness is what makes the output reliable
enough to ship.

This is not a new idea dressed up in new vocabulary. It's the same thing
control systems engineers have known for decades: a powerful but unpredictable
component needs a control framework around it, not just better inputs.

What surprised me is how naturally the Book-to-Audio pipeline had been
converging on exactly this structure — without knowing it had a name.

## The LLM's Failure Modes Are Not Random

Here's the thing about running a language model on thousands of pages of text:
you stop seeing its outputs as probabilistic and start seeing them as
systematic.

The model doesn't randomly mangle things. It consistently mangles the same
things, in the same ways, every time.

- ASCII quotes go in. Curly quotes come back out.
- An em-dash goes in. A plain hyphen comes back out.
- A word split across a line break with a trailing hyphen (`reexamina-` /
  `tion`) sometimes comes back correctly rejoined, but the validator rejects it
  because neither fragment is a valid English word on its own.
- The JSON response occasionally contains a bare `"` inside a string value,
  which breaks the parser — every single retry, identically, because the
  problem is in the model's output pattern for that paragraph, not in a random
  failure.
- A paragraph containing `\'possibility\'` (backslash-escaped quotes from the
  JSON repair step) gets compared against the original, the backslash survives
  the strip, and the word restore fires on a perfectly good LLM output.

Each of these is deterministic. Given the same input, the model produces the
same mistake. That means each one can be caught with a rule.

This is the core insight behind harness engineering applied to an LLM pipeline:
don't try to make the model stop making mistakes. Instead, identify the mistake
patterns and intercept them mechanically.

## The Three Layers

The work on this branch fell naturally into three layers.

### Layer 1: Before the LLM Runs (Feedforward)

The first layer controls what the model sees and in what order.

**Sorting by physical position.** The PDF parser (Docling) doesn't always emit
page items in reading order. On some pages it emits footnotes before body text,
even though the footnotes are physically at the bottom. The footnote classifier
uses a propagation rule: once it sees one footnote, it sweeps everything after
it on that page as a footnote too. If footnotes arrive first due to emission
order, the entire body text of a page disappears.

The fix is to sort each page's items by their bounding-box Y coordinate before
any classification runs. Body text, which sits at the top of the page, always
arrives before footnotes at the bottom — regardless of how the parser emitted
them. For two-column layouts where a Y-only sort would interleave columns, the
sort is skipped.

This is a feedforward control. It doesn't react to model output. It shapes the
input so that the downstream rules can make correct decisions.

**Front matter and index detection.** The pipeline now auto-detects where the
book's actual content begins and ends, skipping the table of contents, index
pages, and prefatory material. Again: no LLM involvement. A position gate (the
signal must appear in the final 30% of the book to count as an index) and a
header scan. Deterministic and fast.

### Layer 2: After the LLM Runs (Feedback Sensors)

The second layer validates what the model returns and corrects the systematic
mistakes.

**`_restore_valid_words` — accumulated edge cases.** This function was
introduced in Part 2. Over this branch it grew considerably as real books
surfaced new failure modes.

Each new case followed the same pattern: observe a systematic mistake in real
output, write a failing unit test that reproduces it, implement the fix, verify
nothing else broke.

The cases added on this branch:

- **Curly/smart quotes in N→1 merges.** The model merges tokens and wraps the
  result in typographic quotes. The validation strips ASCII quotes but not
  curly ones, so `'justify'` fails the word check. Fix: extend the strip string
  to include `''""`.

- **Line-break hyphen joins.** The model correctly rejoins `reexamina-` +
  `tion` → `reexamination`. The validator rejects it because neither fragment
  is a real word. Fix: strip trailing hyphens from each original token,
  concatenate, and check if the result matches the cleaned token.

- **3→1 dash upgrades.** The model joins three tokens — `relationship--` +
  `-` + `whose` — into `relationship—instantiation—whose`. Double OCR hyphens
  on both sides. Fix: normalize `--` to `-` before the comparison on both
  sides.

- **Single-letter fragments.** A lone `p` or `c` is technically a valid English
  word (they're in the corpus). But as a standalone token in a paragraph it's
  almost always an OCR fragment, not the word "c". Fix: single-letter tokens
  trigger an LLM call instead of passing the pre-flight gate, and they don't
  trigger word restore on their own.

- **Backslash-escaped quotes.** The JSON repair step converts `\'word\'` to a
  Python string containing a literal backslash plus quote. The strip removes
  the quote character but not the backslash, so `\'possibility\` doesn't match
  `possibility` and the restore fires on a valid output. Fix: collapse `\'` →
  `'` and `\"` → `"` before the comparison strip on both sides.

- **LLM-introduced em-dashes.** The model sometimes rejoins `['justi',
  'fication', 'is']` as `justification—is` — correctly fixing the word break
  and inserting an em-dash in one step. The existing em-dash upgrade check
  fails because the joined originals (`justificationis`) don't match the
  normalized cleaned token (`justification-is`). Fix: split the cleaned token
  on em/en-dashes, check that each part is a valid word, and accept the merge
  if they are.

None of these fixes required touching the model or the prompt. Each one is a
mechanical rule that intercepts a failure pattern the model produces
systematically.

### Layer 3: When the Model Can't Even Respond Correctly (Resilience)

The third layer handles the cases where the model's output isn't just wrong —
it's unparseable.

**Stray backslash repair.** The model occasionally includes a bare `\alpha` or
`\mu` inside a JSON string value. `\a` is not a valid JSON escape sequence, so
`json.loads` fails. Fix: before retrying, escape any backslash not already part
of a valid JSON escape sequence and re-parse inline. No retry consumed.

**Unescaped inner quotes.** The model occasionally writes something like:

```json
{"cleaned": "the word "probability" is used", "classification": "body"}
```

The inner unescaped `"` terminates the JSON string early. The parser raises
`"Expecting ',' delimiter"` — the existing escape-repair check looks for the
string `"escape"` in the error message, doesn't find it, and lets all three
retries fail identically. The paragraph falls back to raw OCR text.

Fix: a greedy regex fallback that doesn't parse JSON at all:

```python
m = re.search(
    r'"cleaned"\s*:\s*"(.*)"\s*,\s*"classification"\s*:\s*"(\w+)"',
    content, re.DOTALL)
```

The greedy `.*` naturally expands to the last `"` before `,"classification"`,
which is exactly where the cleaned value ends — inner quotes and all. If the
regex finds no match, the error is re-raised and the retry loop continues. If
it matches, the values are extracted without consuming a retry.

## This Is a Harness

ThoughtWorks published a taxonomy of harness controls organized along two axes:
feedforward vs. feedback, and computational vs. inferential.

Almost everything in this pipeline falls into one quadrant: **computational
feedback**. Fast, deterministic rules that run after the model returns output
and correct what's wrong. No second LLM. No semantic analysis. Just code.

That's not a limitation. It's appropriate. The model's failure modes in this
application are structural — wrong quote characters, misformed JSON, predictable
token comparison edge cases. Structural problems have structural solutions.

The only inferential layer in the pipeline is the model itself. Everything else
is old-fashioned software engineering: conditionals, regular expressions, word
lists, bounding-box arithmetic.

## The Uncomfortable Part

There's a concept in harness engineering called "harness decay." Every component
in a harness encodes an assumption about what the model can't do. As models
improve, those assumptions expire. The component that was necessary last month
becomes overhead this month.

Some of what's in this pipeline has an expiry date.

The JSON repair logic exists because the model can't reliably produce valid JSON.
When that gets fixed in a future model version, both repair mechanisms become
dead weight — tokens burned on every call for zero benefit. The backslash-escaped
quote fix in `_restore_valid_words` exists because the JSON repair itself
introduces backslashes. If the upstream problem goes away, so does the downstream
symptom.

The right response to this, per Philipp Schmid at Hugging Face, is "build to
delete." Design every harness component so it can be removed and tested in
isolation. Turn it off, measure whether output quality changes, and delete it
if it doesn't. Don't carry dead harness components out of attachment to the
engineering that went into them.

For now, all of these components earn their keep. The unit test suite that
covers each one makes it straightforward to identify when they stop earning it.

## Where This Leaves Things

The pipeline after this branch is more reliable than it was before — not because
the model got better, but because the machinery around it got better at catching
and correcting its systematic failures.

The breakdown of work is instructive. Of roughly 30 commits on this branch:

- A handful added new user-facing features (index skipping, math symbol
  substitution, front matter detection).
- A handful improved the footnote classification heuristics.
- The majority were fixes to `_restore_valid_words`, the JSON repair layer, or
  the pre-processing sort — each one responding to a failure pattern observed
  in a real book run.

The model did the text cleaning. The harness did almost everything else.

That ratio will probably hold. The model handles the cases that are too
irregular for rules — genuine OCR ambiguity, context-dependent classification,
broken words that need linguistic reasoning. The rules handle the cases that
are too systematic to leave to the model — quote normalization, JSON
malformation, token comparison edge cases, reading-order bugs in the parser.

That division of labor seems about right. The interesting question, which this
branch didn't fully answer, is whether the harness will keep growing with each
book run or whether it's approaching something like a stable set of rules. The
failure modes discovered so far have followed a power-law distribution: the
most common ones showed up early, the recent additions required increasingly
specific inputs to trigger. That's an encouraging sign. But a new book is a new
surface, and new surfaces have a way of finding the gaps.
