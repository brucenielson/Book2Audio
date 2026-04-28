# Keeping the LLM Honest

In Part 1, I built a gate that skips the LLM for paragraphs that are already
clean. That saved most of the LLM calls. But it raised a sharper question: for
the paragraphs that *do* go to the LLM, can I trust what comes back?

Not entirely. That turned out to be the interesting problem.

## What the LLM Does When You're Not Looking

The LLM's job is narrow: fix OCR artifacts, rejoin broken words, remove stray
footnote markers. It is not supposed to edit the text.

It edits the text anyway.

A paragraph about "judiciary powers" comes back as "judicial powers." An
em-dash (`—`) gets quietly downgraded to a plain hyphen. A numbered list item
like "(3) All human actions are egotistic" loses its prefix entirely. ASCII
quotes get upgraded to typographic curly quotes — which sounds like an
improvement until you realise the preprocessor already normalised everything to
ASCII and now your comparison logic is broken.

None of this is malicious. The LLM is a probabilistic text completer. It
doesn't have a notion of "only fix OCR." It has a notion of "what would a
well-edited version of this text look like?" Those are different things, and
the gap between them is where the problems live.

## The Guard Layer: `_restore_valid_words`

The fix is a post-processing step that runs after the LLM returns its output.
It compares the original and cleaned text token by token using `SequenceMatcher`
and silently undoes any substitution where the original word was valid English.

The logic for 1:1 replacements is straightforward:

```python
orig_stripped = normalize_quotes(original_lower[i1+k]).strip('.,;:!?"\'()-[]')
new_stripped  = normalize_quotes(cleaned_lower[j1+k]).strip('.,;:!?"\'()-[]')

if orig_stripped != new_stripped and word_validator.is_valid_word(orig_stripped):
    result.append(original_split[i1+k])   # restore
else:
    result.append(cleaned_split[j1+k])    # keep LLM version
```

This lets OCR fixes through. If the original token is invalid (an OCR artifact)
and the LLM replaced it with a valid word, we keep the fix. If the original
token is valid and the LLM swapped it for something else, we restore the
original. The LLM's other changes — joined word breaks, removed footnote
markers — survive untouched.

## The Harder Case: N→1 Merges

The 1:1 case is clean. The harder case is when the LLM collapses multiple
tokens into one. This happens legitimately:

- `"Scienti fic"` → `"Scientific"` (OCR word break)
- `"i. e.,"` → `"i.e.,"` (spaced abbreviation)
- `"Ph. D"` → `"Ph.D."` (abbreviated title)
- `"proof reading"` → `"proof-reading"` (hyphen compounding)
- `"criticism - and"` → `"criticism—and"` (em-dash upgrade)

It also happens illegitimately. You can't just accept all merges.

The solution is to enumerate the cases where a merge is safe to keep:

```python
merged = _normalize_dashes(cleaned_lower[j1].strip('.,;:!?"\'()-[]'))
joined_orig = ''.join(original_split[i1:i2])

if (word_validator.is_valid_word(merged)          # "Scientific"
        or any(c.isdigit() for c in merged)       # "1959"
        or cleaned_split[j1] == joined_orig       # "i.e.," == "i.e.,"
        or '.' in cleaned_inner                   # internal period → abbreviation
        or (_normalize_dashes(cleaned_split[j1]) == joined_orig
            and original_split[i1][0].isalpha())  # em-dash upgrade
        or '-'.join(original_split[i1:i2]) == cleaned_split[j1]):  # hyphenation
    result.append(cleaned_split[j1])
else:
    result.extend(original_split[i1:i2])  # restore originals
```

The `original_split[i1][0].isalpha()` guard on the em-dash case is there to
prevent `"- including"` → `"—including"` (a standalone dash glued to the next
word) from being accepted. The content is the same after normalisation, but
gluing is not the same as upgrading.

Every one of these conditions was written because a real case from a real book
broke without it.

## The Invalid→Invalid Question

Originally, the pipeline also rejected paragraphs when the LLM replaced one
unrecognisable token with a different unrecognisable token. The reasoning:
if neither the original nor the replacement is in the English word list, the
LLM didn't fix anything — it just guessed differently.

This sounds right until you run it against a book with a German title.

A word like `"heiden"` (an OCR misread of the German `"beiden"`) fails the
English word test both before and after whatever the LLM does with it. The
whole paragraph then fails and reverts to the uncleaned original — including
all the legitimate fixes the LLM made to the other thirty words in that
paragraph.

The question is: what do you lose by trusting the LLM on invalid→invalid? Not
much. The LLM is already doing its best. If it can't identify the correct word,
its guess is at least plausible given the context. Rejecting the whole paragraph
over one unresolvable token is a poor trade.

So we removed the check. The function that used to scan for suspicious
substitutions now always returns `False`. It's kept as a named hook in case
we want to add a different check later, but the invalid→invalid rejection is
gone.

## Two Edge Cases Worth Naming

### Smart Quotes

The preprocessor normalises all typographic curly quotes to ASCII straight
quotes before anything reaches the LLM. The LLM then converts them back.
So the original has `'All` (ASCII U+0027) and the LLM returns `'All`
(U+2018 left single quotation mark). They look identical in any terminal.
They are not the same string.

`SequenceMatcher` sees them as different tokens. The comparison strips
punctuation from the ends, but only ASCII punctuation — so the smart quote
survives stripping on the LLM side, and `"all"` ≠ `"‘all"`. The
original `'all` strips cleanly to `"all"`, which is a valid English word.
The restore fires. Spuriously.

The fix is to call `normalize_quotes` on both tokens before stripping:

```python
orig_stripped = _normalize_dashes(
    normalize_quotes(original_lower[i1+k]).strip('.,;:!?"\'()-[]'))
new_stripped = _normalize_dashes(
    normalize_quotes(cleaned_lower[j1+k]).strip('.,;:!?"\'()-[]'))
```

Now both sides normalise to the same string and the spurious restore disappears.

### Em-Dash Downgrade

`_normalize_dashes` converts em-dashes to hyphens for comparison purposes,
so that `"false—as"` and `"false-as"` are treated as equivalent. This was
intentional: if the LLM *upgrades* a hyphen to an em-dash, that's fine, and
we don't want to undo it.

But it also means that if the LLM *downgrades* an em-dash to a hyphen —
`"false—as"` in, `"false-as"` out — the comparison sees them as equal, no
restore fires, and the em-dash is quietly lost.

The fix is an extra condition after the valid-word check:

```python
elif (orig_stripped == new_stripped
      and ('—' in orig_tok or '–' in orig_tok)
      and '—' not in new_tok and '–' not in new_tok):
    result.append(original_split[i1+k])  # restore the em-dash
```

If the tokens are equal after normalisation but the original had an em or
en-dash and the cleaned version doesn't, the original wins. Upgrade is kept.
Downgrade is reverted.

## Writing the Tests First

Every one of these cases was a failing unit test before it was a code change.
This is not a pious observation about software engineering practice. It's
practical: the test is the spec. Writing `assert result == "false—as"` before
touching the implementation forces you to think clearly about what the correct
behaviour actually is, rather than working backwards from whatever the code
happens to produce.

It also catches interactions. The em-dash downgrade fix had to be written
carefully to avoid breaking the upgrade test that was already passing. Had the
implementation come first, that interaction would have been easy to miss.

## Where This Leaves Things

The pipeline now has a real guard layer between the LLM and the output. Most
legitimate LLM fixes go through. Most unauthorised changes are reverted. The
remaining edge cases — and there will be more; a full book run keeps surfacing
them — get handled one failing test at a time.

What the guard layer cannot do is tell the LLM to do a better job in the first
place. That's a different problem. For now, the approach is: accept probabilistic
output, verify it, and correct what can be corrected mechanically. The LLM
handles the cases that are too irregular for rules. The rules handle the cases
that are too systematic to leave to the LLM.

That division of labour seems about right. Whether it holds up at scale is
what the next run will tell us.
