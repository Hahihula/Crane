# G2P Test Fixtures

Held-out word-to-IPA test sets for evaluating grapheme-to-phoneme accuracy.
These are static test fixtures checked into the repository, not runtime
model assets.

## Files

| File | Language | Entries | Description |
|------|----------|---------|--------------|
| `en_us_test.tsv` | English (US) | 5000 | Held-out word-IPA pairs for CER benchmarking |

## Format

Tab-separated, two columns, UTF-8, no header row, sorted alphabetically:

```
word<TAB>IPA
```

Example:

```
abandons	əbˈændənz
```

## Provenance

`en_us_test.tsv` is a 5000-entry random sample (seed 42, no replacement)
from Moonshine Voice's (https://github.com/moonshine-ai/moonshine, MIT
license) English IPA lexicon, which is itself derived from the CMU
Pronouncing Dictionary (https://github.com/cmusphinx/cmudict, BSD-style
license). These entries are held out and never seen by the G2P engine
during lexicon construction.

## License

Both upstream sources use permissive, MIT-compatible licenses:

- Moonshine Voice: MIT license.
- CMU Pronouncing Dictionary: BSD-style license, Copyright (C) 1993-2015
  Carnegie Mellon University. Full text:
  https://github.com/cmusphinx/cmudict/blob/master/LICENSE
