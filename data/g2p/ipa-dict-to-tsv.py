#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Converts an open-dict-data/ipa-dict wordlist into Crane's G2P lexicon TSV format.

Upstream: https://github.com/open-dict-data/ipa-dict

ipa-dict's raw `data/<lang>.txt` files are `word<TAB>value` lines, where
`value` is one or more `/`-delimited IPA alternatives separated by ", "
(e.g. `a	/ˈeɪ/, /ə/`). This script strips the slash delimiters and expands
each alternative onto its own line, producing the `word<TAB>ipa` format
`crane-core/src/models/g2p/lexicon.rs`'s `Lexicon::from_tsv` expects.

Each ipa-dict language file has its own license and attribution (see
ipa-dict's own README credits section) rather than one blanket license for
the whole repository, so `--license` and `--attribution` are required
arguments here rather than an assumed default -- they are recorded in
`<output>.PROVENANCE.md` alongside the TSV.

Example:
    ./ipa-dict-to-tsv.py --input ../.tmp/ipa-dict/data/en_US.txt \\
        --output en_us.tsv \\
        --license MIT \\
        --attribution "English (US) IPA data based on a modified version of \\
cmudict-ipa by @lingz, with stress markers added via syllabify by \\
@kylebgorman (MIT), via open-dict-data/ipa-dict."
"""

import argparse
import sys
from pathlib import Path


def parse_ipa_dict_line(line):
    """Splits one ipa-dict `word<TAB>value` line into `(word, [ipa, ...])`.

    ipa-dict values are one or more `/`-delimited alternatives joined by
    ", " (e.g. `/ˈeɪ/, /ə/`). Splitting on ", " first and stripping `/`
    from each piece afterwards also handles the rarer case of a single
    slash pair containing an internal comma (e.g. `en_UK.txt`'s
    `the	/ðə, ði/`), since `str.strip` only trims a piece's own
    leading/trailing characters.
    """
    word, raw_value = line.split("\t", 1)
    alternatives = []
    for piece in raw_value.split(", "):
        ipa = piece.strip().strip("/")
        if ipa:
            alternatives.append(ipa)
    return word, alternatives


def convert(input_path):
    """Yields every `(word, ipa)` pair in an ipa-dict file, one per alternative."""
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                print(f"{input_path}:{line_num}: missing tab, skipping: {line!r}", file=sys.stderr)
                continue
            word, alternatives = parse_ipa_dict_line(line)
            for ipa in alternatives:
                yield word, ipa


def write_provenance(path, input_path, license_name, attribution):
    """Writes the license/attribution note accompanying the output TSV."""
    path.write_text(
        "# Provenance and license\n\n"
        f"This word-to-IPA lexicon was converted from open-dict-data/ipa-dict's\n"
        f"`{input_path.name}` using `data/g2p/ipa-dict-to-tsv.py` in the Crane\n"
        "repository: each `/`-delimited IPA alternative is expanded onto its\n"
        "own line, matching this project's lexicon TSV format.\n\n"
        "## License\n\n"
        f"**{license_name}**. {attribution}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Path to an ipa-dict data/<lang>.txt file")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path (word<TAB>ipa lines)")
    parser.add_argument(
        "--license",
        required=True,
        help='Short license identifier for this language file, e.g. "MIT" or "CC BY-SA 4.0" '
        "(see ipa-dict's README credits section -- each language has its own, not one blanket license)",
    )
    parser.add_argument(
        "--attribution",
        required=True,
        help="Attribution text for this language, copied from ipa-dict's README credits section",
    )
    args = parser.parse_args()

    entries = set(convert(args.input))
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(f"{word}\t{ipa}\n" for word, ipa in sorted(entries))

    write_provenance(Path(str(args.output) + ".PROVENANCE.md"), args.input, args.license, args.attribution)

    print(f"Wrote {len(entries)} word/IPA lines to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
