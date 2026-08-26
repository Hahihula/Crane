#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mwparserfromhell"]
# ///
"""Extract a German word -> IPA lexicon from a German Wiktionary XML dump.

The output is `word<TAB>ipa` lines (no header, no slashes), matching
`crane-core/src/models/g2p/lexicon.rs`'s `Lexicon::from_tsv` format. A word
with multiple pronunciations produces multiple lines with the same word.

Examples:
    # Download the latest dump and extract into de_wiktionary_ipa.tsv
    ./extract-de-wiktionary-ipa.py --output de_wiktionary_ipa.tsv

    # Use an already-downloaded dump (preferred for a reproducible build)
    ./extract-de-wiktionary-ipa.py --dump-path dewiktionary-20260804-pages-articles.xml.bz2 \\
        --output de_wiktionary_ipa.tsv

Each run also writes `<output>.PROVENANCE.md` recording the dump URL/date
used and the CC BY-SA attribution chain (Wiktionary contributors -> this
script), since the extracted data is a derivative of Wiktionary content and
must carry that attribution forward.

If interrupted (Ctrl-C, crash, connection drop), re-running the exact same
command resumes: progress is checkpointed to `<output>.checkpoint.json`
every `--checkpoint-every` pages, alongside a partial `<output>` that doubles
as the recovered entry set, so nothing needs to be held only in memory. This
is a separate, coarser cadence than `--progress-every`'s log line: a
checkpoint re-sorts and rewrites the whole (multi-hundred-thousand-line)
output file, so it deliberately doesn't run on every progress tick.
"""

import argparse
import bz2
import html
import json
import re
import signal
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mwparserfromhell

LATEST_DUMP_URL = (
    "https://dumps.wikimedia.org/dewiktionary/latest/"
    "dewiktionary-latest-pages-articles.xml.bz2"
)

LAUTSCHRIFT_TEMPLATE_NAMES = {"Lautschrift", "Lautschrift?"}
IPA_MARKER_RE = re.compile(r"^:\{\{IPA\??\}\}")
LEVEL2_HEADING_RE = re.compile(r"^==([^=].*)==\s*$")
ZERO_WIDTH_RE = re.compile("[\u2060\u200b]")  # word joiner, zero-width space


def strip_tag_ns(tag):
    """Strips a MediaWiki export XML namespace prefix from an element tag."""
    return tag.rsplit("}", 1)[-1]


def iter_dump_pages(dump_path):
    """Streams (title, ns, text) tuples from a MediaWiki export XML dump.

    Transparently decompresses `.bz2` input. Uses `iterparse` with
    element-clearing so memory stays bounded regardless of dump size (a
    German Wiktionary dump has several million `<page>` elements).
    """
    opener = bz2.open if str(dump_path).endswith(".bz2") else open
    with opener(dump_path, "rt", encoding="utf-8") as handle:
        title = None
        ns = None
        text = None
        in_revision = False
        for event, elem in ET.iterparse(handle, events=("start", "end")):
            tag = strip_tag_ns(elem.tag)
            if event == "start":
                if tag == "revision":
                    in_revision = True
                continue
            if tag == "title":
                title = elem.text or ""
            elif tag == "ns":
                ns = elem.text or ""
            elif tag == "text" and in_revision:
                text = elem.text or ""
            elif tag == "revision":
                in_revision = False
            elif tag == "page":
                if title is not None and ns is not None:
                    yield title, ns, text or ""
                title, ns, text = None, None, None
                elem.clear()


def is_deutsch_heading(heading_text):
    """Returns whether a level-2 heading marks a German-language word entry.

    German Wiktionary headings look like `== Haus ({{Sprache|Deutsch}}) ==`.
    """
    for template in mwparserfromhell.parse(heading_text).filter_templates():
        if str(template.name).strip() != "Sprache":
            continue
        if template.params and str(template.params[0].value).strip() == "Deutsch":
            return True
    return False


def clean_ipa_value(raw):
    """Normalizes one `{{Lautschrift|...}}` template argument into plain IPA text."""
    value = html.unescape(html.unescape(raw)).strip()
    return ZERO_WIDTH_RE.sub("", value)


def is_valid_ipa(value):
    """Rejects placeholder/malformed IPA values.

    Covers: missing, leading "-", ellipsis, stray template braces, a literal
    "/" (either leftover IPA-slash delimiters an editor typed into the
    template by mistake, e.g. `{{Lautschrift|/ˈvɪl.jam/}}`, or two alternate
    readings packed into one value with "/" instead of separate templates —
    neither is parseable as a single pronunciation), and stress-mark-only
    stubs (e.g. `{{Lautschrift|ˈ|spr=des}}`, a placeholder some Wiktionary
    entries use when nobody has filled in the actual pronunciation yet).
    """
    if not value or value.startswith("-") or "…" in value or "{" in value or "}" in value or "/" in value:
        return False
    return value.strip("ˈˌ") != ""


def extract_lautschrift_values(line):
    """Yields every valid IPA value from `{{Lautschrift|...}}` templates on one wikitext line."""
    for template in mwparserfromhell.parse(line).filter_templates():
        if str(template.name).strip() not in LAUTSCHRIFT_TEMPLATE_NAMES:
            continue
        if not template.has("1"):
            continue
        value = clean_ipa_value(str(template.get("1").value))
        if is_valid_ipa(value):
            yield value


def extract_pronunciations(wikitext):
    """Extracts every German-section IPA pronunciation from one page's wikitext.

    Walks the page line by line rather than parsing the whole page as one
    wikicode tree: German Wiktionary's `Aussprache`/`Worttrennung`/etc.
    subsections are plain bare template calls on their own line (not `===`
    headings), so a line-oriented state machine is the natural fit, mirroring
    how `devio-at/german-ipa-dict`'s own extractor is structured. Unlike that
    script, this one doesn't capture the regional-variant prose around each
    IPA value, since `Lexicon::from_tsv` has no field for it.
    """
    in_de_section = False
    pron_found = False
    in_ipa = False
    results = []

    for line in wikitext.splitlines():
        stripped = line.strip()

        heading_match = LEVEL2_HEADING_RE.match(stripped)
        if heading_match:
            in_de_section = is_deutsch_heading(heading_match.group(1))
            pron_found = False
            in_ipa = False
            continue

        if not in_de_section:
            continue

        if stripped == "{{Aussprache}}":
            pron_found = True
            in_ipa = False
            continue

        if stripped.startswith("{{"):
            # A different bare pseudo-heading (e.g. {{Bedeutungen}}) ends the
            # Aussprache block.
            pron_found = False
            in_ipa = False
            continue

        if not pron_found:
            continue

        if IPA_MARKER_RE.match(stripped):
            in_ipa = True
        elif in_ipa and not stripped.startswith("::"):
            in_ipa = False

        if not in_ipa or "{{Lautschrift" not in stripped:
            continue

        results.extend(extract_lautschrift_values(stripped))

    return results


def is_eligible_title(title):
    """Filters out non-lemma pages: namespaced pages, suffix stubs, and multi-word phrases.

    Multi-word phrase entries (e.g. "Haus und Hof") are real Wiktionary
    pages but aren't G2P lookup targets, since Crane's lexicon is only ever
    queried one token at a time (see `text_normalize::split_text_to_words`)
    — keeping them would just be dead weight in the lexicon.
    """
    return ":" not in title and not title.startswith("-") and " " not in title


def download_dump(url, dest):
    """Downloads a dump file to `dest`, printing simple progress to stderr."""
    print(f"Downloading {url} -> {dest}", file=sys.stderr)
    # Wikimedia's dump server 403s requests with no User-Agent (bot policy).
    request = urllib.request.Request(url, headers={"User-Agent": "crane-g2p-dict-builder/1.0"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        with open(dest, "wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read / 1e6:.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)


def dump_identity(dump_path):
    """Returns a cheap fingerprint of a dump file, used to detect a stale checkpoint."""
    path = Path(dump_path)
    return {"dump_path": str(path), "size": path.stat().st_size}


def load_checkpoint(checkpoint_path, output_path, dump_path):
    """Loads (pages_already_scanned, entries) from a prior interrupted run.

    Progress is recovered from two files: the checkpoint JSON (page count +
    dump fingerprint) and the partial output TSV itself, which doubles as
    the entry set — so a resumed run never has to re-hold the whole result
    in memory from scratch. Falls back to a fresh start if either file is
    missing or the checkpoint was made against a different dump (page N
    means something different in a different dump snapshot).
    """
    if not checkpoint_path.exists() or not output_path.exists():
        return 0, set()

    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if state.get("dump") != dump_identity(dump_path):
        print("Checkpoint is for a different dump file; starting over.", file=sys.stderr)
        return 0, set()

    entries = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            word, ipa = line.split("\t", 1)
            entries.add((word, ipa))
    return state["pages_scanned"], entries


def save_checkpoint(checkpoint_path, output_path, dump_path, pages_scanned, entries):
    """Atomically flushes progress: rewrites the output TSV, then the checkpoint marker.

    Writing the output before the checkpoint means a crash between the two
    writes is safe — worst case, a resumed run re-derives the same
    `pages_scanned` count and re-flushes the same entries, it never resumes
    from a checkpoint that claims more progress than what's on disk.
    """
    tmp_output = Path(str(output_path) + ".tmp")
    with open(tmp_output, "w", encoding="utf-8") as f:
        f.writelines(f"{word}\t{ipa}\n" for word, ipa in sorted(entries))
    tmp_output.replace(output_path)
    checkpoint_path.write_text(
        json.dumps({"pages_scanned": pages_scanned, "dump": dump_identity(dump_path)}),
        encoding="utf-8",
    )


def write_provenance(path, dump_source):
    """Writes the CC BY-SA attribution/provenance note accompanying the output TSV."""
    path.write_text(
        "# Provenance and license\n\n"
        "This word-to-IPA lexicon was extracted from the German-language\n"
        "edition of Wiktionary (https://de.wiktionary.org) using\n"
        "`data/g2p/extract-de-wiktionary-ipa.py` in the Crane repository,\n"
        "pulling the `{{Lautschrift}}` pronunciation template out of each\n"
        "German (`{{Sprache|Deutsch}}`) entry's `{{Aussprache}}` section.\n\n"
        f"Dump source: {dump_source}\n\n"
        "## License\n\n"
        "Wiktionary text content is dual-licensed under Creative Commons\n"
        "Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) and the\n"
        "GNU Free Documentation License (GFDL). This extracted dataset is a\n"
        "derivative work and is therefore likewise licensed\n"
        "**CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/)\n"
        "— redistributing or adapting it must preserve attribution to\n"
        "Wiktionary contributors and stay under a compatible share-alike\n"
        "license. It is not MIT-licensed.\n\n"
        "Attribution: Wiktionary contributors, https://de.wiktionary.org,\n"
        "extracted via `de.wiktionary.org`'s public XML dump.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-path", type=Path, help="Path to a local dewiktionary-*-pages-articles.xml(.bz2) dump")
    parser.add_argument(
        "--download-to",
        type=Path,
        default=Path("dewiktionary-latest-pages-articles.xml.bz2"),
        help="Where to save the dump if --dump-path is not given and it needs downloading (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path (word<TAB>ipa lines)")
    parser.add_argument(
        "--progress-every", type=int, default=100_000, help="Print a progress line every N pages scanned"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500_000,
        help="Re-sort and flush the output + checkpoint file every N pages scanned (default: %(default)s)",
    )
    args = parser.parse_args()

    dump_source = str(args.dump_path) if args.dump_path else LATEST_DUMP_URL
    dump_path = args.dump_path
    if dump_path is None:
        dump_path = args.download_to
        if not dump_path.exists():
            download_dump(LATEST_DUMP_URL, dump_path)
        else:
            print(f"Using already-downloaded {dump_path}", file=sys.stderr)

    checkpoint_path = Path(str(args.output) + ".checkpoint.json")
    resume_from, entries = load_checkpoint(checkpoint_path, args.output, dump_path)
    if resume_from:
        print(f"Resuming: {resume_from} pages already scanned, {len(entries)} entries loaded", file=sys.stderr)

    pages_scanned = 0
    german_pages = 0

    # A process manager (or `timeout`) stops a long-running job with SIGTERM,
    # not Ctrl-C's SIGINT — handle both the same way: set a flag rather than
    # raising immediately, so the loop only ever stops *between* fully-
    # processed pages. Raising asynchronously (e.g. via KeyboardInterrupt)
    # could land mid-page, after `pages_scanned` was incremented but before
    # that page's entries were added — checkpointing at that instant would
    # mark the page "done" while silently dropping the entries it hadn't
    # gotten to yet.
    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    def flush(final):
        save_checkpoint(checkpoint_path, args.output, dump_path, pages_scanned, entries)
        if final:
            checkpoint_path.unlink(missing_ok=True)
            write_provenance(Path(str(args.output) + ".PROVENANCE.md"), dump_source)

    for title, ns, text in iter_dump_pages(dump_path):
        pages_scanned += 1
        if pages_scanned <= resume_from:
            continue

        # Process this page fully *before* printing/checkpointing/stopping on
        # it, so a checkpoint claiming "page N scanned" always has page N's
        # own contribution folded into `entries` already.
        if ns == "0" and is_eligible_title(title):
            pronunciations = extract_pronunciations(text)
            if pronunciations:
                german_pages += 1
                for ipa in pronunciations:
                    entries.add((title, ipa))

        if args.progress_every and pages_scanned % args.progress_every == 0:
            print(f"  scanned {pages_scanned} pages, {len(entries)} entries so far", file=sys.stderr)

        if stop_requested or (args.checkpoint_every and pages_scanned % args.checkpoint_every == 0):
            flush(final=False)
            if stop_requested:
                print(f"Stopped after {pages_scanned} pages; re-run the same command to resume.", file=sys.stderr)
                return

    flush(final=True)

    print(
        f"Scanned {pages_scanned} pages, {german_pages} German entries with "
        f"pronunciations, wrote {len(entries)} word/IPA lines to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
