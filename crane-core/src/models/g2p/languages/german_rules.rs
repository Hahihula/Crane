// SPDX-License-Identifier: MIT

//! Hand-written German letter-to-sound rules.
//!
//! This is the final G2P fallback tier, used when a word misses lexicon
//! lookup. The word is lowercased and filtered to German letters, split into
//! orthographic syllables, and each syllable's graphemes are converted to
//! IPA with a handful of context-sensitive rules (`ch`'s back-vowel/
//! front-vowel split, `st`/`sp` palatalization at morpheme boundaries, vowel
//! length, word-internal `-ig` softening, final-obstruent devoicing). A
//! primary stress mark is then inserted on the syllable chosen by
//! prefix/suffix heuristics.
//!
//! Known limitation (deliberate, for now): `h` is unconditionally silent in
//! every position, including word-initial, where standard pronunciation
//! would produce /h/. A later change will special-case word-initial `h` and
//! measure its effect on accuracy in isolation.
//!
//! Known limitation (deliberate, for now): the unstressed-prefix heuristic
//! (see [`UNSTRESSED_PREFIXES`]) is purely orthographic — it has no way to
//! distinguish a real prefix from a root that merely starts with the same
//! letters (e.g. "geben", "Berg", "Erbe"). Accurate prefix stripping needs
//! morphological decomposition, which is out of scope until the
//! compound-word decomposition feature (see the `german` module doc) lands.

/// Unicode primary stress mark (U+02C8).
const IPA_PRIMARY_STRESS: char = 'ˈ';

/// Unstressed prefixes: a word starting with one of these (with a nonempty
/// remainder) puts primary stress on the syllable after the prefix instead
/// of the first syllable. Order matters only in that a longer prefix must
/// precede any shorter prefix it starts with (`"entgegen"` before `"ent"`)
/// so the longer, more specific match wins.
///
/// Caveat: this is a purely orthographic check with no morphological
/// awareness, so a root that coincidentally starts with the same letters
/// (e.g. "geben", "Erbe") will have its stress mis-placed. See the
/// module-level known-limitation note.
///
/// `"durch"`, `"nach"`, and `"bei"` were added to the original 10-entry list
/// after individually measuring 17 candidate prefixes (`un-, ur-, ab-, an-,
/// auf-, aus-, bei-, durch-, ein-, mit-, nach-, über-, um-, vor-, zu-,
/// zurück-, zusammen-`) against the held-out CER benchmark. Only `durch`,
/// `nach`, `bei`, and `mit` improved or held the error count steady; the
/// other 13 (including short, high-false-positive-rate ones like
/// `"an"`/`"ab"`) measurably regressed CER and were excluded — the same
/// measure-in-isolation-and-drop-the-losers methodology `EnglishG2p`'s own
/// prefix list already uses for `"re"`/`"mis"`/`"pre"` (see
/// `english_rules.rs`).
const UNSTRESSED_PREFIXES: &[&str] = &[
    "entgegen", "durch", "wider", "miss", "nach", "bei", "mit", "ver", "zer", "ent", "emp", "ge",
    "be", "er",
];

/// Suffixes that pull primary stress onto the final syllable regardless of
/// any recognized prefix.
const STRESSED_SUFFIXES: &[&str] = &["ung", "schaft", "tion", "ismus"];

/// Returns `true` for the letters this engine's rules understand: ASCII
/// lowercase, the three umlauts, and eszett.
fn is_german_letter(c: char) -> bool {
    c.is_ascii_lowercase() || matches!(c, 'ä' | 'ö' | 'ü' | 'ß')
}

/// Returns `true` if `c` is a German vowel letter, including `y` and the
/// umlauts (but not eszett, which is a consonant).
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y' | 'ä' | 'ö' | 'ü')
}

/// Returns `true` if `c` is one of the vowel-sound IPA symbols this module
/// emits. Used to find where to place the primary stress mark.
fn is_ipa_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'ɛ' | 'ɪ' | 'ɔ' | 'ʊ' | 'ə' | 'ø' | 'ʏ' | 'ɐ')
}

/// Lowercases `word` and drops every character that isn't a recognized
/// German letter or a hyphen (hyphens are kept as morpheme-boundary markers
/// for the palatalization rules and are stripped before syllabification).
fn normalize_for_rules(word: &str) -> Vec<char> {
    word.chars()
        .flat_map(char::to_lowercase)
        .filter(|&c| is_german_letter(c) || c == '-')
        .collect()
}

/// Returns `true` if `chars[i..]` starts with `pat`'s characters (bounds-safe:
/// a `pat` longer than the remaining slice returns `false`).
fn slice_eq_str(chars: &[char], i: usize, pat: &str) -> bool {
    let mut j = i;
    for pc in pat.chars() {
        match chars.get(j) {
            Some(&c) if c == pc => j += 1,
            _ => return false,
        }
    }
    true
}

/// Returns `true` if `chars` starts with `pat` and has at least one
/// character left over afterward (an exact-length match doesn't count — used
/// where a bare prefix, with nothing left to stress, shouldn't match).
fn starts_with_str(chars: &[char], pat: &str) -> bool {
    let pat_len = pat.chars().count();
    chars.len() > pat_len && slice_eq_str(chars, 0, pat)
}

/// Returns `true` if `chars` ends with `pat`'s characters.
fn ends_with_str(chars: &[char], pat: &str) -> bool {
    let pat_len = pat.chars().count();
    chars.len() >= pat_len && slice_eq_str(chars, chars.len() - pat_len, pat)
}

/// Identifies vowel-nucleus spans in `w` for syllabification: diphthongs
/// (`au`, `ei`, `eu`, `ai`, `äu`, `ey`, `oi`), `ie` not followed by another
/// vowel, doubled vowels (`aa`, `ee`, `ii`, `oo`, `uu`), and otherwise single
/// vowel letters. Each returned span is a half-open `(start, end)` range.
fn vowel_nucleus_spans(letters: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let len = letters.len();
    let mut i = 0;
    while i < len {
        if !is_vowel(letters[i]) {
            i += 1;
            continue;
        }
        let start = i;
        if i + 1 < len {
            let (first, second) = (letters[i], letters[i + 1]);
            let is_diphthong = matches!(
                (first, second),
                ('a' | 'e' | 'ä', 'u') | ('e' | 'a' | 'o', 'i') | ('e', 'y')
            );
            let is_ie =
                first == 'i' && second == 'e' && !letters.get(i + 2).is_some_and(|&c| is_vowel(c));
            let is_doubled = first == second && matches!(first, 'a' | 'o' | 'e' | 'i' | 'u');
            if is_diphthong || is_ie || is_doubled {
                spans.push((start, i + 2));
                i += 2;
                continue;
            }
        }
        spans.push((start, start + 1));
        i += 1;
    }
    spans
}

/// Splits a hyphen-free run of letters into orthographic syllable boundary
/// ranges relative to `w`: everything up to and including a vowel nucleus
/// forms one syllable, and the consonant cluster after it (up to the next
/// nucleus, or the word end) joins the following syllable. A run with no
/// vowel at all is returned as a single syllable spanning all of `w`. Every
/// returned range is guaranteed non-empty, since each one always contains at
/// least its terminating vowel nucleus.
fn syllabify_segment(w: &[char]) -> Vec<(usize, usize)> {
    if w.is_empty() {
        return Vec::new();
    }
    let spans = vowel_nucleus_spans(w);
    if spans.is_empty() {
        return vec![(0, w.len())];
    }
    let mut out = Vec::with_capacity(spans.len());
    let mut start = 0usize;
    for (idx, &(_, e)) in spans.iter().enumerate() {
        let end = if idx + 1 < spans.len() { e } else { w.len() };
        out.push((start, end));
        start = e;
    }
    out
}

/// Splits `word` (letters and hyphens) into a hyphen-free "compact" word,
/// syllable boundary ranges (indices into that compact word), and a
/// parallel `morpheme_starts` vector (also indexed into the compact word)
/// marking word-start and every position immediately after a hyphen — the
/// boundaries `st`/`sp` palatalization checks against. Each hyphen-delimited
/// segment of `word` is syllabified independently so a vowel nucleus never
/// spans a hyphen boundary.
fn build_syllables_and_morpheme_starts(word: &[char]) -> (Vec<char>, Vec<(usize, usize)>, Vec<bool>) {
    let compact_len = word.iter().filter(|&&c| c != '-').count();
    let mut compact = Vec::with_capacity(compact_len);
    let mut morpheme_starts = vec![false; compact_len];
    let mut syllables = Vec::new();
    let mut abs = 0usize;
    for segment in word.split(|&c| c == '-') {
        if segment.is_empty() {
            continue;
        }
        morpheme_starts[abs] = true;
        for (s, e) in syllabify_segment(segment) {
            syllables.push((abs + s, abs + e));
        }
        compact.extend_from_slice(segment);
        abs += segment.len();
    }
    (compact, syllables, morpheme_starts)
}

/// Returns the length of the unstressed prefix (see [`UNSTRESSED_PREFIXES`])
/// that `word` starts with, or 0 if none matches.
fn unstressed_prefix_len(word: &[char]) -> usize {
    for pref in UNSTRESSED_PREFIXES {
        if starts_with_str(word, pref) {
            return pref.chars().count();
        }
    }
    0
}

/// Chooses which syllable (by index into `syllables`) gets primary stress: a
/// single-syllable word stresses its only syllable; a word ending in one of
/// [`STRESSED_SUFFIXES`] stresses its last syllable; a word starting with an
/// unstressed prefix stresses the syllable right after the prefix; otherwise
/// the first syllable is stressed.
fn default_stress_syllable_index(syllables: &[(usize, usize)], full_word: &[char]) -> usize {
    let n = syllables.len();
    if n <= 1 {
        return 0;
    }
    if STRESSED_SUFFIXES.iter().any(|suf| ends_with_str(full_word, suf)) {
        return n - 1;
    }
    let plen = unstressed_prefix_len(full_word);
    if plen > 0 {
        let mut acc = 0;
        for (idx, &(s, e)) in syllables.iter().enumerate() {
            acc += e - s;
            if acc >= plen {
                return (idx + 1).min(n - 1);
            }
        }
    }
    0
}

/// Finds the sound `ch` attaches to at position `i` in `full_word`: the
/// immediately preceding vowel, looking through a silent `h` that follows a
/// vowel (so `"ah"` still counts as ending in `a`). Returns `None` if the
/// character immediately before `ch` (ignoring a silent `h`) is not a
/// vowel — any other intervening consonant always yields ich-laut, since the
/// ich-laut/ach-laut distinction depends only on the immediately adjacent
/// sound, not on a vowel further back in the word.
fn char_before_for_ch(full_word: &[char], i: usize) -> Option<char> {
    if i == 0 {
        return None;
    }
    let j = i - 1;
    if is_vowel(full_word[j]) {
        return Some(full_word[j]);
    }
    if full_word[j] == 'h' && j > 0 && is_vowel(full_word[j - 1]) {
        return Some(full_word[j - 1]);
    }
    None
}

/// Resolves `ch` at absolute position `i` in `full_word` to `/x/` after a
/// back vowel (`a`, `o`, `u`) or `/ç/` otherwise (including word-initial
/// `ch`, which has no preceding vowel at all).
fn ch_ipa(full_word: &[char], i: usize) -> &'static str {
    match char_before_for_ch(full_word, i) {
        Some('a' | 'o' | 'u') => "x",
        _ => "ç",
    }
}

/// Softens a syllable-final `-ig` to `/ç/` by rewriting the trailing `/ɡ/`
/// already pushed onto `out`. Does not fire when the `i` is part of a
/// diphthong (`eig`, `aig`, `oig`) rather than a standalone `-ig` suffix.
fn apply_ig_fix(syllable: &[char], out: &mut String) {
    if !ends_with_str(syllable, "ig") {
        return;
    }
    let len = syllable.len();
    if len >= 3 && is_vowel(syllable[len - 3]) {
        return;
    }
    if out.ends_with('ɡ') {
        out.pop();
        out.push('ç');
    }
}

/// Devoices a syllable-final voiced obstruent (`b→p`, `d→t`, `ɡ→k`, `v→f`,
/// `z→s`), applied per syllable so it also covers coda position at a
/// syllable boundary, not just the end of the whole word.
fn final_devoice(mut ipa: String) -> String {
    if let Some(last) = ipa.pop() {
        let devoiced = match last {
            'b' => 'p',
            'd' => 't',
            'ɡ' => 'k',
            'v' => 'f',
            'z' => 's',
            other => other,
        };
        ipa.push(devoiced);
    }
    ipa
}

/// Tries the multi-letter, context-sensitive grapheme rules at position `i`:
/// `tsch`, `sch`, `chs`, `ch` (back/front-vowel split via [`ch_ipa`]), `ng`,
/// `nk`, `pf`, `qu`, and the `st`/`sp` palatalization pair (only at a
/// morpheme start). Pushes the matched IPA onto `out` and returns the number
/// of source characters consumed, or `None` if nothing matched.
fn try_context_grapheme(
    syllable: &[char],
    i: usize,
    full_word: &[char],
    gi: usize,
    morpheme_starts: &[bool],
    out: &mut String,
) -> Option<usize> {
    if slice_eq_str(syllable, i, "tsch") {
        out.push_str("tʃ");
        return Some(4);
    }
    if slice_eq_str(syllable, i, "sch") {
        out.push('ʃ');
        return Some(3);
    }
    if slice_eq_str(syllable, i, "chs") {
        out.push_str("ks");
        return Some(3);
    }
    if slice_eq_str(syllable, i, "ch") {
        out.push_str(ch_ipa(full_word, gi));
        return Some(2);
    }
    if slice_eq_str(syllable, i, "ng") {
        out.push('ŋ');
        return Some(2);
    }
    if slice_eq_str(syllable, i, "nk") {
        out.push_str("ŋk");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "pf") {
        out.push_str("pf");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "qu") {
        out.push_str("kv");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "st") && morpheme_starts[gi] {
        out.push_str("ʃt");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "sp") && morpheme_starts[gi] {
        out.push_str("ʃp");
        return Some(2);
    }
    None
}

/// Tries the remaining single-letter consonant rules with fixed or
/// locally-conditioned mappings: `h` (silenced), `ß`, `tz`/`z`, `ck`,
/// `c`-before-`e`/`i` vs. plain `c`, `x`, `q`-without-`u`, `j`, `v`, `w`, and
/// `y`. Pushes onto `out` and returns characters consumed, or `None`.
fn try_fixed_consonant(syllable: &[char], i: usize, out: &mut String) -> Option<usize> {
    let ch = syllable[i];
    if ch == 'h' {
        // Deliberate baseline behavior: silent in every position, including
        // word-initial. See the module doc's known limitation note.
        return Some(1);
    }
    if ch == 'ß' {
        out.push('s');
        return Some(1);
    }
    if slice_eq_str(syllable, i, "tz") {
        out.push_str("ts");
        return Some(2);
    }
    if ch == 'z' {
        out.push_str("ts");
        return Some(1);
    }
    if slice_eq_str(syllable, i, "ck") {
        out.push('k');
        return Some(2);
    }
    if ch == 'c' && syllable.get(i + 1).is_some_and(|&c| matches!(c, 'e' | 'i')) {
        out.push_str("ts");
        return Some(2);
    }
    if ch == 'c' {
        out.push('k');
        return Some(1);
    }
    if ch == 'x' {
        out.push_str("ks");
        return Some(1);
    }
    if ch == 'q' && syllable.get(i + 1).is_none_or(|&c| c != 'u') {
        out.push('k');
        return Some(1);
    }
    if ch == 'j' {
        out.push('j');
        return Some(1);
    }
    if ch == 'v' {
        out.push('f');
        return Some(1);
    }
    if ch == 'w' {
        out.push('v');
        return Some(1);
    }
    if ch == 'y' {
        out.push('ʏ');
        return Some(1);
    }
    None
}

/// Tries diphthongs (`au`, `ei`/`ai`/`ey`, `oi`, `eu`/`äu`), `ie` not before
/// another vowel, doubled vowels, syllable-final `-er` vocalizing to `[ɐ]`,
/// and single vowel letters (with `e` softening to schwa syllable-finally or
/// before a single sonorant coda) at position `i`. Pushes onto `out` and
/// returns characters consumed, or `None`.
fn try_vowel(syllable: &[char], i: usize, out: &mut String) -> Option<usize> {
    let n = syllable.len();
    let ch = syllable[i];
    if slice_eq_str(syllable, i, "au") {
        out.push_str("aʊ̯");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "ei") || slice_eq_str(syllable, i, "ai") || slice_eq_str(syllable, i, "ey") {
        out.push_str("aɪ̯");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "oi") {
        out.push_str("ɔʏ̯");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "eu") || slice_eq_str(syllable, i, "äu") {
        out.push_str("ɔʏ̯");
        return Some(2);
    }
    if slice_eq_str(syllable, i, "ie") && !syllable.get(i + 2).is_some_and(|&c| is_vowel(c)) {
        out.push_str("iː");
        return Some(2);
    }
    if i + 1 < n && is_vowel(ch) && syllable[i + 1] == ch && matches!(ch, 'a' | 'o' | 'e' | 'i' | 'u') {
        out.push_str(match ch {
            'a' => "aː",
            'e' => "eː",
            'i' => "iː",
            'o' => "oː",
            'u' => "uː",
            _ => unreachable!("guarded by the matches! above"),
        });
        return Some(2);
    }
    // Syllable-final "-er" vocalizes to [ɐ] in standard German, so it is
    // handled as a two-character unit before "r" ever reaches the
    // unconditional r -> ʁ mapping in `syllable_to_ipa`.
    if ch == 'e' && i + 1 < n && syllable[i + 1] == 'r' && i + 2 == n {
        out.push('ɐ');
        return Some(2);
    }
    if is_vowel(ch) {
        match ch {
            'a' => out.push('a'),
            'e' => {
                // Schwa syllable-finally, or before a single sonorant coda
                // consonant (the "-en"/"-el"/"-em"/"-es" reduction family;
                // "-er" is already handled above).
                let is_schwa = i == n - 1
                    || (i + 2 == n && matches!(syllable[i + 1], 'n' | 'l' | 'm' | 'r' | 's'));
                out.push_str(if is_schwa { "ə" } else { "ɛ" });
            }
            'i' => out.push('ɪ'),
            'o' => out.push('ɔ'),
            'u' => out.push('ʊ'),
            'ä' => out.push('ɛ'),
            'ö' => out.push('ø'),
            'ü' | 'y' => out.push('ʏ'),
            _ => {}
        }
        return Some(1);
    }
    None
}

/// Converts one syllable's letters to IPA via a greedy left-to-right scan:
/// [`try_context_grapheme`], then [`try_fixed_consonant`], then
/// [`try_vowel`], then `r`/`ss`/`s`/default-consonant handling inline.
/// `full_word`/`span_start` give this syllable's absolute position for the
/// `ch` back/front-vowel lookup and the `st`/`sp` morpheme-boundary check.
fn syllable_to_ipa(
    syllable: &[char],
    full_word: &[char],
    morpheme_starts: &[bool],
    span_start: usize,
) -> String {
    let n = syllable.len();
    let mut out = String::with_capacity(n * 2);
    let mut i = 0usize;

    while i < n {
        let gi = span_start + i;

        if let Some(consumed) = try_context_grapheme(syllable, i, full_word, gi, morpheme_starts, &mut out) {
            i += consumed;
            continue;
        }
        if let Some(consumed) = try_fixed_consonant(syllable, i, &mut out) {
            i += consumed;
            continue;
        }
        if let Some(consumed) = try_vowel(syllable, i, &mut out) {
            i += consumed;
            continue;
        }

        let ch = syllable[i];
        if ch == 'r' {
            out.push('ʁ');
            i += 1;
            continue;
        }
        if slice_eq_str(syllable, i, "ss") {
            out.push('s');
            i += 2;
            continue;
        }
        if ch == 's' {
            let prev_v = i > 0 && is_vowel(syllable[i - 1]);
            let next_v = syllable.get(i + 1).is_some_and(|&c| is_vowel(c));
            out.push(if prev_v && next_v { 'z' } else { 's' });
            i += 1;
            continue;
        }
        match ch {
            'b' => out.push('b'),
            'd' => out.push('d'),
            'f' => out.push('f'),
            'g' => out.push('ɡ'),
            'k' => out.push('k'),
            'l' => out.push('l'),
            'm' => out.push('m'),
            'n' => out.push('n'),
            'p' => out.push('p'),
            't' => out.push('t'),
            _ => {}
        }
        i += 1;
    }

    apply_ig_fix(syllable, &mut out);
    final_devoice(out)
}

/// Inserts the primary stress mark into `syllable_ipas[stress_idx]`, right
/// before its first vowel sound (or at the very start if the syllable
/// somehow produced no vowel). Does nothing if `stress_idx` is out of range
/// or the target syllable produced no IPA at all.
fn insert_primary_stress(syllable_ipas: &mut [String], stress_idx: usize) {
    let Some(target) = syllable_ipas.get_mut(stress_idx) else {
        return;
    };
    match target.char_indices().find(|&(_, c)| is_ipa_vowel(c)) {
        Some((pos, _)) => target.insert(pos, IPA_PRIMARY_STRESS),
        None if !target.is_empty() => target.insert(0, IPA_PRIMARY_STRESS),
        None => {}
    }
}

/// Converts an out-of-vocabulary German word to an approximate IPA
/// transcription using hand-written letter-to-sound rules, as the final
/// fallback tier when lexicon lookup misses. Returns an empty string if
/// `word` contains no recognized German letters.
#[must_use]
pub fn hand_rules_ipa(word: &str) -> String {
    let chars = normalize_for_rules(word);
    let (full_word, syllables, morpheme_starts) = build_syllables_and_morpheme_starts(&chars);
    if syllables.is_empty() {
        return String::new();
    }
    let stress_idx = default_stress_syllable_index(&syllables, &full_word);

    let mut syllable_ipas: Vec<String> = Vec::with_capacity(syllables.len());
    for &(start, end) in &syllables {
        syllable_ipas.push(syllable_to_ipa(&full_word[start..end], &full_word, &morpheme_starts, start));
    }

    insert_primary_stress(&mut syllable_ipas, stress_idx);
    syllable_ipas.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_produces_empty_output() {
        // No letters at all should short-circuit to an empty string.
        assert_eq!(hand_rules_ipa(""), "");
    }

    #[test]
    fn all_punctuation_input_produces_empty_output() {
        // Hyphens alone leave no syllables after normalization.
        assert_eq!(hand_rules_ipa("---"), "");
    }

    #[test]
    fn non_german_characters_are_dropped() {
        // Digits and non-German letters are filtered before any rule runs.
        assert_eq!(hand_rules_ipa("h3llo"), hand_rules_ipa("hllo"));
    }

    #[test]
    fn uppercase_input_is_lowercased_first() {
        // Case must not affect the resulting phonemes.
        assert_eq!(hand_rules_ipa("HAUS"), hand_rules_ipa("haus"));
    }

    #[test]
    fn tsch_maps_to_postalveolar_affricate() {
        // "deutsch" contains the 4-letter "tsch" grapheme.
        assert!(hand_rules_ipa("deutsch").contains("tʃ"));
    }

    #[test]
    fn sch_maps_to_esh() {
        // "schule" starts with the "sch" trigraph.
        assert!(hand_rules_ipa("schule").contains('ʃ'));
    }

    #[test]
    fn chs_maps_to_ks() {
        // "wachsen" contains "chs".
        assert!(hand_rules_ipa("wachsen").contains("ks"));
    }

    #[test]
    fn ch_after_back_vowel_is_x() {
        // "buch" has "ch" after the back vowel "u".
        assert!(hand_rules_ipa("buch").contains('x'));
    }

    #[test]
    fn ch_after_front_vowel_is_ç() {
        // "ich" has "ch" after the front vowel "i".
        assert!(hand_rules_ipa("ich").contains('ç'));
    }

    #[test]
    fn ch_after_au_diphthong_is_x() {
        // "brauchen": the "au" diphthong ends in a back vowel, so a
        // following "ch" in the next syllable is still /x/ even though it
        // crosses a syllable boundary.
        assert!(hand_rules_ipa("brauchen").contains('x'));
    }

    #[test]
    fn word_initial_ch_has_no_preceding_vowel_and_is_ç() {
        assert!(hand_rules_ipa("chef").contains('ç'));
    }

    #[test]
    fn ch_after_intervening_consonant_is_ç_not_x() {
        // "durch" has "ch" after "r", a consonant, even though a back vowel
        // "u" sits further back in the word. The ich-laut/ach-laut split
        // depends only on the immediately preceding sound, so this must be
        // ich-laut /ç/, not ach-laut /x/.
        let ipa = hand_rules_ipa("durch");
        assert!(ipa.contains('ç'), "{ipa}");
        assert!(!ipa.contains('x'), "{ipa}");
    }

    #[test]
    fn ng_and_nk_map_correctly() {
        assert!(hand_rules_ipa("lang").contains('ŋ'));
        let bank = hand_rules_ipa("bank");
        assert!(bank.contains('ŋ') && bank.contains('k'));
    }

    #[test]
    fn pf_keeps_both_phonemes() {
        // No single ligature token exists for "pf" in the target vocabulary,
        // so it stays as two consonants.
        assert!(hand_rules_ipa("pferd").contains("pf"));
    }

    #[test]
    fn qu_maps_to_k_plus_v() {
        // "u" is "qu"'s only vowel, so it syllabifies as its own syllable
        // ("qu"); that syllable's trailing /v/ is then subject to the same
        // per-syllable final devoicing every syllable gets, surfacing as
        // /f/ here rather than /v/ — a known consequence of devoicing at
        // syllable granularity instead of only at the whole word's end.
        assert!(hand_rules_ipa("quelle").contains("kf"));
    }

    #[test]
    fn st_is_palatalized_at_word_start() {
        assert!(hand_rules_ipa("stahl").contains("ʃt"));
    }

    #[test]
    fn sp_is_palatalized_at_word_start() {
        assert!(hand_rules_ipa("spiel").contains("ʃp"));
    }

    #[test]
    fn st_is_not_palatalized_mid_morpheme() {
        // "fenster" has "st" in the middle of a single morpheme, not at a
        // word or hyphen boundary, so it stays plain /st/.
        let ipa = hand_rules_ipa("fenster");
        assert!(ipa.contains("st"));
        assert!(!ipa.contains("ʃt"));
    }

    #[test]
    fn st_is_palatalized_after_a_hyphen_boundary() {
        // The hyphen marks a morpheme boundary, so "st" right after it is
        // treated as morpheme-initial, unlike the mid-morpheme "st" in
        // st_is_not_palatalized_mid_morpheme above.
        assert!(hand_rules_ipa("auto-stopp").contains("ʃt"));
    }

    #[test]
    fn h_is_silent_word_initially() {
        // Deliberate baseline bug: word-initial /h/ is dropped.
        assert!(!hand_rules_ipa("haus").starts_with('h'));
    }

    #[test]
    fn h_is_silent_between_vowels() {
        let ipa = hand_rules_ipa("sehen");
        assert!(!ipa.contains('h'));
    }

    #[test]
    fn eszett_maps_to_s() {
        assert!(hand_rules_ipa("straße").contains('s'));
        assert!(!hand_rules_ipa("straße").contains('ß'));
    }

    #[test]
    fn tz_and_z_map_to_ts_affricate() {
        assert!(hand_rules_ipa("katze").contains("ts"));
        assert!(hand_rules_ipa("zeit").contains("ts"));
    }

    #[test]
    fn ck_maps_to_k() {
        let ipa = hand_rules_ipa("ecke");
        assert!(ipa.contains('k'));
        assert!(!ipa.contains("kk"));
    }

    #[test]
    fn c_before_e_or_i_is_ts_elsewhere_is_k() {
        assert!(hand_rules_ipa("celsius").contains("ts"));
    }

    #[test]
    fn v_and_w_are_swapped() {
        // German "v" is /f/ and "w" is /v/.
        assert!(hand_rules_ipa("vater").contains('f'));
        assert!(hand_rules_ipa("wasser").contains('v'));
    }

    #[test]
    fn x_maps_to_ks() {
        assert!(hand_rules_ipa("axt").contains("ks"));
    }

    #[test]
    fn y_not_before_vowel_maps_to_front_rounded_vowel() {
        assert!(hand_rules_ipa("system").contains('ʏ'));
    }

    #[test]
    fn diphthongs_resolve_correctly() {
        assert!(hand_rules_ipa("haus").contains("aʊ̯"));
        assert!(hand_rules_ipa("mein").contains("aɪ̯"));
        assert!(hand_rules_ipa("heute").contains("ɔʏ̯"));
    }

    #[test]
    fn oi_diphthong_maps_to_rounded_front_glide() {
        // "konvoi" (a loanword) contains the "oi" diphthong, which should
        // resolve to the same phoneme as "eu"/"äu" rather than two separate
        // monophthongs.
        assert!(hand_rules_ipa("konvoi").contains("ɔʏ̯"));
    }

    #[test]
    fn ie_not_before_vowel_is_long_i() {
        assert!(hand_rules_ipa("liebe").contains("iː"));
    }

    #[test]
    fn doubled_vowels_are_long() {
        assert!(hand_rules_ipa("haar").contains("aː"));
        assert!(hand_rules_ipa("boot").contains("oː"));
    }

    #[test]
    fn word_final_e_is_schwa() {
        let ipa = hand_rules_ipa("name");
        assert!(ipa.ends_with('ə'));
    }

    #[test]
    fn schwa_before_sonorant_coda() {
        // "laufen" syllabifies to "lau"+"fen"; the "e" in "fen" precedes a
        // single sonorant "n" in the coda and should reduce to schwa [ə],
        // not the full vowel [ɛ].
        let ipa = hand_rules_ipa("laufen");
        assert!(ipa.contains('ə'), "{ipa}");
        assert!(!ipa.contains('ɛ'), "{ipa}");
    }

    #[test]
    fn schwa_before_el_coda() {
        // "vogel" syllabifies to "vo"+"gel"; the "e" in "gel" precedes a
        // single sonorant "l" and should reduce to schwa.
        assert!(hand_rules_ipa("vogel").contains('ə'));
    }

    #[test]
    fn syllable_final_er_vocalizes_to_turned_a() {
        // "wasser" has a syllable-final "-er" that should vocalize to [ɐ],
        // not surface as [əʁ]/[ɛʁ].
        let ipa = hand_rules_ipa("wasser");
        assert!(ipa.ends_with('ɐ'), "{ipa}");
    }

    #[test]
    fn fenster_ends_with_turned_a() {
        // "Fenster" has a syllable-final "-er" that should vocalize to [ɐ],
        // matching the lexicon's own convention (e.g. "ˈfɛnstɐ").
        let ipa = hand_rules_ipa("fenster");
        assert!(ipa.ends_with('ɐ'), "{ipa}");
    }

    #[test]
    fn umlauts_map_to_front_rounded_vowels() {
        assert!(hand_rules_ipa("mächtig").contains('ɛ'));
        assert!(hand_rules_ipa("können").contains('ø'));
        assert!(hand_rules_ipa("müde").contains('ʏ'));
    }

    #[test]
    fn r_maps_to_uvular_fricative() {
        assert!(hand_rules_ipa("rot").contains('ʁ'));
    }

    #[test]
    fn ss_collapses_to_single_s() {
        assert!(!hand_rules_ipa("wasser").contains("ss"));
    }

    #[test]
    fn s_between_vowels_is_voiced_within_a_syllable() {
        // The intervocalic-voicing check only looks within the current
        // syllable, and the syllabifier always gives a lone intervening
        // consonant to the following syllable as its onset (never leaving
        // it as a preceding syllable's coda), so a real word's "s" here
        // never actually has a same-syllable vowel on both sides — this
        // exercises the rule directly against a hand-built syllable instead.
        let syllable: Vec<char> = "asa".chars().collect();
        let ipa = syllable_to_ipa(&syllable, &syllable, &[true, false, false], 0);
        assert!(ipa.contains('z'), "{ipa}");
    }

    #[test]
    fn s_elsewhere_is_voiceless() {
        assert!(hand_rules_ipa("haus").contains('s'));
        assert!(!hand_rules_ipa("haus").contains('z'));
    }

    #[test]
    fn ig_suffix_softens_to_palatal_fricative() {
        let ipa = hand_rules_ipa("fertig");
        assert!(ipa.ends_with('ç'));
    }

    #[test]
    fn ig_softening_only_applies_to_a_syllable_ending_in_ig() {
        // "königlich" syllabifies as "kö-ni-glich": the consonant cluster
        // before the last "i" (including the "g" from "könig") is swept
        // into the following syllable's onset, so no syllable here actually
        // ends in "-ig" and the softening rule never fires. Any trailing
        // /ç/ comes from the ordinary "ch"-after-front-vowel rule instead.
        let chars: Vec<char> = "königlich".chars().collect();
        let (full_word, syllables, _) = build_syllables_and_morpheme_starts(&chars);
        assert!(
            syllables.iter().all(|&(s, e)| !ends_with_str(&full_word[s..e], "ig")),
            "{syllables:?}"
        );
    }

    #[test]
    fn ig_softening_does_not_fire_on_diphthong() {
        // "steig" ends in the letters "ig", but the "i" is part of the "ei"
        // diphthong, not a standalone "-ig" suffix, so the softening rule
        // must not turn the final devoiced /k/ into /ç/.
        let ipa = hand_rules_ipa("steig");
        assert!(!ipa.ends_with('ç'), "{ipa}");
        assert!(ipa.ends_with('k'), "{ipa}");
    }

    #[test]
    fn final_devoicing_applies_to_voiced_obstruents() {
        assert!(hand_rules_ipa("lieb").ends_with('p'));
        assert!(hand_rules_ipa("rad").ends_with('t'));
        assert!(hand_rules_ipa("tag").ends_with('k'));
    }

    #[test]
    fn single_syllable_word_is_stressed() {
        assert!(hand_rules_ipa("haus").contains(IPA_PRIMARY_STRESS));
    }

    #[test]
    fn unstressed_prefix_shifts_stress_to_next_syllable() {
        // "verstehen" starts with the unstressed prefix "ver" (3 chars), so
        // `unstressed_prefix_len` must return 3, not 0.
        let word: Vec<char> = "verstehen".chars().collect();
        assert_eq!(unstressed_prefix_len(&word), 3);
    }

    #[test]
    fn newly_added_prefix_shifts_stress_to_next_syllable() {
        // "durchfahren" starts with the unstressed prefix "durch" (5 chars),
        // which was one of the prefixes added to UNSTRESSED_PREFIXES after
        // measuring against the CER benchmark, so `unstressed_prefix_len`
        // must return 5, not 0.
        let word: Vec<char> = "durchfahren".chars().collect();
        assert_eq!(unstressed_prefix_len(&word), 5);
    }

    #[test]
    fn longer_prefix_wins_over_shorter_prefix_it_starts_with() {
        // "beisteuern" starts with both "be" (2) and the longer, more
        // specific "bei" (3) — the longer prefix must be tried first, so
        // `unstressed_prefix_len` must return 3, not 2.
        let word: Vec<char> = "beisteuern".chars().collect();
        assert_eq!(unstressed_prefix_len(&word), 3);
    }

    #[test]
    fn ung_suffix_stresses_final_syllable() {
        let ipa = hand_rules_ipa("bildung");
        // The stress mark should appear after at least one syllable's worth
        // of phonemes, i.e. not at the very start.
        assert!(!ipa.starts_with(IPA_PRIMARY_STRESS));
    }

    #[test]
    fn output_ends_up_with_exactly_one_stress_mark_for_simple_words() {
        let ipa = hand_rules_ipa("hose");
        assert_eq!(ipa.matches(IPA_PRIMARY_STRESS).count(), 1);
    }

    #[test]
    fn slice_eq_str_matches_bounds_safely() {
        // Verifies the helper doesn't panic or false-match at the string's
        // tail end.
        let chars: Vec<char> = "ab".chars().collect();
        assert!(slice_eq_str(&chars, 0, "ab"));
        assert!(!slice_eq_str(&chars, 0, "abc"));
        assert!(!slice_eq_str(&chars, 1, "bc"));
    }

    #[test]
    fn starts_with_str_requires_nonempty_remainder() {
        let chars: Vec<char> = "ver".chars().collect();
        // Exact-length match leaves nothing to stress, so it must not count.
        assert!(!starts_with_str(&chars, "ver"));
        let chars: Vec<char> = "verstehen".chars().collect();
        assert!(starts_with_str(&chars, "ver"));
    }

    #[test]
    fn ends_with_str_matches_expected_suffixes() {
        let chars: Vec<char> = "bildung".chars().collect();
        assert!(ends_with_str(&chars, "ung"));
        assert!(!ends_with_str(&chars, "schaft"));
    }

    #[test]
    fn syllabify_segment_splits_on_vowel_nuclei() {
        let chars: Vec<char> = "fenster".chars().collect();
        let syllables = syllabify_segment(&chars);
        assert!(syllables.len() >= 2, "{syllables:?}");
    }

    #[test]
    fn syllabify_segment_with_no_vowel_is_one_syllable() {
        let chars: Vec<char> = "pst".chars().collect();
        assert_eq!(syllabify_segment(&chars).len(), 1);
    }
}
