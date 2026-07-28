// SPDX-License-Identifier: MIT

//! Hand-written English letter-to-sound rules.
//!
//! This is the final G2P fallback tier: applied when a word is not found in
//! the lexicon. Greedily matches multi-letter graphemes (digraphs/trigraphs)
//! and function words first, then falls back to context-sensitive
//! single-vowel and single-consonant rules (magic-e lengthening, r-colored
//! vowels, open/closed syllable detection, soft c/g), and finally inserts a
//! primary stress mark before the first vowel if none is already present.
//!
//! Known limitation: grapheme matching is a context-free left-to-right scan
//! with no morpheme/compound-boundary awareness, so digraphs can span
//! compound-word boundaries incorrectly (e.g. the "gh" in "doghouse" is read
//! as the single silent digraph rather than separate /g/ and /h/ sounds).

/// Unicode primary stress mark (U+02C8).
const IPA_PRIMARY_STRESS: char = 'ˈ';
/// Unicode secondary stress mark (U+02CC).
const IPA_SECONDARY_STRESS: char = 'ˌ';

/// Small, closed set of function words with irreducible, unstressed
/// pronunciations that the letter-by-letter rules below would get wrong.
const FUNCTION_WORDS: &[(&str, &str)] = &[
    ("the", "ðə"),
    ("a", "ə"),
    ("an", "æn"),
    ("to", "tə"),
    ("of", "əv"),
    ("and", "ænd"),
    ("or", "ɔɹ"),
    ("are", "ɑɹ"),
    ("was", "wəz"),
    ("were", "wɝ"),
    ("from", "fɹʌm"),
    ("have", "hæv"),
    ("has", "hæz"),
    ("been", "bɪn"),
    ("do", "du"),
    ("does", "dʌz"),
    ("your", "jɔɹ"),
    ("you", "ju"),
    ("they", "ðeɪ"),
    ("their", "ðɛɹ"),
    ("there", "ðɛɹ"),
];

/// Multi-letter grapheme -> IPA rules, longest patterns checked first within
/// each length cluster. `"gh"` and `"th"` are matched here but resolved by
/// dedicated context-sensitive logic in [`oov_grapheme_to_ipa`] rather than
/// this table's placeholder IPA.
const LITERALS: &[(&str, &str)] = &[
    ("tch", "tʃ"),
    ("dge", "dʒ"),
    ("tion", "ʃən"),
    ("sion", "ʒən"),
    ("sure", "ʒɚ"),
    ("ture", "tʃɚ"),
    ("ough", "oʊ"),
    ("augh", "ɔː"),
    ("eigh", "eɪ"),
    ("igh", "aɪ"),
    ("oar", "ɔɹ"),
    ("our", "aʊɹ"),
    ("oor", "ɔɹ"),
    ("ear", "ɪɹ"),
    ("eer", "ɪɹ"),
    ("ier", "ɪɹ"),
    ("air", "ɛɹ"),
    ("are", "ɛɹ"),
    ("ire", "aɪɹ"),
    ("ure", "jʊɹ"),
    ("ai", "eɪ"),
    ("ay", "eɪ"),
    ("au", "ɔː"),
    ("aw", "ɔː"),
    ("ea", "iː"),
    ("ee", "iː"),
    ("ei", "eɪ"),
    ("ey", "eɪ"),
    ("eu", "juː"),
    ("ew", "juː"),
    ("ie", "iː"),
    ("oa", "oʊ"),
    ("oe", "oʊ"),
    ("oi", "ɔɪ"),
    ("oy", "ɔɪ"),
    ("oo", "uː"),
    ("ou", "aʊ"),
    ("ow", "oʊ"),
    ("ph", "f"),
    ("gh", ""),
    ("ng", "ŋ"),
    ("ch", "tʃ"),
    ("sh", "ʃ"),
    ("th", "θ"),
    ("wh", "w"),
    ("qu", "kw"),
    ("ck", "k"),
    ("sch", "sk"),
    ("ss", "s"),
    ("ll", "l"),
    ("mm", "m"),
    ("nn", "n"),
    ("ff", "f"),
    ("pp", "p"),
    ("tt", "t"),
    ("zz", "z"),
    ("rr", "ɹ"),
    ("dd", "d"),
    ("bb", "b"),
    ("gg", "ɡ"),
];

/// Priority order for locating the syllable that gets the primary stress
/// mark when none is already present: the first prefix in this list that
/// occurs anywhere in the IPA string wins, regardless of byte position.
const VOWEL_PREFIXES: &[&str] = &[
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "juː", "iː", "uː", "ɑː", "ɔː", "ɜː", "ɛɹ", "ɑɹ", "ɔɹ", "ɪɹ",
    "ʊɹ", "aɪɹ", "ɪə", "eə", "ʊə", "iə", "ə", "ɪ", "ɛ", "æ", "ʌ", "ʊ", "ɑ", "ɔ", "i", "u", "e",
    "o", "ɚ", "ɝ", "ɒ",
];

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

fn is_consonant(c: u8) -> bool {
    c.is_ascii_lowercase() && !is_vowel(c)
}

/// Returns the index of the next vowel at or after `start`, or `None` if
/// there isn't one.
fn next_vowel_index(w: &[u8], start: usize) -> Option<usize> {
    w.get(start..)?
        .iter()
        .position(|&c| is_vowel(c))
        .map(|i| i + start)
}

/// Returns `true` if `c` is one of the vowel-sound IPA symbols this module
/// emits, or the length mark `ː` (which always follows one).
fn is_vowel_ipa_char(c: char) -> bool {
    matches!(
        c,
        'æ' | 'ɛ' | 'ɪ' | 'ɔ' | 'ʊ' | 'ɑ' | 'ɒ' | 'ə' | 'ɚ' | 'ɝ' | 'ɨ' | 'ʉ' | 'ː'
    )
}

/// Returns `true` if the last codepoint of `ipa` is a vowel sound. Used to
/// decide whether a `"gh"` following a vowel should be silent.
fn last_ipa_unit_is_vowel(ipa: &str) -> bool {
    ipa.chars().last().is_some_and(is_vowel_ipa_char)
}

/// Returns `true` if `ipa` contains any vowel sound. Used to tell a word's
/// only vowel (which must not be silenced) apart from a true silent
/// magic-e.
fn contains_vowel_sound(ipa: &str) -> bool {
    ipa.chars().any(is_vowel_ipa_char)
}

/// Detects the "magic-e" pattern: a silent trailing `e` that lengthens the
/// vowel at `vowel_i` (e.g. "make" vs "mac"). True only when exactly one
/// consonant sits between the vowel and the final `e`.
fn magic_e_lengthens(w: &[u8], vowel_i: usize) -> bool {
    if vowel_i >= w.len() {
        return false;
    }
    if *w.last().unwrap() != b'e' || w.len() < vowel_i + 3 {
        return false;
    }
    let j = vowel_i + 1;
    if j >= w.len() - 1 {
        return false;
    }
    let second_last = w[w.len() - 2];
    if !second_last.is_ascii_lowercase() || is_vowel(second_last) {
        return false;
    }
    let mid = &w[j..w.len() - 1];
    if mid.is_empty() || mid.iter().any(|&c| is_vowel(c)) {
        return false;
    }
    mid.len() == 1
}

/// If the vowel at `i` is followed by `r`, returns its r-colored IPA and the
/// number of graphemes consumed (2: the vowel plus the `r`).
fn r_controlled(w: &[u8], i: usize) -> Option<(&'static str, usize)> {
    if i + 1 >= w.len() || w[i + 1] != b'r' {
        return None;
    }
    let ipa = match w[i] {
        b'a' => "ɑɹ",
        b'e' => "ɛɹ",
        b'i' => "ɪɹ",
        b'o' => "ɔɹ",
        b'u' => "ʊɹ",
        b'y' => "aɪɹ",
        _ => return None,
    };
    Some((ipa, 2))
}

/// Resolves a single vowel letter at `i` to its IPA and consumed length (1),
/// checking r-colored vowels and magic-e lengthening first, then falling
/// back to an open-vs-closed-syllable heuristic.
fn oov_vowel(w: &[u8], i: usize) -> (&'static str, usize) {
    if let Some(rc) = r_controlled(w, i) {
        return rc;
    }

    let magic = magic_e_lengthens(w, i);
    let next_vowel = next_vowel_index(w, i + 1);
    let closed = if let Some(next_vowel) = next_vowel {
        let between = &w[i + 1..next_vowel];
        !between.is_empty() && !between.iter().any(|&c| is_vowel(c))
    } else {
        i + 1 < w.len() && !is_vowel(w[i + 1])
    };

    match w[i] {
        b'a' if magic => ("eɪ", 1),
        b'a' if closed => ("æ", 1),
        b'a' => ("ɑː", 1),
        b'e' if magic => ("iː", 1),
        b'e' if closed || i == w.len() - 1 => ("ɛ", 1),
        b'e' => ("iː", 1),
        b'i' | b'y' if magic => ("aɪ", 1),
        b'i' | b'y' if closed => ("ɪ", 1),
        b'o' if magic => ("oʊ", 1),
        b'o' if closed => ("ɒ", 1),
        b'o' => ("oʊ", 1),
        b'u' if magic => ("juː", 1),
        b'u' if closed => ("ʌ", 1),
        b'u' => ("uː", 1),
        b'i' | b'y' => ("aɪ", 1),
        _ => ("ə", 1),
    }
}

/// True when the `b` in a "-mb" cluster starting at `i` is silent: the
/// cluster is word-final, or followed only by a safe inflectional suffix
/// (`-s`, `-ed`, `-ing`, `-ness`) that never turns "mb" into a different
/// cluster. Deliberately excludes `-er`/`-ers`/`-le`/`-ly`, which would
/// misfire on "timber", "amber", "humble", "nimbly" (all sound both
/// consonants). Mid-word "mb" otherwise (e.g. "combat") sounds both.
///
/// Note: silencing the `b` does not correct vowel quality — "climb" still
/// resolves through the closed-syllable heuristic to a short vowel instead
/// of the correct /aɪ/, since the magic-e check no longer sees a trailing
/// `e`. This is a pre-existing heuristic limitation, not fixed here.
fn is_silent_final_mb(w: &[u8], i: usize) -> bool {
    if i + 1 >= w.len() || w[i] != b'm' || w[i + 1] != b'b' {
        return false;
    }
    matches!(&w[i + 2..], b"" | b"s" | b"ed" | b"ing" | b"ness")
}

/// Resolves a single consonant letter at `i`, pushing its IPA (or, for `c`
/// and `g`, a context-dependent soft/hard variant) directly onto `out`.
fn push_single_consonant(out: &mut String, w: &[u8], i: usize) {
    let next = w.get(i + 1).copied();
    let soft_context = matches!(next, Some(b'e' | b'i' | b'y'));

    let ipa = match w[i] {
        b'c' if soft_context => "s",
        b'g' if soft_context => "dʒ",
        b'g' => "ɡ",
        b'c' | b'q' | b'k' => "k",
        b'j' => "dʒ",
        b'x' => "ks",
        b'r' => "ɹ",
        b'h' => "h",
        b'b' => "b",
        b'd' => "d",
        b'f' => "f",
        b'l' => "l",
        b'm' => "m",
        b'n' => "n",
        b'p' => "p",
        b's' => "s",
        b't' => "t",
        b'v' => "v",
        b'w' => "w",
        b'z' => "z",
        _ => unreachable!(
            "push_single_consonant is only called for consonants, all of which are matched above"
        ),
    };
    out.push_str(ipa);
}

/// Words whose `th` is voiced (`ð`) rather than the default voiceless `θ`.
fn th_voiced_word(w: &str) -> bool {
    matches!(
        w,
        "the" | "this" | "that" | "they" | "them" | "then" | "than" | "there" | "these"
            | "those" | "though" | "thus" | "thence" | "thy" | "thee" | "thou" | "thine"
            | "with" | "within" | "without"
    )
}

/// Greedily converts an already-normalized word into IPA: function-word
/// overrides first, then a left-to-right scan matching multi-letter
/// graphemes, then single vowel/consonant rules for anything left over.
///
/// Expects `word` to already be normalized via `normalize_word_for_lookup`
/// (the sole caller, [`hand_oov_rules_ipa`], is only ever invoked with
/// pre-normalized input). Only ASCII lowercase letters survive into the
/// grapheme scan — digits, apostrophes, hyphens, and non-ASCII characters
/// are silently dropped.
fn oov_grapheme_to_ipa(word: &str) -> String {
    let letters: String = word.chars().filter(char::is_ascii_lowercase).collect();
    if letters.is_empty() {
        return String::new();
    }

    if let Some(&(_, ipa)) = FUNCTION_WORDS.iter().find(|&&(k, _)| k == letters) {
        return ipa.to_string();
    }

    let voiced_th = th_voiced_word(&letters);
    let w = letters.as_bytes();
    let n = w.len();
    let mut out = String::with_capacity(n * 2);
    let mut i = 0usize;

    while i < n {
        // Silent initial consonant clusters: the first letter of "kn-",
        // "wr-", "gn-", "ps-" is silent only at word start (e.g. "know",
        // "write", "gnat", "psalm") — mid-word occurrences like "unknot" or
        // "handwritten" sound both consonants.
        if i == 0
            && n >= 2
            && matches!(
                (w[0], w[1]),
                (b'k' | b'g', b'n') | (b'w', b'r') | (b'p', b's')
            )
        {
            i += 1;
            continue;
        }
        // Silent final "b" in "-mb" (e.g. "lamb", "climb", "climbing"); see
        // is_silent_final_mb for the excluded mid-word/suffix cases.
        if is_silent_final_mb(w, i) {
            out.push('m');
            i += 2;
            continue;
        }

        // Silent final "e" (e.g. the trailing 'e' in "make"), but only once
        // a vowel sound has already been emitted — otherwise this `e` is
        // the word's only vowel (e.g. "he", "be", "she") and must not be
        // dropped.
        if w[i] == b'e' && i == n - 1 && contains_vowel_sound(&out) {
            i += 1;
            continue;
        }

        let mut matched = false;
        for &(grapheme, ipa) in LITERALS {
            let len = grapheme.len();
            if i + len > n || &w[i..i + len] != grapheme.as_bytes() {
                continue;
            }

            if grapheme == "gh" {
                if !last_ipa_unit_is_vowel(&out) {
                    out.push('ɡ');
                }
            } else if grapheme == "th" {
                out.push(if voiced_th { 'ð' } else { 'θ' });
            } else {
                out.push_str(ipa);
            }
            i += len;
            matched = true;
            break;
        }
        if matched {
            continue;
        }

        if is_vowel(w[i]) {
            let (ipa, consumed) = oov_vowel(w, i);
            out.push_str(ipa);
            i += consumed;
        } else if is_consonant(w[i]) {
            push_single_consonant(&mut out, w, i);
            i += 1;
        } else {
            i += 1;
        }
    }

    out
}

/// Inserts a primary stress mark before the highest-priority vowel found in
/// `ipa`, unless it already starts with a stress mark.
///
/// This is a rough heuristic: it picks by vowel-quality priority rather
/// than syllable position, so it will disagree with real English stress
/// patterns for many multi-syllable words.
fn add_primary_stress_if_missing(ipa: String) -> String {
    if ipa.is_empty() || ipa.starts_with(IPA_PRIMARY_STRESS) || ipa.starts_with(IPA_SECONDARY_STRESS)
    {
        return ipa;
    }

    for prefix in VOWEL_PREFIXES {
        if let Some(pos) = ipa.find(prefix) {
            let mut out = String::with_capacity(ipa.len() + IPA_PRIMARY_STRESS.len_utf8());
            out.push_str(&ipa[..pos]);
            out.push(IPA_PRIMARY_STRESS);
            out.push_str(&ipa[pos..]);
            return out;
        }
    }

    let mut out = String::with_capacity(ipa.len() + IPA_PRIMARY_STRESS.len_utf8());
    out.push(IPA_PRIMARY_STRESS);
    out.push_str(&ipa);
    out
}

/// Converts an out-of-vocabulary English word to an approximate IPA
/// transcription using hand-written letter-to-sound rules, as the final
/// fallback tier when lexicon lookup misses.
///
/// Expects `word` to already be normalized via `normalize_word_for_lookup`.
#[must_use]
pub fn hand_oov_rules_ipa(word: &str) -> String {
    add_primary_stress_if_missing(oov_grapheme_to_ipa(word))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_words_use_reduced_forms() {
        // Function-word IPA still gets a primary stress mark inserted, same
        // as every other path through `hand_oov_rules_ipa`.
        assert_eq!(hand_oov_rules_ipa("the"), "ðˈə");
        assert_eq!(hand_oov_rules_ipa("a"), "ˈə");
        assert_eq!(hand_oov_rules_ipa("they"), "ðˈeɪ");
    }

    #[test]
    fn digraphs_resolve_correctly() {
        assert!(hand_oov_rules_ipa("ship").contains('ʃ'));
        assert!(hand_oov_rules_ipa("chip").contains("tʃ"));
        assert!(hand_oov_rules_ipa("phone").contains('f'));
    }

    #[test]
    fn th_is_voiced_only_in_specific_words() {
        assert!(hand_oov_rules_ipa("the").contains('ð'));
        assert!(hand_oov_rules_ipa("think").contains('θ'));
        assert!(!hand_oov_rules_ipa("think").contains('ð'));
    }

    #[test]
    fn th_voiced_in_expanded_word_list() {
        for word in [
            "them", "though", "thus", "thence", "thy", "thee", "thou", "thine", "with",
            "within", "without",
        ] {
            assert!(
                hand_oov_rules_ipa(word).contains('ð'),
                "{word} should have voiced th"
            );
        }
    }

    #[test]
    fn gh_is_silent_after_a_vowel() {
        // "night" -> /n/ + "igh" (-> aɪ) + "t"; the "gh" digraph should not
        // separately emit a hard /ɡ/.
        let ipa = hand_oov_rules_ipa("night");
        assert!(!ipa.contains('ɡ'));
    }

    #[test]
    fn gh_is_hard_at_word_start() {
        assert!(hand_oov_rules_ipa("ghost").contains('ɡ'));
    }

    #[test]
    fn silent_k_in_initial_kn() {
        // "knife" has no other 'k'/'ck' spelling, unlike e.g. "knack".
        let ipa = hand_oov_rules_ipa("knife");
        assert!(!ipa.contains('k'), "kn- should silence the k: {ipa}");
        assert!(ipa.contains('n'));
    }

    #[test]
    fn silent_w_in_initial_wr() {
        let ipa = hand_oov_rules_ipa("wring");
        assert!(!ipa.contains('w'), "wr- should silence the w: {ipa}");
        assert!(ipa.contains('ɹ'));
    }

    #[test]
    fn silent_g_in_initial_gn() {
        let ipa = hand_oov_rules_ipa("gnat");
        assert!(!ipa.contains('ɡ'), "gn- should silence the g: {ipa}");
        assert!(ipa.contains('n'));
    }

    #[test]
    fn silent_p_in_initial_ps() {
        let ipa = hand_oov_rules_ipa("psalm");
        assert!(!ipa.contains('p'), "ps- should silence the p: {ipa}");
        assert!(ipa.contains('s'));
    }

    #[test]
    fn silent_b_in_final_mb() {
        // Pinned to the exact output (not just substring presence) so a
        // future change to vowel resolution around the silenced "b" is
        // caught, not just the consonant-presence check.
        assert_eq!(hand_oov_rules_ipa("lamb"), "lˈæm");
    }

    #[test]
    fn silent_b_in_final_mb_before_inflectional_suffix() {
        // "-mb" followed by a safe inflectional suffix should still
        // silence the b: the suffix doesn't turn "mb" into a different
        // cluster the way "-er"/"-le"/"-ly" do (see plumber test below).
        for word in ["climbing", "numbness", "combed", "lambs"] {
            let ipa = hand_oov_rules_ipa(word);
            assert!(!ipa.contains('b'), "{word} should silence the b: {ipa}");
            assert!(ipa.contains('m'), "{word} should keep the m: {ipa}");
        }
    }

    #[test]
    fn mb_before_er_suffix_preserves_both() {
        // "-mber" is excluded from the silent-mb suffix set: "timber",
        // "amber", "chamber" all sound both consonants, so a generic
        // "-mb + suffix" rule can't include "-er" without false positives.
        for word in ["plumber", "timber", "chamber"] {
            let ipa = hand_oov_rules_ipa(word);
            assert!(ipa.contains('b'), "{word} should sound the b: {ipa}");
        }
    }

    #[test]
    fn mb_mid_word_preserves_both() {
        // "combat" has "mb" mid-word; both consonants should sound.
        let ipa = hand_oov_rules_ipa("combat");
        assert!(ipa.contains('m'));
        assert!(ipa.contains('b'));
    }

    #[test]
    fn kn_mid_word_preserves_both() {
        // "unknown" has "kn" mid-word (not at position 0); both should sound.
        let ipa = hand_oov_rules_ipa("unknown");
        assert!(ipa.contains('k'));
        assert!(ipa.contains('n'));
    }

    #[test]
    fn magic_e_lengthens_the_preceding_vowel() {
        // "mak" (no magic e) would give /æ/; "make" should give /eɪ/.
        assert!(hand_oov_rules_ipa("make").contains("eɪ"));
    }

    #[test]
    fn r_controlled_vowels() {
        assert!(hand_oov_rules_ipa("car").contains("ɑɹ"));
        assert!(hand_oov_rules_ipa("her").contains("ɛɹ"));
    }

    #[test]
    fn soft_and_hard_c_and_g() {
        assert!(hand_oov_rules_ipa("cell").contains('s'));
        assert!(!hand_oov_rules_ipa("cat").contains('s'));
        assert!(hand_oov_rules_ipa("gem").contains("dʒ"));
        assert!(hand_oov_rules_ipa("gap").contains('ɡ'));
    }

    #[test]
    fn doubled_consonants_collapse() {
        assert!(!hand_oov_rules_ipa("miss").contains("ss"));
        assert!(hand_oov_rules_ipa("miss").contains('s'));
    }

    #[test]
    fn output_has_primary_stress_marker() {
        assert!(hand_oov_rules_ipa("xyzzy").contains('ˈ'));
    }

    #[test]
    fn empty_input_produces_empty_output() {
        assert_eq!(hand_oov_rules_ipa(""), "");
    }

    #[test]
    fn all_punctuation_input_produces_empty_output() {
        assert_eq!(hand_oov_rules_ipa("---"), "");
    }

    #[test]
    fn short_e_only_words_keep_their_vowel() {
        for word in ["he", "be", "we", "she"] {
            let ipa = hand_oov_rules_ipa(word);
            assert!(contains_vowel_sound(&ipa), "{word} produced no vowel: {ipa}");
        }
    }

    #[test]
    fn y_as_magic_e_vowel_lengthens_to_ai_diphthong() {
        assert!(hand_oov_rules_ipa("type").contains("aɪ"));
        assert!(hand_oov_rules_ipa("style").contains("aɪ"));
    }

    #[test]
    fn compound_word_digraph_boundary_is_a_known_limitation() {
        // "doghouse" = dog + house, but the greedy context-free scanner has
        // no morpheme-boundary awareness: it folds the cross-boundary "ou"
        // into a single "aʊ" grapheme, and the "gh" is read as
        // silent-after-vowel rather than separate /g/ + /h/ sounds. This
        // pins the current (known-imperfect) behavior, not correctness.
        assert_eq!(hand_oov_rules_ipa("doghouse"), "dɒˈaʊs");
    }

    #[test]
    fn non_ascii_and_digit_characters_are_dropped_not_panicking() {
        assert_eq!(hand_oov_rules_ipa("café"), "kˈæf");
        assert_eq!(hand_oov_rules_ipa("3d"), "ˈd");
    }
}
