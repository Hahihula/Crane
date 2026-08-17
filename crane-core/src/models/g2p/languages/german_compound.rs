// SPDX-License-Identifier: MIT

//! German compound-word decomposition.
//!
//! Tries to split a word that missed whole-word lexicon lookup into two or
//! more components that each individually hit the lexicon, preferring the
//! longest possible component at each step and backtracking to a shorter
//! one if the remainder can't be split further. This is the tier between
//! lexicon lookup and hand rules. Component strings themselves are
//! borrowed slices of the original word, but the search is not
//! allocation-free: each recursive step builds a byte-position table for
//! its remaining substring (see [`char_byte_positions`]), and a lookup
//! miss during backtracking can allocate a case-folded key (see
//! [`lookup_cascade`](super::german::lookup_cascade)) — allocation scales
//! with the number of candidate splits tried, not just the final
//! assembled IPA string.

use crate::models::g2p::lexicon::Lexicon;

use super::german::lookup_cascade;

/// Minimum codepoint length a single compound component must have to be
/// accepted as a split point.
const MIN_COMPONENT_LEN: usize = 4;

/// Maximum number of components a compound may be split into.
const MAX_COMPONENTS: usize = 4;

/// Word length (in codepoints) above which a whole-word lookup miss is
/// worth attempting to split. Shorter misses are unlikely to be genuine
/// compounds and go straight to hand rules.
const MIN_COMPOUND_LEN: usize = 12;

/// Word length (in codepoints) above which decomposition is not attempted
/// at all, regardless of whether a valid split exists. Bounds the cost of
/// `find_split`'s backtracking search to a constant independent of input
/// length — without this cap, a single long lexicon-miss token (which may
/// originate from untrusted TTS request text) could drive an amount of
/// candidate-lookup work proportional to the square of its length. 40
/// codepoints comfortably covers real German compounds (e.g.
/// "Donaudampfschifffahrtskapitän" is 29).
const MAX_COMPOUND_LEN: usize = 40;

/// Unicode primary stress mark (U+02C8).
const IPA_PRIMARY_STRESS: char = 'ˈ';
/// Unicode secondary stress mark (U+02CC).
const IPA_SECONDARY_STRESS: char = 'ˌ';

/// Returns each character's starting byte offset in `s`, plus `s.len()` as
/// a trailing sentinel, so `positions[k]` is always a valid char-boundary
/// byte offset for `0 <= k <= s.chars().count()`.
fn char_byte_positions(s: &str) -> Vec<usize> {
    s.char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(s.len()))
        .collect()
}

/// Recursively finds a run of lexicon-hitting components that together
/// cover all of `remaining`, trying the longest possible next component
/// first and backtracking to shorter ones if a longer choice's remainder
/// can't itself be split within the component budget. Checking whether
/// `remaining` itself (subject to the same [`MIN_COMPONENT_LEN`] floor as
/// every other candidate) hits lookup *before* trying to split it handles
/// both the top-level call (already known to miss) and every recursive
/// remainder, where it is the actual base case: "the rest of the word is a
/// genuine final component." This whole-remainder-first check is a
/// deliberate "prefer the longest match" bias: if a long compound is
/// itself already a lexicon entry, it is accepted as a single component
/// even when a finer split into more, shorter components (with different
/// secondary-stress placement) also exists.
fn find_split<'lex, 'w>(
    lexicon: &'lex Lexicon,
    remaining: &'w str,
    components_left: usize,
) -> Option<Vec<(&'w str, &'lex str)>> {
    if components_left == 0 {
        return None;
    }
    let byte_positions = char_byte_positions(remaining);
    let total_chars = byte_positions.len() - 1;
    if total_chars >= MIN_COMPONENT_LEN
        && let Some(ipa) = lookup_cascade(lexicon, remaining)
    {
        return Some(vec![(remaining, ipa)]);
    }
    if components_left == 1 || total_chars < 2 * MIN_COMPONENT_LEN {
        return None;
    }

    for prefix_len in (MIN_COMPONENT_LEN..=(total_chars - MIN_COMPONENT_LEN)).rev() {
        let split_at = byte_positions[prefix_len];
        let prefix = &remaining[..split_at];
        let Some(prefix_ipa) = lookup_cascade(lexicon, prefix) else {
            continue;
        };
        let tail = &remaining[split_at..];
        if let Some(mut rest) = find_split(lexicon, tail, components_left - 1) {
            let mut components = Vec::with_capacity(1 + rest.len());
            components.push((prefix, prefix_ipa));
            components.append(&mut rest);
            return Some(components);
        }
    }
    None
}

/// Concatenates each component's IPA directly (no glue phoneme between
/// components), downgrading every component after the first from primary
/// to secondary stress — German compound stress keeps primary stress on
/// the first component only. A component with no primary stress mark at
/// all is copied through unchanged.
fn assemble_ipa(components: &[(&str, &str)]) -> String {
    let capacity = components.iter().map(|&(_, ipa)| ipa.len()).sum();
    let mut ipa = String::with_capacity(capacity);
    for (idx, &(_, component_ipa)) in components.iter().enumerate() {
        if idx == 0 {
            ipa.push_str(component_ipa);
        } else {
            for c in component_ipa.chars() {
                ipa.push(if c == IPA_PRIMARY_STRESS {
                    IPA_SECONDARY_STRESS
                } else {
                    c
                });
            }
        }
    }
    ipa
}

/// Tries to decompose `word` into two or more lexicon-hitting components,
/// assembling their IPA with compound stress rules (see [`assemble_ipa`]).
/// Returns `None` if `word` is too short or too long to bother trying (see
/// [`MIN_COMPOUND_LEN`]/[`MAX_COMPOUND_LEN`]), or if no full decomposition
/// exists within the component-count and per-component length bounds — the
/// caller should fall through to hand rules in that case, unchanged from
/// before this tier existed.
pub(super) fn decompose(lexicon: &Lexicon, word: &str) -> Option<String> {
    let word_len = word.chars().count();
    if word_len <= MIN_COMPOUND_LEN || word_len > MAX_COMPOUND_LEN {
        return None;
    }
    let components = find_split(lexicon, word, MAX_COMPONENTS)?;
    if components.len() < 2 {
        // The whole word matching on its own isn't a real decomposition.
        return None;
    }
    Some(assemble_ipa(&components))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_component_split_succeeds_with_per_component_case_cascade() {
        // "handschuhfach" is long enough to attempt splitting; "Hand" keeps
        // its natural capitalization while "schuhfach" only hits the
        // lexicon under a title-cased "Schuhfach", exercising the
        // per-component case cascade.
        let lexicon = Lexicon::from_tsv("Hand\thant\nSchuhfach\tʃuːfax\n").unwrap();
        let ipa = decompose(&lexicon, "Handschuhfach").unwrap();
        assert_eq!(ipa, "hantʃuːfax");
    }

    #[test]
    fn three_component_split_succeeds() {
        // "autobahnschule" (14 chars) clears MIN_COMPOUND_LEN (12); a
        // 12-char word would be rejected by the length gate before the
        // split search even runs.
        let lexicon = Lexicon::from_tsv("auto\taʊto\nbahn\tban\nschule\tʃuːlə\n").unwrap();
        let ipa = decompose(&lexicon, "autobahnschule").unwrap();
        assert_eq!(ipa, "aʊtobanʃuːlə");
    }

    #[test]
    fn backtracks_from_a_dead_end_longest_match() {
        // At position 0, "handschuh" (9 chars) is the longest prefix that
        // hits the lexicon, but its remainder "fach" (4 chars) has no
        // lexicon entry of its own and is too short to be split further
        // (4 chars < 2 * MIN_COMPONENT_LEN), so find_split backtracks to
        // the shorter "hand" (4 chars) prefix, whose remainder
        // "schuhfach" (9 chars) hits the lexicon directly.
        let lexicon =
            Lexicon::from_tsv("handschuh\thantʃuː\nhand\thant\nschuhfach\tʃuːfax\n").unwrap();
        let ipa = decompose(&lexicon, "handschuhfach").unwrap();
        assert_eq!(ipa, "hantʃuːfax");
    }

    #[test]
    fn no_valid_split_returns_none() {
        // A word long enough to attempt splitting, but with only one of
        // its would-be components in the lexicon, has no valid full
        // decomposition.
        let lexicon = Lexicon::from_tsv("hand\thant\n").unwrap();
        assert_eq!(decompose(&lexicon, "handschuhfach"), None);
    }

    #[test]
    fn word_under_min_compound_len_is_never_split() {
        // "handschuh" is 9 chars, under MIN_COMPOUND_LEN (12), even though
        // both components exist in the lexicon.
        let lexicon = Lexicon::from_tsv("hand\thant\nschuh\tʃuː\n").unwrap();
        assert_eq!(decompose(&lexicon, "handschuh"), None);
    }

    #[test]
    fn word_at_exact_min_compound_len_is_never_split() {
        // "handelschuhe" is exactly 12 chars (== MIN_COMPOUND_LEN), so the
        // "<=" boundary in `decompose` must reject it even though it
        // splits cleanly into two lexicon-hitting 6-char components.
        let lexicon = Lexicon::from_tsv("handel\thandl\nschuhe\tʃuːə\n").unwrap();
        assert_eq!(decompose(&lexicon, "handelschuhe"), None);
    }

    #[test]
    fn word_over_max_compound_len_is_never_split() {
        // A word one codepoint over MAX_COMPOUND_LEN (40) is rejected by
        // the length gate before the split search even runs, even though
        // it would otherwise split cleanly.
        let long_prefix = "a".repeat(MAX_COMPOUND_LEN - 3);
        let word = format!("{long_prefix}bahn");
        let lexicon = Lexicon::from_tsv(&format!("{long_prefix}\tx\nbahn\tban\n")).unwrap();
        assert_eq!(word.chars().count(), MAX_COMPOUND_LEN + 1);
        assert_eq!(decompose(&lexicon, &word), None);
    }

    #[test]
    fn components_shorter_than_min_component_len_are_rejected() {
        // "a" and "utobahnhofstrasse" would technically cover the word, but
        // a 1-char component is below MIN_COMPONENT_LEN, so the search must
        // not accept it as a split point.
        let lexicon = Lexicon::from_tsv("a\ta\nutobahnhofstrasse\tuːtoːbaːnhoːfʃtraːsə\n").unwrap();
        assert_eq!(decompose(&lexicon, "autobahnhofstrasse"), None);
    }

    #[test]
    fn max_components_bound_is_enforced() {
        // A 5-component split ("auto"+"bahn"+"turm"+"haus"+"gang") exceeds
        // MAX_COMPONENTS (4), so no split should be found even though every
        // individual component is in the lexicon.
        let lexicon =
            Lexicon::from_tsv("auto\taʊto\nbahn\tban\nturm\ttʊʁm\nhaus\thaʊ̯s\ngang\tɡaŋ\n")
                .unwrap();
        assert_eq!(decompose(&lexicon, "autobahnturmhausgang"), None);
    }

    #[test]
    fn first_component_keeps_primary_stress_later_components_downgrade() {
        // Verifies assemble_ipa downgrades every component after the first
        // from primary (ˈ) to secondary (ˌ) stress.
        let lexicon = Lexicon::from_tsv("hand\tˈhant\nschuhfach\tˈʃuːfax\n").unwrap();
        let ipa = decompose(&lexicon, "handschuhfach").unwrap();
        assert_eq!(ipa, "ˈhantˌʃuːfax");
    }

    #[test]
    fn later_component_without_stress_mark_is_unchanged() {
        // Verifies a later component with no stress mark at all passes
        // through assemble_ipa unmodified.
        let lexicon = Lexicon::from_tsv("hand\tˈhant\nschuhfach\tʃuːfax\n").unwrap();
        let ipa = decompose(&lexicon, "handschuhfach").unwrap();
        assert_eq!(ipa, "ˈhantʃuːfax");
    }

    #[test]
    fn whole_word_lexicon_hit_is_not_treated_as_a_split() {
        // Long enough to pass the length gate, and happens to hit the
        // lexicon whole — decompose must still return None, since a single
        // "component" covering the whole word isn't a real decomposition.
        let lexicon = Lexicon::from_tsv("handschuhfacharoo\thant\n").unwrap();
        assert_eq!(decompose(&lexicon, "handschuhfacharoo"), None);
    }

    #[test]
    fn decompose_splits_at_multibyte_character_boundary() {
        // "straßebahnhof" splits right after "straße", whose ß is a
        // 2-byte-in-UTF-8 character — verifies find_split's byte-offset
        // slicing (via char_byte_positions) lands on a valid boundary
        // instead of panicking mid-codepoint.
        let lexicon = Lexicon::from_tsv("straße\tʃtʁaːsə\nbahnhof\tbaːnhoːf\n").unwrap();
        let ipa = decompose(&lexicon, "straßebahnhof").unwrap();
        assert_eq!(ipa, "ʃtʁaːsəbaːnhoːf");
    }

    #[test]
    fn char_byte_positions_handles_multibyte_letters() {
        // Verifies char_byte_positions returns byte offsets (not char
        // indices), so callers can safely slice at each boundary even
        // around the 2-byte ö.
        let positions = char_byte_positions("schön");
        assert_eq!(positions, vec![0, 1, 2, 3, 5, 6]);
    }
}
