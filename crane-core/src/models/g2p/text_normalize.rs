// SPDX-License-Identifier: MIT

//! Text normalization for the G2P pipeline.
//!
//! Three pure functions sit between raw input text and lexicon lookup:
//! splitting text into words, normalizing a word for dictionary lookup, and
//! normalizing a `CMUdict`-style key by stripping alternate-pronunciation
//! markers.

use std::borrow::Cow;

/// Splits `text` on runs of ASCII whitespace, returning non-empty tokens.
///
/// Whitespace is `u8::is_ascii_whitespace` (space, tab, `\n`, `\r`, form
/// feed, vertical tab). Non-ASCII bytes are never whitespace, so multi-byte
/// UTF-8 sequences pass through intact as part of a token. Punctuation
/// attached to a word (e.g. a trailing comma) is not stripped here — that is
/// [`normalize_word_for_lookup`]'s job.
#[must_use]
pub fn split_text_to_words(text: &str) -> Vec<&str> {
    text.split_ascii_whitespace().collect()
}

/// Returns `true` if `c` counts as a word character for trimming purposes:
/// an ASCII alphanumeric, or any non-ASCII codepoint.
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || !c.is_ascii()
}

/// Trims non-word characters from both edges of `token`, preserving case.
///
/// Multi-byte UTF-8 codepoints (accents, IPA symbols, etc.) always count as
/// word characters and are never trimmed. Interior characters are never
/// trimmed, only the leading/trailing run of non-word characters (see
/// [`is_word_char`]).
///
/// Returns an empty string if the token has no word characters at all.
///
/// # Panics
///
/// Never panics: if a leading word character is found, a trailing one is
/// guaranteed to exist too (the same character, at minimum).
#[must_use]
pub fn trim_edge_punctuation(token: &str) -> &str {
    let Some((start, _)) = token.char_indices().find(|&(_, c)| is_word_char(c)) else {
        return "";
    };
    let (end, last_char) = token
        .char_indices()
        .rev()
        .find(|&(_, c)| is_word_char(c))
        .expect("start match implies a matching char exists from the end too");

    &token[start..end + last_char.len_utf8()]
}

/// Normalizes a word token for lexicon lookup.
///
/// ASCII bytes are lowercased (non-ASCII bytes are left as-is — no Unicode
/// case folding); then non-word characters are trimmed from both edges via
/// [`trim_edge_punctuation`].
///
/// Returns an empty string if the token has no word characters at all.
///
/// Returns a borrowed slice when `token` is already trimmed and contains no
/// uppercase ASCII bytes, avoiding an allocation on the common case.
#[must_use]
pub fn normalize_word_for_lookup(token: &str) -> Cow<'_, str> {
    let trimmed = trim_edge_punctuation(token);
    if trimmed.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(trimmed.to_ascii_lowercase())
    } else {
        Cow::Borrowed(trimmed)
    }
}

/// Normalizes a `CMUdict`-style dictionary key.
///
/// ASCII bytes are lowercased; then a trailing `(N)` alternate-pronunciation
/// marker is stripped, where `N` is one or more ASCII digits (e.g.
/// `"hello(2)"` -> `"hello"`). Unlike [`normalize_word_for_lookup`], edges
/// are not otherwise trimmed — hyphens and apostrophes are preserved.
#[must_use]
pub fn normalize_grapheme_key(word: &str) -> String {
    if word.ends_with(')')
        && let Some(open) = word.rfind('(')
    {
        let inner = &word[open + 1..word.len() - 1];
        if !inner.is_empty() && inner.bytes().all(|b| b.is_ascii_digit()) {
            return word[..open].to_ascii_lowercase();
        }
    }

    word.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_simple_words() {
        assert_eq!(split_text_to_words("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn split_collapses_runs_of_whitespace() {
        assert_eq!(
            split_text_to_words("  hello  \t world  \n"),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn split_empty_string_yields_no_tokens() {
        assert_eq!(split_text_to_words(""), Vec::<&str>::new());
    }

    #[test]
    fn split_whitespace_only_yields_no_tokens() {
        assert_eq!(split_text_to_words("   \t\n"), Vec::<&str>::new());
    }

    #[test]
    fn split_keeps_attached_punctuation() {
        assert_eq!(
            split_text_to_words("Hello, world!"),
            vec!["Hello,", "world!"]
        );
    }

    #[test]
    fn split_preserves_non_ascii_tokens() {
        assert_eq!(split_text_to_words("café résumé"), vec!["café", "résumé"]);
    }

    #[test]
    fn trim_edge_punctuation_preserves_case() {
        assert_eq!(trim_edge_punctuation("Haus."), "Haus");
    }

    #[test]
    fn trim_edge_punctuation_trims_both_edges_keeps_interior() {
        assert_eq!(trim_edge_punctuation("--don't--"), "don't");
    }

    #[test]
    fn trim_edge_punctuation_all_punctuation_is_empty() {
        assert_eq!(trim_edge_punctuation("---"), "");
    }

    #[test]
    fn trim_edge_punctuation_empty_string_is_empty() {
        assert_eq!(trim_edge_punctuation(""), "");
    }

    #[test]
    fn normalize_word_basic_lowercase() {
        assert_eq!(normalize_word_for_lookup("Hello"), "hello");
    }

    #[test]
    fn normalize_word_trims_trailing_punctuation() {
        assert_eq!(normalize_word_for_lookup("Hello,"), "hello");
    }

    #[test]
    fn normalize_word_trims_both_edges_keeps_interior() {
        assert_eq!(normalize_word_for_lookup("--don't--"), "don't");
    }

    #[test]
    fn normalize_word_keeps_non_ascii_untrimmed_and_uncased() {
        assert_eq!(normalize_word_for_lookup("café"), "café");
    }

    #[test]
    fn normalize_word_trims_ascii_punctuation_after_multibyte_char() {
        assert_eq!(normalize_word_for_lookup("café!"), "café");
    }

    #[test]
    fn normalize_word_empty_string_is_empty() {
        assert_eq!(normalize_word_for_lookup(""), "");
    }

    #[test]
    fn normalize_word_all_punctuation_is_empty() {
        assert_eq!(normalize_word_for_lookup("---"), "");
    }

    #[test]
    fn normalize_word_mixed_quotes() {
        assert_eq!(normalize_word_for_lookup("'Hello!'"), "hello");
    }

    #[test]
    fn normalize_word_keeps_digits() {
        assert_eq!(normalize_word_for_lookup("abc123"), "abc123");
    }

    #[test]
    fn normalize_key_basic_lowercase() {
        assert_eq!(normalize_grapheme_key("HELLO"), "hello");
    }

    #[test]
    fn normalize_key_strips_alternate_marker() {
        assert_eq!(normalize_grapheme_key("hello(2)"), "hello");
    }

    #[test]
    fn normalize_key_strips_marker_with_leading_zero() {
        assert_eq!(normalize_grapheme_key("hello(02)"), "hello");
    }

    #[test]
    fn normalize_key_empty_string_is_empty() {
        assert_eq!(normalize_grapheme_key(""), "");
    }

    #[test]
    fn normalize_key_strips_marker_keeps_hyphen() {
        assert_eq!(normalize_grapheme_key("re-entry(13)"), "re-entry");
    }

    #[test]
    fn normalize_key_keeps_non_digit_parens() {
        assert_eq!(normalize_grapheme_key("foo(bar)"), "foo(bar)");
    }

    #[test]
    fn normalize_key_keeps_empty_parens() {
        assert_eq!(normalize_grapheme_key("foo()"), "foo()");
    }

    #[test]
    fn normalize_key_no_parens_unchanged() {
        assert_eq!(normalize_grapheme_key("hello"), "hello");
    }

    #[test]
    fn normalize_key_only_strips_outermost_trailing_marker() {
        assert_eq!(normalize_grapheme_key("foo(1)(2)"), "foo(1)");
    }
}
