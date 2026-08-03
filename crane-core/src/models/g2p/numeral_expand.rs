// SPDX-License-Identifier: MIT

//! Language-agnostic numeral expansion for the G2P pipeline.
//!
//! [`expand_numerals`] finds digit runs in running text and replaces each
//! with its cardinal word form, produced by a language-specific
//! [`NumeralWords`] implementation. This runs as a text-normalization pass
//! ahead of lexicon lookup, so downstream G2P stages never see raw digits.
//!
//! Only ASCII digits (`0`-`9`) are recognized; text using native digit
//! scripts (Arabic-Indic, Devanagari, full-width, etc.) passes through
//! untouched.

use std::borrow::Cow;

/// Produces number words for a specific language.
pub trait NumeralWords {
    /// Returns the cardinal (counting) word form of `n` (e.g. `21` ->
    /// `"twenty-one"` in English, `"einundzwanzig"` in German).
    fn cardinal(&self, n: u64) -> String;
}

/// Returns `true` if `c` counts as a word character adjacent to a digit run:
/// any Unicode letter, in any script. Digits themselves are handled
/// separately by the run scan and are not word characters here; punctuation,
/// whitespace, and symbols (ASCII or not) are also not word characters.
fn is_word_char(c: char) -> bool {
    c.is_alphabetic()
}

/// Expands standalone digit runs in `text` into cardinal number words via
/// `words`.
///
/// A digit run is "standalone" when it is not immediately adjacent (on
/// either side) to an ASCII letter or non-ASCII codepoint — e.g. `"21"` in
/// `"I have 21 cats"` expands, but the digits in `"abc123"` do not, since
/// `expand_numerals` has no way to know where an alphanumeric identifier
/// ends and a number begins. Punctuation and whitespace are not word characters,
/// so `"(42)"` and `"21,"` still expand.
///
/// Digit runs that don't fit in a `u64` are left untouched, since
/// [`NumeralWords::cardinal`] has no representation for them.
///
/// Returns a borrowed slice unchanged when no digit run is found at all,
/// which is the overwhelming majority of real text — ordinary sentences pay
/// zero allocation cost for this pass.
///
/// # Limitations
///
/// Each maximal digit run is expanded independently, so multi-part numerals
/// are not recognized as a single value:
/// - Decimal points split the two sides, e.g. `"3.14"` expands as if it were
///   `"3"` and `"14"` separately, not "three point one four".
/// - A leading `-` is not consumed, e.g. `"-5"` expands to a negated cardinal
///   of `5` rather than "negative five".
/// - Grouping separators split runs the same way, e.g. `"1,000"` expands `1`
///   and `0` independently, not "one thousand".
/// - Time-of-day separators split runs the same way, e.g. `"14:30"` expands
///   `14` and `30` independently, not as a single time value.
#[must_use]
pub fn expand_numerals<'a>(text: &'a str, words: &dyn NumeralWords) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let mut result = String::new();
    let mut last_copied = 0;
    let mut i = 0;

    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }

        let run_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let run_end = i;

        let preceded_by_word_char = text[..run_start]
            .chars()
            .next_back()
            .is_some_and(is_word_char);
        let followed_by_word_char = text[run_end..].chars().next().is_some_and(is_word_char);

        if preceded_by_word_char || followed_by_word_char {
            continue;
        }

        let Ok(n) = text[run_start..run_end].parse::<u64>() else {
            continue;
        };

        result.push_str(&text[last_copied..run_start]);
        result.push_str(&words.cardinal(n));
        last_copied = run_end;
    }

    if last_copied == 0 {
        Cow::Borrowed(text)
    } else {
        result.push_str(&text[last_copied..]);
        Cow::Owned(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial `NumeralWords` stub for testing the scanner in isolation from
    /// any real language's number-word rules.
    struct EchoNumerals;

    impl NumeralWords for EchoNumerals {
        fn cardinal(&self, n: u64) -> String {
            format!("<{n}>")
        }
    }

    // Verifies plain text with no digits is returned as the exact same
    // borrowed slice, proving the zero-allocation fast path.
    #[test]
    fn no_digits_returns_borrowed() {
        let text = "hello world";
        let result = expand_numerals(text, &EchoNumerals);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello world");
    }

    // Verifies a single standalone digit run expands correctly.
    #[test]
    fn single_digit_run_expands() {
        assert_eq!(expand_numerals("I have 21 cats", &EchoNumerals), "I have <21> cats");
    }

    // Verifies multiple digit runs in one string all expand.
    #[test]
    fn multiple_digit_runs_expand() {
        assert_eq!(expand_numerals("12 and 34", &EchoNumerals), "<12> and <34>");
    }

    // Verifies digits embedded inside an alphanumeric token are left alone,
    // since there's no reliable way to tell an identifier from a number.
    #[test]
    fn digits_embedded_in_word_are_not_expanded() {
        assert_eq!(expand_numerals("abc123", &EchoNumerals), "abc123");
        assert_eq!(expand_numerals("123abc", &EchoNumerals), "123abc");
    }

    // Verifies a digit run at the very start or end of the string still
    // expands (no word character on the missing side).
    #[test]
    fn digit_run_at_string_edges_expands() {
        assert_eq!(expand_numerals("21 cats", &EchoNumerals), "<21> cats");
        assert_eq!(expand_numerals("cats 21", &EchoNumerals), "cats <21>");
    }

    // Verifies leading zeros are preserved in the parsed value, not treated
    // as octal or truncated.
    #[test]
    fn leading_zeros_preserved_in_value() {
        assert_eq!(expand_numerals("007 agent", &EchoNumerals), "<7> agent");
    }

    // Verifies a digit run too large for u64 is left untouched rather than
    // panicking or silently wrapping.
    #[test]
    fn overflowing_digit_run_left_as_is() {
        let text = "id 99999999999999999999 done";
        assert_eq!(expand_numerals(text, &EchoNumerals), text);
    }

    // Verifies a string that is only digits expands fully.
    #[test]
    fn only_digits_string_expands() {
        assert_eq!(expand_numerals("42", &EchoNumerals), "<42>");
    }

    // Verifies punctuation adjacent to a digit run does not block expansion,
    // since punctuation is not a word character.
    #[test]
    fn punctuation_adjacent_digits_expand() {
        assert_eq!(expand_numerals("21,", &EchoNumerals), "<21>,");
        assert_eq!(expand_numerals("(42)", &EchoNumerals), "(<42>)");
    }

    // Verifies non-ASCII letters adjacent to a digit run block expansion,
    // matching the ASCII-letter case.
    #[test]
    fn non_ascii_adjacent_digits_not_expanded() {
        assert_eq!(expand_numerals("café21", &EchoNumerals), "café21");
    }

    // Verifies non-ASCII punctuation adjacent to a digit run does not block
    // expansion, since only letters (not punctuation) count as word
    // characters, regardless of ASCII-ness.
    #[test]
    fn non_ascii_punctuation_adjacent_digits_expand() {
        assert_eq!(expand_numerals("«21»", &EchoNumerals), "«<21>»");
    }

    // Verifies a non-breaking space adjacent to a digit run does not block
    // expansion, since whitespace is not a word character.
    #[test]
    fn non_breaking_space_adjacent_digits_expand() {
        assert_eq!(expand_numerals("\u{00A0}42", &EchoNumerals), "\u{00A0}<42>");
    }

    // Verifies a decimal point currently splits the two sides into separate
    // expansions rather than being recognized as one value (see the
    // `# Limitations` section on `expand_numerals`).
    #[test]
    fn decimal_point_splits_runs() {
        assert_eq!(expand_numerals("3.14", &EchoNumerals), "<3>.<14>");
    }

    // Verifies a leading minus sign is not consumed into the cardinal, so
    // the sign and magnitude expand independently (see `# Limitations`).
    #[test]
    fn negative_sign_not_consumed() {
        assert_eq!(expand_numerals("-5", &EchoNumerals), "-<5>");
    }

    // Verifies a thousands separator splits the two sides into separate
    // expansions rather than being recognized as one value (see
    // `# Limitations`).
    #[test]
    fn thousands_separator_splits_runs() {
        assert_eq!(expand_numerals("1,000", &EchoNumerals), "<1>,<0>");
    }

    // Verifies a time-of-day separator splits the two sides into separate
    // expansions rather than being recognized as one value (see
    // `# Limitations`).
    #[test]
    fn time_separator_splits_runs() {
        assert_eq!(expand_numerals("14:30", &EchoNumerals), "<14>:<30>");
    }

    // Verifies an empty string is handled without panicking.
    #[test]
    fn empty_string_returns_borrowed() {
        let result = expand_numerals("", &EchoNumerals);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "");
    }
}
