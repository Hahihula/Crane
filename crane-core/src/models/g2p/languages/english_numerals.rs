// SPDX-License-Identifier: MIT

//! English cardinal number-to-words conversion, for
//! [`numeral_expand`](crate::models::g2p::numeral_expand)'s pre-lexicon text
//! normalization pass.
//!
//! Output is space-separated (`"twenty one"`, not `"twenty-one"`), not
//! hyphenated, since the result is fed back through
//! [`text_to_ipa`](super::english::EnglishG2p::text_to_ipa)'s ordinary
//! whitespace tokenizer — each spelled-out word then resolves through the
//! lexicon/OOV/rules chain independently, rather than needing a dedicated
//! lexicon entry for every hyphenated compound number.

use crate::models::g2p::numeral_expand::NumeralWords;

/// Words for 0-19, where the position is the value itself.
const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
];

/// Words for the tens digit (index 2-9); indices 0-1 are unused since values
/// under 20 are handled directly by [`ONES`].
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Scale words for each group of three digits, least significant first.
/// Index 0 (the ones/hundreds group) has no scale word. `u64::MAX` is about
/// 18.4 quintillion, so seven groups (up to "quintillion") cover every
/// representable value.
const SCALES: [&str; 7] =
    ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"];

/// English [`NumeralWords`] implementation: spells out cardinal numbers in
/// standard American English form (no "and" before the final group, e.g.
/// `"one hundred one"` not `"one hundred and one"`).
pub struct EnglishNumerals;

impl NumeralWords for EnglishNumerals {
    fn cardinal(&self, n: u64) -> String {
        if n == 0 {
            return ONES[0].to_string();
        }

        let mut groups = Vec::new();
        let mut remaining = n;
        while remaining > 0 {
            groups.push((remaining % 1000) as u32);
            remaining /= 1000;
        }

        groups
            .iter()
            .enumerate()
            .rev()
            .filter(|&(_, &group)| group != 0)
            .map(|(scale, &group)| {
                let words = three_digit_words(group);
                if scale == 0 { words } else { format!("{words} {}", SCALES[scale]) }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Spells out `n` (0-999) as hundreds plus a two-digit remainder.
fn three_digit_words(n: u32) -> String {
    debug_assert!(n < 1000);
    let hundreds = n / 100;
    let rest = n % 100;

    let mut parts = Vec::new();
    if hundreds > 0 {
        parts.push(format!("{} hundred", ONES[hundreds as usize]));
    }
    if rest > 0 {
        parts.push(two_digit_words(rest));
    }
    parts.join(" ")
}

/// Spells out `n` (0-99).
fn two_digit_words(n: u32) -> String {
    debug_assert!(n < 100);
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    let tens = TENS[(n / 10) as usize];
    let ones = n % 10;
    if ones == 0 { tens.to_string() } else { format!("{tens} {}", ONES[ones as usize]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies zero spells out as "zero" rather than an empty string.
    #[test]
    fn zero() {
        assert_eq!(EnglishNumerals.cardinal(0), "zero");
    }

    // Verifies single-digit and teen values use the direct ONES lookup.
    #[test]
    fn ones_and_teens() {
        assert_eq!(EnglishNumerals.cardinal(5), "five");
        assert_eq!(EnglishNumerals.cardinal(11), "eleven");
        assert_eq!(EnglishNumerals.cardinal(19), "nineteen");
    }

    // Verifies a round tens value has no trailing ones word.
    #[test]
    fn round_tens() {
        assert_eq!(EnglishNumerals.cardinal(20), "twenty");
        assert_eq!(EnglishNumerals.cardinal(90), "ninety");
    }

    // Verifies a tens-plus-ones value joins both words with a space, not a
    // hyphen.
    #[test]
    fn tens_and_ones() {
        assert_eq!(EnglishNumerals.cardinal(21), "twenty one");
        assert_eq!(EnglishNumerals.cardinal(99), "ninety nine");
    }

    // Verifies a round hundred has no trailing tens/ones words.
    #[test]
    fn round_hundred() {
        assert_eq!(EnglishNumerals.cardinal(100), "one hundred");
    }

    // Verifies a hundred plus a low remainder omits "and", per American
    // convention.
    #[test]
    fn hundred_plus_ones_no_and() {
        assert_eq!(EnglishNumerals.cardinal(105), "one hundred five");
    }

    // Verifies a hundred plus a two-digit remainder spells out all three
    // parts.
    #[test]
    fn hundred_plus_tens_and_ones() {
        assert_eq!(EnglishNumerals.cardinal(121), "one hundred twenty one");
    }

    // Verifies the largest three-digit value spells out correctly.
    #[test]
    fn max_three_digit() {
        assert_eq!(EnglishNumerals.cardinal(999), "nine hundred ninety nine");
    }

    // Verifies a round thousand has no trailing group.
    #[test]
    fn round_thousand() {
        assert_eq!(EnglishNumerals.cardinal(1000), "one thousand");
    }

    // Verifies a thousand plus a low remainder still includes the scale
    // word before the remainder.
    #[test]
    fn thousand_plus_ones() {
        assert_eq!(EnglishNumerals.cardinal(1001), "one thousand one");
    }

    // Verifies a year-style four-digit number spells out every group.
    #[test]
    fn year_style_number() {
        assert_eq!(EnglishNumerals.cardinal(1891), "one thousand eight hundred ninety one");
    }

    // Verifies a middle thousands group is fully spelled out, not skipped,
    // when the lowest group is zero.
    #[test]
    fn thousands_with_zero_low_group() {
        assert_eq!(EnglishNumerals.cardinal(21_000), "twenty one thousand");
    }

    // Verifies a round million has no trailing groups.
    #[test]
    fn round_million() {
        assert_eq!(EnglishNumerals.cardinal(1_000_000), "one million");
    }

    // Verifies every group below the highest is included even when some
    // groups are zero, by spanning millions, thousands, and ones.
    #[test]
    fn multi_group_with_gaps() {
        assert_eq!(
            EnglishNumerals.cardinal(1_002_003),
            "one million two thousand three"
        );
    }

    // Verifies the largest u64 value resolves without panicking and uses
    // the highest scale word (quintillion).
    #[test]
    fn max_u64_uses_quintillion_scale() {
        let result = EnglishNumerals.cardinal(u64::MAX);
        assert!(result.starts_with("eighteen quintillion"));
    }
}
