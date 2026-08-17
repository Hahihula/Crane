// SPDX-License-Identifier: MIT

//! German cardinal number-to-words conversion, for
//! [`numeral_expand`](crate::models::g2p::numeral_expand)'s pre-lexicon text
//! normalization pass.
//!
//! Unlike English, German compounds each three-digit group's tens and ones
//! into a single word with the ones digit *before* the tens digit, joined by
//! `und` (`21` -> `"einundzwanzig"`, not `"zwanzigeins"`). The digit `1` has
//! two forms: the bound form `"ein"` used whenever more material follows in
//! the same group or scale (`"einundzwanzig"`, `"einhundert"`, `"ein
//! tausend"`), and the standalone form `"eins"` used only when `1` is the
//! very last thing spelled out with nothing following it at all (`"eins"`,
//! `"einhundert eins"`). Scale words from `Million` upward take grammatical
//! gender and plural agreement (`"eine Million"` vs `"zwei Millionen"`);
//! `tausend` does neither.
//!
//! Like English's [`EnglishNumerals`](super::english_numerals::EnglishNumerals),
//! output is space-separated at group boundaries so each spelled-out word
//! resolves independently through the lexicon/rules chain, rather than
//! needing a dedicated lexicon entry for every compound number.

use crate::models::g2p::numeral_expand::NumeralWords;

/// Bound forms of the digits 1-9, used everywhere except when `1` stands
/// entirely alone (see [`GermanNumerals`]'s module docs). Index 0 is unused
/// (never indexed with a zero digit by callers).
const ONES_BOUND: [&str; 10] = [
    "", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
];

/// Words for 10-19, including the irregular contractions `sechzehn` (not
/// `sechszehn`) and `siebzehn` (not `siebenzehn`).
const TEENS: [&str; 10] = [
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
];

/// Words for the tens digit (index 2-9), including the irregular contractions
/// `dreißig`, `sechzig`, `siebzig`; indices 0-1 are unused since values under
/// 20 are handled directly by [`ONES_BOUND`]/[`TEENS`].
const TENS: [&str; 10] = [
    "", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig",
];

/// Singular/plural noun forms for each group of three digits from the
/// million upward, least significant first. `tausend` (the first scale
/// above the bare ones/hundreds group) is handled separately since it is
/// grammatically an invariant numeral, not a gendered/pluralizable noun like
/// these. `u64::MAX` is about 18.4 quintillion, so five entries (up to
/// `Trillion`, the German long-scale 10^18) cover every representable value.
const LARGE_SCALES: [(&str, &str); 5] = [
    ("Million", "Millionen"),
    ("Milliarde", "Milliarden"),
    ("Billion", "Billionen"),
    ("Billiarde", "Billiarden"),
    ("Trillion", "Trillionen"),
];

/// German [`NumeralWords`] implementation: spells out cardinal numbers using
/// standard German ones-before-tens compounding and long-scale (`Million`,
/// `Milliarde`, `Billion`, ...) scale words.
pub struct GermanNumerals;

impl NumeralWords for GermanNumerals {
    fn cardinal(&self, n: u64) -> String {
        if n == 0 {
            return "null".to_string();
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
            .map(|(scale, &group)| match scale {
                // The lowest group is always the last spelled out, so `1`
                // here (with no accompanying tens digit) is the standalone
                // "eins" case; every other scale always has a scale word
                // following it, so it always uses the bound "ein" form.
                0 => three_digit_words(group, true),
                1 => format!("{} tausend", three_digit_words(group, false)),
                _ => {
                    debug_assert!(scale < LARGE_SCALES.len() + 2);
                    let (singular, plural) = LARGE_SCALES[scale - 2];
                    if group == 1 {
                        format!("eine {singular}")
                    } else {
                        format!("{} {plural}", three_digit_words(group, false))
                    }
                },
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Spells out `n` (1-999) as an `"Xhundert"` compound plus a two-digit
/// remainder. `standalone_one` is forwarded unchanged to the remainder even
/// when a hundreds prefix is present, since a bare `1` remainder (e.g. `101`)
/// still uses "eins" — the hundreds prefix doesn't change whether anything
/// follows the ones digit.
fn three_digit_words(n: u32, standalone_one: bool) -> String {
    debug_assert!(n > 0 && n < 1000);
    let hundreds = n / 100;
    let rest = n % 100;

    let mut parts = Vec::new();
    if hundreds > 0 {
        parts.push(format!("{}hundert", ONES_BOUND[hundreds as usize]));
    }
    if rest > 0 {
        parts.push(two_digit_words(rest, standalone_one));
    }
    parts.join(" ")
}

/// Spells out `n` (1-99). `standalone_one` selects `"eins"` over the bound
/// `"ein"` for the bare value `1` — irrelevant for every other value, since a
/// tens digit always pulls the ones digit into an `"...und<tens>"` compound
/// that needs the bound form regardless.
fn two_digit_words(n: u32, standalone_one: bool) -> String {
    debug_assert!(n > 0 && n < 100);
    if n < 10 {
        if n == 1 && standalone_one {
            return "eins".to_string();
        }
        return ONES_BOUND[n as usize].to_string();
    }
    if n < 20 {
        return TEENS[(n - 10) as usize].to_string();
    }
    let tens = TENS[(n / 10) as usize];
    let ones = n % 10;
    if ones == 0 {
        tens.to_string()
    } else {
        format!("{}und{tens}", ONES_BOUND[ones as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies zero spells out as "null" rather than an empty string.
    #[test]
    fn zero() {
        assert_eq!(GermanNumerals.cardinal(0), "null");
    }

    // Verifies the standalone form "eins" is used when 1 is spelled out with
    // nothing following it.
    #[test]
    fn one_uses_standalone_form() {
        assert_eq!(GermanNumerals.cardinal(1), "eins");
    }

    // Verifies teens use the direct TEENS lookup, including the irregular
    // contractions sechzehn/siebzehn.
    #[test]
    fn teens_including_irregulars() {
        assert_eq!(GermanNumerals.cardinal(11), "elf");
        assert_eq!(GermanNumerals.cardinal(16), "sechzehn");
        assert_eq!(GermanNumerals.cardinal(17), "siebzehn");
        assert_eq!(GermanNumerals.cardinal(19), "neunzehn");
    }

    // Verifies a round tens value has no trailing ones word, including the
    // irregular contraction dreißig.
    #[test]
    fn round_tens_including_irregular() {
        assert_eq!(GermanNumerals.cardinal(20), "zwanzig");
        assert_eq!(GermanNumerals.cardinal(30), "dreißig");
        assert_eq!(GermanNumerals.cardinal(90), "neunzig");
    }

    // Verifies a tens-plus-ones value compounds ones-before-tens with "und",
    // using the bound "ein" form for a ones digit of 1.
    #[test]
    fn ones_before_tens_compounding() {
        assert_eq!(GermanNumerals.cardinal(21), "einundzwanzig");
        assert_eq!(GermanNumerals.cardinal(99), "neunundneunzig");
    }

    // Verifies a round hundred compounds as one word with the bound "ein"
    // form, not "einshundert".
    #[test]
    fn round_hundred() {
        assert_eq!(GermanNumerals.cardinal(100), "einhundert");
    }

    // Verifies a hundred plus a bare remainder of 1 uses the standalone
    // "eins" form even with a hundreds prefix present.
    #[test]
    fn hundred_plus_one_uses_standalone_form() {
        assert_eq!(GermanNumerals.cardinal(101), "einhundert eins");
    }

    // Verifies a hundred plus a two-digit remainder spells out both parts.
    #[test]
    fn hundred_plus_tens_and_ones() {
        assert_eq!(GermanNumerals.cardinal(321), "dreihundert einundzwanzig");
    }

    // Verifies the largest three-digit value spells out correctly.
    #[test]
    fn max_three_digit() {
        assert_eq!(GermanNumerals.cardinal(999), "neunhundert neunundneunzig");
    }

    // Verifies a round thousand uses the bound "ein" form before the
    // invariant "tausend" scale word.
    #[test]
    fn round_thousand() {
        assert_eq!(GermanNumerals.cardinal(1000), "ein tausend");
    }

    // Verifies a thousand plus a bare remainder of 1 uses the standalone
    // "eins" form for the final group.
    #[test]
    fn thousand_plus_one() {
        assert_eq!(GermanNumerals.cardinal(1001), "ein tausend eins");
    }

    // Verifies a year-style four-digit number spells out every group.
    #[test]
    fn year_style_number() {
        assert_eq!(
            GermanNumerals.cardinal(1891),
            "ein tausend achthundert einundneunzig"
        );
    }

    // Verifies a middle thousands group is fully spelled out, not skipped,
    // when the lowest group is zero.
    #[test]
    fn thousands_with_zero_low_group() {
        assert_eq!(GermanNumerals.cardinal(21_000), "einundzwanzig tausend");
    }

    // Verifies a round million uses the feminine "eine" article, not the
    // bound "ein" form used for tausend.
    #[test]
    fn round_million_uses_feminine_article() {
        assert_eq!(GermanNumerals.cardinal(1_000_000), "eine Million");
    }

    // Verifies a million count above 1 uses the plural noun form.
    #[test]
    fn plural_million() {
        assert_eq!(GermanNumerals.cardinal(2_000_000), "zwei Millionen");
    }

    // Verifies every group below the highest is included even when some
    // groups are zero, by spanning millions, thousands, and ones.
    #[test]
    fn multi_group_with_gaps() {
        assert_eq!(
            GermanNumerals.cardinal(1_002_003),
            "eine Million zwei tausend drei"
        );
    }

    // Verifies a bare-1 remainder before "tausend" uses the bound "ein" form,
    // not the standalone "eins" form the same remainder would use as the
    // very last group.
    #[test]
    fn hundred_plus_one_before_thousand_scale_uses_bound_form() {
        assert_eq!(GermanNumerals.cardinal(101_000), "einhundert ein tausend");
    }

    // Verifies a bare-1 remainder before a plural large-scale noun uses the
    // bound "ein" form, not the standalone "eins" form or the feminine
    // "eine" article (which only applies when the whole group is exactly 1).
    #[test]
    fn hundred_plus_one_before_million_scale_uses_bound_form() {
        assert_eq!(
            GermanNumerals.cardinal(101_000_000),
            "einhundert ein Millionen"
        );
    }

    // Verifies the largest u64 value resolves without panicking and uses the
    // highest scale word (Trillion, German long-scale 10^18).
    #[test]
    fn max_u64_uses_trillion_scale() {
        let result = GermanNumerals.cardinal(u64::MAX);
        assert!(result.starts_with("achtzehn Trillionen"));
    }
}
