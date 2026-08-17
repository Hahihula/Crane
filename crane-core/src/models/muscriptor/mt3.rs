//! MT3-style MIDI event tokenizer (audiocraft-trans / YourMT3+ lineage).
//!
//! The vocabulary is fixed at construction time and intentionally never
//! trained against new tokens — extending it would invalidate every
//! published checkpoint. The layout is byte-for-byte identical to the
//! upstream `muscriptor.tokenizer.mt3.MT3Tokenizer`:
//!
//! ```text
//! indices [0,    3)  → PAD / EOS / UNK  (special tokens)
//! indices [3, 1004)  → shift 0..1000
//! indices [1004, 1132) → pitch 0..127
//! indices [1132, 1134) → velocity 0..1
//! indices [1134, 1135) → tie     (single token: tie)
//! indices [1135, 1265) → program 0..129
//! indices [1265, 1393) → drum 0..127
//! ```
//!
//! Total vocabulary = 1393, matching the upstream `config.json::card`.
//! The model's embedding table has `card + 1 = 1394` rows; the `+1` is
//! reserved for `zero_idx = -1` (returned as a zero vector by the
//! `ScaledEmbedding` wrapper) and is never produced by the generator.

use std::collections::HashMap;

/// Special-token order: index 0 = PAD, 1 = EOS, 2 = UNK.
const SPECIAL_TOKENS: [&str; 3] = ["PAD", "EOS", "UNK"];

/// Number of `shift` tokens (`shift` 0..1000 → 1001 entries).
const MAX_SHIFT_STEPS: usize = 1001;

/// Total vocabulary size.
pub const CARD: usize = 1393;

/// Public aliases matching the upstream `Model.{initial,zero,ungenerated}_token_id`.
pub const EOS_ID: u32 = 1;
pub const INITIAL_TOKEN_ID: u32 = 1393;
pub const ZERO_TOKEN_ID: i32 = -1;
pub const UNGENERATED_TOKEN_ID: i32 = -2;

/// Same group-name ↔ ID table as the upstream `MT3_FULL_PLUS_GROUP_NAMES`.
pub const MT3_FULL_PLUS_GROUP_NAMES: &[(&str, u8)] = &[
    ("acoustic_piano", 0),
    ("electric_piano", 1),
    ("chromatic_percussion", 2),
    ("organ", 3),
    ("acoustic_guitar", 4),
    ("clean_electric_guitar", 5),
    ("distorted_electric_guitar", 6),
    ("acoustic_bass", 7),
    ("electric_bass", 8),
    ("violin", 9),
    ("viola", 10),
    ("cello", 11),
    ("contrabass", 12),
    ("orchestral_harp", 13),
    ("timpani", 14),
    ("string_ensemble", 15),
    ("synth_strings", 16),
    ("voice", 17),
    ("orchestra_hit", 18),
    ("trumpet", 19),
    ("trombone", 20),
    ("tuba", 21),
    ("french_horn", 22),
    ("brass_section", 23),
    ("soprano_and_alto_sax", 24),
    ("tenor_sax", 25),
    ("baritone_sax", 26),
    ("oboe", 27),
    ("english_horn", 28),
    ("bassoon", 29),
    ("clarinet", 30),
    ("flutes", 31),
    ("synth_lead", 32),
    ("synth_pad", 33),
    ("drums", 36),
];

/// Sentinel program number used for drum notes (matches upstream `DRUM_PROGRAM`).
pub const DRUM_PROGRAM: u8 = 128;

/// One decoded MT3 event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Pad,
    Eos,
    Unk,
    Shift(u16),
    Pitch(u8),
    Velocity(bool),
    Tie,
    Program(u8),
    Drum(u8),
}

impl Token {
    /// Stable token ID in `[0, CARD)`.
    #[must_use]
    pub fn id(self) -> u32 {
        match self {
            Token::Pad => 0,
            Token::Eos => 1,
            Token::Unk => 2,
            Token::Shift(v) => 3 + u32::from(v),
            Token::Pitch(v) => 1004 + u32::from(v),
            Token::Velocity(on) => 1132 + u32::from(on),
            Token::Tie => 1134,
            Token::Program(v) => 1135 + u32::from(v),
            Token::Drum(v) => 1265 + u32::from(v),
        }
    }

    /// Inverse of [`Token::id`]. Returns `None` for `zero_idx = -1` and
    /// any other out-of-range ID.
    #[must_use]
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Token::Pad),
            1 => Some(Token::Eos),
            2 => Some(Token::Unk),
            3..=1003 => Some(Token::Shift((id - 3) as u16)),
            1004..=1131 => Some(Token::Pitch((id - 1004) as u8)),
            1132 => Some(Token::Velocity(false)),
            1133 => Some(Token::Velocity(true)),
            1134 => Some(Token::Tie),
            1135..=1264 => Some(Token::Program((id - 1135) as u8)),
            1265..=1392 => Some(Token::Drum((id - 1265) as u8)),
            _ => None,
        }
    }

    /// Frame count for time-advance tokens. All non-`Shift` tokens hold
    /// the current time at 0; `Shift(0)` holds the current time at 0,
    /// `Shift(1)` advances one frame (10 ms at 100 Hz).
    #[must_use]
    pub fn shift_frames(self) -> u16 {
        match self {
            Token::Shift(v) => v,
            _ => 0,
        }
    }
}

/// Cheap decode iterator over a slice of token IDs. Hands each token
/// out alongside its implied absolute time (seconds). Holds a reference
/// to the tokenizer's vocab so it doesn't need to allocate.
pub struct TokenIter<'a> {
    ids: &'a [u32],
    vocab: &'a [Token],
    frame_rate_hz: u32,
    /// Running absolute time in frames; reset on `Shift(0)` to "current
    /// time" convention.
    time_frames: i32,
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = (Token, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let (&id, rest) = self.ids.split_first()?;
        self.ids = rest;
        let tok = Token::from_id(id).unwrap_or(Token::Unk);
        // `Shift` advances *to* time `v` (so `Shift(0)` is a no-op time
        // anchor); match the upstream's `_tick_state = _start_tick + value`
        // semantics — only a `Shift` token updates the clock. Every other
        // token type holds the current time (matching `OpenNoteTracker.feed`,
        // which only ever touches `_tick_state` in the `shift` branch); a
        // previous version of this method reset `time_frames` to 0 on every
        // non-`Shift` token, so every event immediately following a `Shift`
        // reported the right time but everything after that collapsed back
        // to time 0 — the reconstructed notes all appeared to happen at once.
        if let Token::Shift(v) = tok {
            self.time_frames = i32::from(v);
        }
        let secs = f64::from(self.time_frames.max(0)) / f64::from(self.frame_rate_hz);
        Some((tok, secs))
    }
}

/// Tokenizer state: the (immutable) decode table, a `(type, value) → ID`
/// lookup for utilities like the tie-section prelude builder, and the
/// `frame_rate` the `Shift` values are denominated in.
pub struct MT3Tokenizer {
    vocab: Vec<Token>,
    frame_rate: u32,
    by_type_value: HashMap<(u8, u32), u32>,
}

impl Default for MT3Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MT3Tokenizer {
    /// Build a tokenizer matching the upstream defaults (MT3_FULL_PLUS
    /// vocabulary, 1001 shift steps, 100 Hz frame rate).
    #[must_use]
    pub fn new() -> Self {
        let mut vocab = Vec::with_capacity(CARD);
        vocab.push(Token::Pad);
        vocab.push(Token::Eos);
        vocab.push(Token::Unk);
        for v in 0..MAX_SHIFT_STEPS as u16 {
            vocab.push(Token::Shift(v));
        }
        for v in 0..128u8 {
            vocab.push(Token::Pitch(v));
        }
        vocab.push(Token::Velocity(false));
        vocab.push(Token::Velocity(true));
        vocab.push(Token::Tie);
        for v in 0..130u8 {
            vocab.push(Token::Program(v));
        }
        for v in 0..128u8 {
            vocab.push(Token::Drum(v));
        }
        debug_assert_eq!(vocab.len(), CARD);

        let mut by_type_value = HashMap::with_capacity(CARD);
        for (i, t) in vocab.iter().enumerate() {
            let (ty, val) = match t {
                Token::Pad => (0, 0),
                Token::Eos => (1, 0),
                Token::Unk => (2, 0),
                Token::Shift(v) => (3, u32::from(*v)),
                Token::Pitch(v) => (4, u32::from(*v)),
                Token::Velocity(false) => (5, 0),
                Token::Velocity(true) => (5, 1),
                Token::Tie => (6, 0),
                Token::Program(v) => (7, u32::from(*v)),
                Token::Drum(v) => (8, u32::from(*v)),
            };
            by_type_value.insert((ty, val), i as u32);
        }

        Self {
            vocab,
            frame_rate: 100,
            by_type_value,
        }
    }

    /// Decode a single token ID.
    #[must_use]
    pub fn decode(&self, id: u32) -> Option<Token> {
        Token::from_id(id)
    }

    /// Iterate decoded `(token, time_seconds)` pairs over a slice of IDs.
    #[must_use]
    pub fn iter<'a>(&'a self, ids: &'a [u32]) -> TokenIter<'a> {
        TokenIter {
            ids,
            vocab: &self.vocab,
            frame_rate_hz: self.frame_rate,
            time_frames: 0,
        }
    }

    /// Look up the token ID for a `(type, value)` pair. Used by the
    /// tie-section prologue builder (`tie_section_token_ids`) and any
    /// other utility that needs to produce structured event-token
    /// sequences without going through a sampled generation.
    ///
    /// `token_type` follows the encoder's own convention:
    /// `1 = program, 2 = pitch, 4 = tie, 5 = velocity, 6 = drum`.
    /// These IDs are *encoder* tags, not the `Token` enum's internal
    /// ordering — keep them compatible with the upstream
    /// `build_event_vocab` index table.
    #[must_use]
    pub fn encode_type_value(&self, token_type: u8, value: u32) -> Option<u32> {
        // The encoder tags line up with the upstream convention:
        // 0=PAD, 1=EOS, 2=UNK are reserved; below uses
        // the public tag values used by the audio-craft event encoder.
        let ty = match token_type {
            0 => 0, // PAD
            1 => 1, // EOS
            2 => 2, // UNK
            3 => 3, // shift
            4 => 4, // pitch
            5 => 5, // velocity
            6 => 6, // tie
            7 => 7, // program
            8 => 8, // drum
            _ => return None,
        };
        self.by_type_value.get(&(ty, value)).copied()
    }

    /// Total tokens in the vocabulary.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Frame rate the tokenizer's `Shift` values are denominated in.
    #[must_use]
    pub const fn frame_rate_hz(&self) -> u32 {
        self.frame_rate
    }

    /// Compute the set of `program` / `drum` token IDs that must be
    /// masked so only the named [`MT3_FULL_PLUS_GROUP_NAMES`] groups can
    /// appear in the output. Mirrors the upstream
    /// `MT3Tokenizer.forbidden_token_ids`.
    #[must_use]
    pub fn forbidden_token_ids(&self, instruments: &[&str]) -> Vec<u32> {
        let mut allowed_programs: std::collections::HashSet<u8> = std::collections::HashSet::new();
        let mut allow_drums = false;

        for name in instruments {
            if *name == "drums" {
                allow_drums = true;
                continue;
            }
            let Some(gid) = group_id_by_name(name) else {
                continue;
            };
            if let Some(first_prog) = first_program_of_group(gid) {
                allowed_programs.insert(first_prog);
            }
        }

        let mut out = Vec::new();
        for t in &self.vocab {
            match t {
                Token::Program(p) if !allowed_programs.contains(p) => out.push(t.id()),
                Token::Drum(_) if !allow_drums => out.push(t.id()),
                _ => {},
            }
        }
        out
    }
}

/// Encode a tie prologue declaring the given open `(program, pitch)`
/// pairs as sustained at a chunk boundary. Matches the upstream
/// `MT3Tokenizer.tie_section_token_ids`: pairs sorted by `(program,
/// pitch)`, each program token emitted once for its run of pitches,
/// terminated by a single `tie` token. Used to teacher-force the
/// beginning-of-chunk tie section so the model can't restate the previous
/// chunk's still-sounding notes with the wrong instruments.
///
/// Callers that have lost the actual open-note set should omit the
/// prologue; doing so forces the model to guess, but the guess is at
/// least token-budget-bounded by the chunk's `max_gen_len`.
#[must_use]
pub fn tie_section_token_ids(tokenizer: &MT3Tokenizer, open_notes: &[(u8, u8)]) -> Vec<u32> {
    let mut sorted = open_notes.to_vec();
    sorted.sort_unstable();

    let mut out = Vec::with_capacity(open_notes.len() * 2 + 2);
    let mut current_program: Option<u8> = None;
    for &(program, pitch) in &sorted {
        if Some(program) != current_program {
            if let Some(id) = tokenizer.encode_type_value(7, u32::from(program)) {
                out.push(id);
            }
            current_program = Some(program);
        }
        if let Some(id) = tokenizer.encode_type_value(4, u32::from(pitch)) {
            out.push(id);
        }
    }
    if let Some(tie_id) = tokenizer.encode_type_value(6, 0) {
        out.push(tie_id);
    }
    out
}

// ── Free-function helpers ──────────────────────────────────────────────

/// Convert exact instrument group names to the space-joined ID string
/// the model expects as conditioning input.
///
/// # Errors
///
/// Returns `Err` listing any unknown names.
pub fn instrument_group_from_names(names: &[&str]) -> Result<String, String> {
    let valid: HashMap<&str, u8> = MT3_FULL_PLUS_GROUP_NAMES.iter().copied().collect();
    let mut ids = Vec::with_capacity(names.len());
    let mut unknown = Vec::new();
    for n in names {
        match valid.get(n) {
            Some(&id) => ids.push(id),
            None => unknown.push(*n),
        }
    }
    if !unknown.is_empty() {
        let valid_list = MT3_FULL_PLUS_GROUP_NAMES
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "unknown instrument name(s): {}; valid names: {}",
            unknown
                .iter()
                .map(|n| format!("{n:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            valid_list
        ));
    }
    Ok(ids.iter().map(u8::to_string).collect::<Vec<_>>().join(" "))
}

/// Resolve loosely-typed instrument tokens to canonical group names
/// (case-insensitive, single-substring matches that uniquely identify a
/// group). Mirrors `resolve_instrument_names` from the upstream
/// `muscriptor.tokenizer.mt3` module.
///
/// # Errors
///
/// Returns `Err` listing bad tokens (with "did you mean …" hints).
pub fn resolve_instrument_names(tokens: &[&str]) -> Result<Vec<String>, String> {
    let valid_names: Vec<&str> = MT3_FULL_PLUS_GROUP_NAMES.iter().map(|(n, _)| *n).collect();

    fn closeness(needle: &str, name: &str) -> f64 {
        // Cheap char-overlap ratio over `name` and each underscore-
        // separated word of `name`. Not jaro-winkler — strong enough for
        // typo detection against 36 names, no extra crate dep.
        let score_pair = |a: &str, b: &str| -> f64 {
            let mut count = 0usize;
            let mut i = 0usize;
            for ch in a.chars() {
                if let Some(rest) = b[i..].find(ch) {
                    count += 1;
                    i += rest + ch.len_utf8();
                }
            }
            let denom = a.chars().count().max(b.chars().count()) as f64;
            if denom == 0.0 {
                0.0
            } else {
                count as f64 / denom
            }
        };
        std::iter::once(name)
            .chain(name.split('_'))
            .map(|part| score_pair(needle, part))
            .fold(0.0_f64, f64::max)
    }

    let mut out = Vec::with_capacity(tokens.len());
    for &tok in tokens {
        let t = tok.trim().to_ascii_lowercase();
        if valid_names.iter().any(|n| *n == t) {
            out.push(t);
            continue;
        }
        let hits: Vec<&&str> = valid_names.iter().filter(|n| n.contains(&t)).collect();
        if hits.len() == 1 {
            out.push((*hits[0]).to_string());
        } else if !hits.is_empty() {
            return Err(format!(
                "ambiguous instrument name {tok:?}: matches {}",
                hits.iter().map(|n| **n).collect::<Vec<_>>().join(", ")
            ));
        } else {
            let mut ranked: Vec<(&str, f64)> =
                valid_names.iter().map(|n| (*n, closeness(&t, n))).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let suggestions: Vec<&str> = ranked
                .iter()
                .take(3)
                .filter(|(_, s)| *s >= 0.6)
                .map(|(n, _)| *n)
                .collect();
            let hint = if suggestions.is_empty() {
                String::new()
            } else {
                format!(" — did you mean {}?", suggestions.join(", "))
            };
            return Err(format!("unknown instrument name {tok:?}{hint}"));
        }
    }
    Ok(out)
}

fn group_id_by_name(name: &str) -> Option<u8> {
    MT3_FULL_PLUS_GROUP_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, id)| *id)
}

/// Map a decoded `program` token back to its human-readable instrument
/// name, or `None` if it isn't the representative program of any
/// `MT3_FULL_PLUS` group (the caller falls back to `program_<n>`).
/// Mirrors the upstream's `_build_instrument_for_program` — the model only
/// ever emits a group's first (representative) program, so this is the
/// exact inverse of [`first_program_of_group`].
#[must_use]
pub fn instrument_name_for_program(program: u8) -> Option<&'static str> {
    if program == DRUM_PROGRAM {
        return Some("drums");
    }
    MT3_FULL_PLUS_GROUP_NAMES
        .iter()
        .find_map(|&(name, gid)| (first_program_of_group(gid) == Some(program)).then_some(name))
}

// Borrowed from the upstream `get_group_program_map("MT3_FULL_PLUS")` —
// only the first (representative) program of each group matters for the
// forbidden-token filter.
fn first_program_of_group(gid: u8) -> Option<u8> {
    let progs: &[u8] = match gid {
        0 => &[0, 1, 3, 6, 7],
        1 => &[2, 4, 5],
        2 => &[8, 9, 10, 11, 12, 13, 14, 15],
        3 => &[16, 17, 18, 19, 20, 21, 22, 23],
        4 => &[24, 25],
        5 => &[26, 27, 28],
        6 => &[29, 30, 31],
        7 => &[32, 35],
        8 => &[33, 34, 36, 37, 38, 39],
        9 => &[40],
        10 => &[41],
        11 => &[42],
        12 => &[43],
        13 => &[46],
        14 => &[47],
        15 => &[48, 49, 44, 45],
        16 => &[50, 51],
        17 => &[52, 53, 54],
        18 => &[55],
        19 => &[56, 59],
        20 => &[57],
        21 => &[58],
        22 => &[60],
        23 => &[61, 62, 63],
        24 => &[64, 65],
        25 => &[66],
        26 => &[67],
        27 => &[68],
        28 => &[69],
        29 => &[70],
        30 => &[71],
        31 => &[72, 73, 74, 75, 76, 77, 78, 79],
        32 => &[80, 81, 82, 83, 84, 85, 86, 87],
        33 => &[88, 89, 90, 91, 92, 93, 94, 95],
        34 => &[100],
        35 => &[101],
        36 => &[DRUM_PROGRAM],
        _ => return None,
    };
    progs.first().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_count_matches_card() {
        let t = MT3Tokenizer::new();
        assert_eq!(t.vocab_size(), CARD);
    }

    #[test]
    fn special_token_layout() {
        assert_eq!(Token::Pad.id(), 0);
        assert_eq!(Token::Eos.id(), 1);
        assert_eq!(Token::Unk.id(), 2);
        assert_eq!(EOS_ID, 1);
    }

    #[test]
    fn shift_range_boundary() {
        assert_eq!(Token::Shift(0).id(), 3);
        assert_eq!(Token::Shift(1000).id(), 1003);
        assert_eq!(Token::from_id(3), Some(Token::Shift(0)));
        assert_eq!(Token::from_id(1003), Some(Token::Shift(1000)));
        assert_eq!(Token::from_id(1004), Some(Token::Pitch(0)));
    }

    #[test]
    fn instrument_name_for_program_round_trips_group_names() {
        assert_eq!(instrument_name_for_program(0), Some("acoustic_piano"));
        assert_eq!(instrument_name_for_program(40), Some("violin"));
        assert_eq!(instrument_name_for_program(DRUM_PROGRAM), Some("drums"));
        // Program 5 isn't any group's representative (acoustic_piano's group
        // is [0,1,3,6,7]; 5 belongs to electric_piano's group but 2 is its
        // representative) — falls through to the `program_<n>` fallback.
        assert_eq!(instrument_name_for_program(5), None);
    }

    #[test]
    fn forbidden_masks_other_programs() {
        let t = MT3Tokenizer::new();
        let allow = vec!["acoustic_piano"];
        let forbid = t.forbidden_token_ids(&allow);
        let prog_zero = Token::Program(0).id();
        assert!(!forbid.contains(&prog_zero), "program 0 should be allowed");
        assert!(forbid.contains(&Token::Program(1).id()));
        assert!(forbid.contains(&Token::Drum(60).id()));
    }

    #[test]
    fn instrument_group_round_trip() {
        let s = instrument_group_from_names(&["acoustic_piano", "drums"]).unwrap();
        assert_eq!(s, "0 36");
    }

    #[test]
    fn tie_section_pairs_get_encoded() {
        let t = MT3Tokenizer::new();
        let ids = tie_section_token_ids(&t, &[(0, 60), (0, 64), (40, 67)]);
        // Expected layout: program 0, pitch 60, pitch 64, program 40,
        // pitch 67, tie.
        let decoded: Vec<Token> = ids.iter().map(|&i| Token::from_id(i).unwrap()).collect();
        assert_eq!(
            decoded,
            vec![
                Token::Program(0),
                Token::Pitch(60),
                Token::Pitch(64),
                Token::Program(40),
                Token::Pitch(67),
                Token::Tie,
            ]
        );
    }

    #[test]
    fn iterator_advances_on_shift() {
        let t = MT3Tokenizer::new();
        // shift 0 → 0s; shift 50 → 0.5s; pitch 60 → 0.5s (carried forward).
        let ids = vec![
            Token::Shift(0).id(),
            Token::Pitch(60).id(),
            Token::Shift(50).id(),
            Token::Pitch(64).id(),
        ];
        let pairs: Vec<(Token, f64)> = t.iter(&ids).collect();
        assert_eq!(pairs[0].0, Token::Shift(0));
        assert!((pairs[0].1 - 0.0).abs() < 1e-9);
        assert!((pairs[2].1 - 0.5).abs() < 1e-9);
    }
}
