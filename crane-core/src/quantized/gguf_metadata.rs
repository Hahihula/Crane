// SPDX-License-Identifier: MIT

//! Typed access to GGUF metadata, shared by every model's `from_gguf`.
//!
//! Two flavours of getter: the required ones (`string`, `u32`, `usize`, `f32`,
//! `u64s`, ...) fail with a "missing GGUF metadata <key>" error, and the
//! `opt_*` ones return `None` when the key is absent *or* has an unexpected
//! type, for keys a loader has a default for.

use std::collections::HashMap;

use candle_core::Result;
use candle_core::quantized::gguf_file::Value;

/// Typed view of a GGUF file's key/value metadata
/// ([`Gguf::metadata`](super::gguf_file::Gguf::metadata) or
/// `Content::metadata`).
pub struct GgufMetadata<'a>(pub &'a HashMap<String, Value>);

impl<'a> GgufMetadata<'a> {
    #[must_use]
    pub fn new(metadata: &'a HashMap<String, Value>) -> Self {
        Self(metadata)
    }

    /// The raw value, or an error naming the missing key.
    ///
    /// # Errors
    ///
    /// Returns an error if `key` is absent.
    pub fn get(&self, key: &str) -> Result<&'a Value> {
        self.0
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF metadata {key}")))
    }

    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// # Errors
    ///
    /// Returns an error if `key` is absent or not a string.
    pub fn string(&self, key: &str) -> Result<String> {
        self.get(key)?.to_string().cloned()
    }

    /// # Errors
    ///
    /// Returns an error if `key` is absent or not a float.
    pub fn f32(&self, key: &str) -> Result<f32> {
        self.get(key)?.to_f32()
    }

    /// # Errors
    ///
    /// Returns an error if `key` is absent or not an unsigned integer.
    pub fn u32(&self, key: &str) -> Result<u32> {
        self.get(key)?.to_u32()
    }

    /// # Errors
    ///
    /// Returns an error if `key` is absent or not an unsigned integer.
    pub fn usize(&self, key: &str) -> Result<usize> {
        Ok(self.u32(key)? as usize)
    }

    /// Like [`Self::u32`], but tolerant of what some converters (e.g. unsloth)
    /// write for a scalar: a non-negative `I32`, or a per-layer array whose
    /// first element is taken.
    ///
    /// # Errors
    ///
    /// Returns an error if `key` is absent, an empty array, or not an integer.
    pub fn u32_lenient(&self, key: &str) -> Result<u32> {
        fn lenient(v: &Value) -> Result<u32> {
            match v {
                Value::Array(arr) => match arr.first() {
                    Some(first) => lenient(first),
                    None => candle_core::bail!("empty GGUF array where a u32 was expected"),
                },
                Value::I32(n) if *n >= 0 => Ok(n.cast_unsigned()),
                v => v.to_u32(),
            }
        }
        lenient(self.get(key)?)
    }

    /// An integer array of any width, as `u64`s.
    ///
    /// # Errors
    ///
    /// Returns an error if `key` is absent, not an array, or holds a
    /// non-integer / negative element.
    pub fn u64s(&self, key: &str) -> Result<Vec<u64>> {
        self.get(key)?
            .to_vec()?
            .iter()
            .map(|v| match v {
                Value::U8(x) => Ok(u64::from(*x)),
                Value::U16(x) => Ok(u64::from(*x)),
                Value::U32(x) => Ok(u64::from(*x)),
                Value::U64(x) => Ok(*x),
                Value::I32(x) => u64::try_from(*x).map_err(|e| e.to_string()),
                Value::I64(x) => u64::try_from(*x).map_err(|e| e.to_string()),
                other => Err(format!("{other:?} is not an unsigned integer")),
            })
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| candle_core::Error::Msg(format!("GGUF metadata {key}: {e}")))
    }

    /// [`Self::u64s`] as `usize`s.
    ///
    /// # Errors
    ///
    /// As [`Self::u64s`], plus an element that does not fit `usize`.
    pub fn usizes(&self, key: &str) -> Result<Vec<usize>> {
        self.u64s(key)?
            .into_iter()
            .map(|v| usize::try_from(v).map_err(|e| candle_core::Error::Msg(e.to_string())))
            .collect()
    }

    /// # Errors
    ///
    /// Returns an error if `key` is absent or not an array.
    pub fn array_len(&self, key: &str) -> Result<usize> {
        Ok(self.get(key)?.to_vec()?.len())
    }

    #[must_use]
    pub fn opt_string(&self, key: &str) -> Option<String> {
        self.0.get(key)?.to_string().ok().cloned()
    }

    #[must_use]
    pub fn opt_u32(&self, key: &str) -> Option<u32> {
        self.0.get(key)?.to_u32().ok()
    }

    #[must_use]
    pub fn opt_usize(&self, key: &str) -> Option<usize> {
        self.opt_u32(key).map(|v| v as usize)
    }

    #[must_use]
    pub fn opt_f32(&self, key: &str) -> Option<f32> {
        self.0.get(key)?.to_f32().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HashMap<String, Value> {
        [
            ("a.u32", Value::U32(7)),
            ("a.i32", Value::I32(5)),
            ("a.neg", Value::I32(-1)),
            ("a.f32", Value::F32(0.5)),
            ("a.str", Value::String("x".into())),
            (
                "a.per_layer",
                Value::Array(vec![Value::I32(9), Value::I32(3)]),
            ),
            ("a.wide", Value::Array(vec![Value::U8(1), Value::I64(2)])),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    #[test]
    fn required_getters_report_missing_keys() {
        let map = sample();
        let md = GgufMetadata::new(&map);
        assert_eq!(md.usize("a.u32").unwrap(), 7);
        assert_eq!(md.string("a.str").unwrap(), "x");
        assert!((md.f32("a.f32").unwrap() - 0.5).abs() < f32::EPSILON);
        let err = md.u32("a.missing").unwrap_err().to_string();
        assert!(err.contains("missing GGUF metadata a.missing"), "{err}");
    }

    #[test]
    fn optional_getters_swallow_absence_and_wrong_type() {
        let map = sample();
        let md = GgufMetadata::new(&map);
        assert_eq!(md.opt_u32("a.u32"), Some(7));
        assert_eq!(md.opt_u32("a.missing"), None);
        assert_eq!(md.opt_u32("a.str"), None);
        assert_eq!(md.opt_f32("a.f32"), Some(0.5));
        assert_eq!(md.opt_string("a.str").as_deref(), Some("x"));
    }

    #[test]
    fn lenient_u32_accepts_i32_and_per_layer_arrays() {
        let map = sample();
        let md = GgufMetadata::new(&map);
        assert_eq!(md.u32_lenient("a.i32").unwrap(), 5);
        assert_eq!(md.u32_lenient("a.per_layer").unwrap(), 9);
        assert!(md.u32_lenient("a.neg").is_err());
    }

    #[test]
    fn integer_arrays_of_any_width() {
        let map = sample();
        let md = GgufMetadata::new(&map);
        assert_eq!(md.u64s("a.wide").unwrap(), vec![1, 2]);
        assert_eq!(md.usizes("a.per_layer").unwrap(), vec![9, 3]);
        assert_eq!(md.array_len("a.per_layer").unwrap(), 2);
        assert!(md.u64s("a.u32").is_err());
    }
}
