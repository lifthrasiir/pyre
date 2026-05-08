//! Bitstring helpers, line-by-line port of `rpython/tool/algo/bitstring.py`.
//!
//! `EffectInfo.bitstring_*` fields use this representation: a sequence of
//! bytes where bit `n` is set iff the descr with `ei_index == n` is in the
//! corresponding readonly/write set. `make_bitstring` constructs the
//! sequence from a list of indices; `bitcheck` queries one bit; `num_bits`
//! returns the total bit width.

/// `bitstring.py:3-13` `make_bitstring(lst)`.
///
/// Returns an empty byte vector for an empty index list (matching
/// upstream's `''` empty-string return). Otherwise allocates
/// `(max(lst) + 1 + 7) // 8` bytes and OR-sets one bit per index.
pub fn make_bitstring(lst: &[u32]) -> Vec<u8> {
    if lst.is_empty() {
        return Vec::new();
    }
    let num_bits = (*lst.iter().max().unwrap() as usize) + 1;
    let num_bytes = (num_bits + 7) / 8;
    let mut entries = vec![0u8; num_bytes];
    for &x in lst {
        entries[(x >> 3) as usize] |= 1u8 << (x & 7);
    }
    entries
}

/// `bitstring.py:15-20` `bitcheck(bitstring, n)`.
///
/// Returns `false` when `n`'s byte index is outside the bitstring's
/// length, mirroring upstream's `byte_number >= len(bitstring)` short
/// return. RPython's `assert n >= 0` is encoded by the unsigned `u32`
/// argument type.
pub fn bitcheck(bitstring: &[u8], n: u32) -> bool {
    let byte_number = (n >> 3) as usize;
    if byte_number >= bitstring.len() {
        return false;
    }
    (bitstring[byte_number] & (1u8 << (n & 7))) != 0
}

/// `bitstring.py:22-23` `num_bits(bitstring)`.
pub fn num_bits(bitstring: &[u8]) -> usize {
    bitstring.len() << 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_bitstring_empty() {
        assert_eq!(make_bitstring(&[]), Vec::<u8>::new());
    }

    #[test]
    fn make_bitstring_single_byte() {
        // bits 0, 3, 7 → byte 0b10001001 = 0x89.
        assert_eq!(make_bitstring(&[0, 3, 7]), vec![0x89]);
    }

    #[test]
    fn make_bitstring_multi_byte() {
        // bits 0, 8, 15 → bytes [0x01, 0x81].
        assert_eq!(make_bitstring(&[0, 8, 15]), vec![0x01, 0x81]);
    }

    #[test]
    fn bitcheck_within_range() {
        let bs = make_bitstring(&[0, 3, 7, 8, 15]);
        for n in [0u32, 3, 7, 8, 15] {
            assert!(bitcheck(&bs, n));
        }
        for n in [1u32, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14] {
            assert!(!bitcheck(&bs, n));
        }
    }

    #[test]
    fn bitcheck_oob_returns_false() {
        let bs = make_bitstring(&[0, 3]);
        assert!(!bitcheck(&bs, 100));
        assert!(!bitcheck(&[], 0));
    }

    #[test]
    fn num_bits_counts_bytes_times_eight() {
        assert_eq!(num_bits(&[]), 0);
        assert_eq!(num_bits(&[0xff]), 8);
        assert_eq!(num_bits(&[0, 0, 0]), 24);
    }
}
