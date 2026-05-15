//! Raw buffer write tracking.
//!
//! RPython parity target:
//! `rpython/jit/metainterp/optimizeopt/rawbuffer.py`.

use majit_ir::{DescrRef, OpRef};

/// rawbuffer.py:13 RawBuffer — 4 parallel lists: offsets, lengths, descrs, values.
/// Sorted by offset. Invariant: offsets[i]+lengths[i] <= offsets[i+1].
#[derive(Clone, Debug)]
pub struct RawBuffer {
    /// rawbuffer.py:14: self.offsets — signed because RPython's
    /// unbounded int allows `basesize + itemsize*index` to be negative
    /// when `index < 0`. `write_value` keeps the list sorted using
    /// signed comparison (rawbuffer.py:104 `self.offsets[i] > offset`).
    offsets: Vec<i64>,
    /// rawbuffer.py:15: self.lengths — always non-negative (unsigned
    /// upstream itemsize from `unpack_arraydescr_size`).
    lengths: Vec<usize>,
    /// rawbuffer.py:16: self.descrs — per-entry ArrayDescr.
    descrs: Vec<DescrRef>,
    /// rawbuffer.py:17: self.values
    values: Vec<OpRef>,
}

/// Error returned when a raw buffer operation violates invariants.
///
/// RPython raises `InvalidRawWrite` / `InvalidRawRead`. The Rust optimizer
/// only needs the category to give up the optimization, but tests inspect the
/// details to keep the port honest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawBufferError {
    /// A write overlaps with an existing write, or a same-offset write has
    /// incompatible length/descr. Both are `InvalidRawWrite` upstream.
    OverlappingWrite {
        new_offset: i64,
        new_length: usize,
        existing_offset: i64,
        existing_length: usize,
    },
    /// A read from an offset that was never written.
    UninitializedRead { offset: i64, length: usize },
    /// A read whose length/offset doesn't match the write at that offset.
    IncompatibleRead {
        offset: i64,
        read_length: usize,
        write_length: usize,
    },
}

impl RawBuffer {
    /// Construct the empty rawbuffer.py parallel-list state.
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
            lengths: Vec::new(),
            descrs: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn offsets(&self) -> &[i64] {
        &self.offsets
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    pub fn descrs(&self) -> &[DescrRef] {
        &self.descrs
    }

    pub fn values(&self) -> &[OpRef] {
        &self.values
    }

    pub fn iter_entries(&self) -> impl Iterator<Item = (i64, usize, &DescrRef, OpRef)> + '_ {
        self.offsets
            .iter()
            .copied()
            .zip(self.lengths.iter().copied())
            .zip(self.descrs.iter())
            .zip(self.values.iter().copied())
            .map(|(((offset, length), descr), value)| (offset, length, descr, value))
    }

    pub(crate) fn drain_entries(&mut self) -> Vec<(i64, usize, DescrRef, OpRef)> {
        let offsets = std::mem::take(&mut self.offsets);
        let lengths = std::mem::take(&mut self.lengths);
        let descrs = std::mem::take(&mut self.descrs);
        let values = std::mem::take(&mut self.values);
        debug_assert_eq!(offsets.len(), lengths.len());
        debug_assert_eq!(offsets.len(), descrs.len());
        debug_assert_eq!(offsets.len(), values.len());
        offsets
            .into_iter()
            .zip(lengths)
            .zip(descrs)
            .zip(values)
            .map(|(((offset, length), descr), value)| (offset, length, descr, value))
            .collect()
    }

    /// rawbuffer.py:83: _descrs_are_compatible(d1, d2)
    /// Two arraydescrs are compatible if they have the same basesize,
    /// itemsize and sign.
    fn descrs_are_compatible(d1: &DescrRef, d2: &DescrRef) -> bool {
        let (Some(a1), Some(a2)) = (d1.as_array_descr(), d2.as_array_descr()) else {
            return false;
        };
        a1.base_size() == a2.base_size()
            && a1.item_size() == a2.item_size()
            && a1.is_item_signed() == a2.is_item_signed()
    }

    /// rawbuffer.py:89: write_value(offset, length, descr, value).
    ///
    /// Maintains sorted order by offset. Same-offset update only
    /// replaces value (rawbuffer.py:102), never descr.
    pub fn write_value(
        &mut self,
        offset: i64,
        length: usize,
        descr: DescrRef,
        value: OpRef,
    ) -> Result<(), RawBufferError> {
        // RPython rawbuffer.py uses unbounded-int `length`. The pyre
        // length is `usize`; on 64-bit platforms `usize > i64::MAX`
        // would wrap to a negative i64 and break the signed overlap
        // checks. Realistically itemsize never approaches 2^63, but
        // bail conservatively for spec-strict parity. Treat the
        // failure as `InvalidRawWrite` (OverlappingWrite is the
        // closest existing error variant).
        let Ok(length_i) = i64::try_from(length) else {
            return Err(RawBufferError::OverlappingWrite {
                new_offset: offset,
                new_length: length,
                existing_offset: offset,
                existing_length: length,
            });
        };
        let mut insert_pos = 0;
        for i in 0..self.offsets.len() {
            let wo = self.offsets[i];
            let wl = self.lengths[i];
            if wo == offset {
                // rawbuffer.py:94-95: length and descr must be compatible.
                if wl != length || !Self::descrs_are_compatible(&descr, &self.descrs[i]) {
                    return Err(RawBufferError::OverlappingWrite {
                        new_offset: offset,
                        new_length: length,
                        existing_offset: wo,
                        existing_length: wl,
                    });
                }
                // rawbuffer.py:102: only replace value, keep existing descr.
                self.values[i] = value;
                return Ok(());
            } else if wo > offset {
                break;
            }
            insert_pos = i + 1;
        }
        // rawbuffer.py:108: `if i < len(self.offsets) and offset+length
        // > self.offsets[i]: _invalid_write("overlap with next bytes")`.
        // RPython int is unbounded; in Rust an i64 overflow on
        // `offset + length` (length is non-negative usize) means the
        // write extends past i64::MAX, which by signed compare is
        // greater than every legitimate next_off — i.e. an overlap.
        // checked_add returns None on overflow → treat as overlap.
        if insert_pos < self.offsets.len() {
            let next_off = self.offsets[insert_pos];
            let end = offset.checked_add(length_i);
            if end.map_or(true, |e| e > next_off) {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: next_off,
                    existing_length: self.lengths[insert_pos],
                });
            }
        }
        // rawbuffer.py:111: `if i > 0 and self.offsets[i-1]+self.lengths[i-1]
        // > offset: _invalid_write("overlap with previous bytes")`.
        // Same overflow argument: a saturated/overflowed `prev_off+prev_len`
        // is unbounded-greater-than `offset` in RPython, so checked_add
        // None → treat as overlap.
        if insert_pos > 0 {
            let prev_off = self.offsets[insert_pos - 1];
            let prev_len = self.lengths[insert_pos - 1];
            let prev_len_i = i64::try_from(prev_len).ok();
            let prev_end = prev_len_i.and_then(|l| prev_off.checked_add(l));
            if prev_end.map_or(true, |e| e > offset) {
                return Err(RawBufferError::OverlappingWrite {
                    new_offset: offset,
                    new_length: length,
                    existing_offset: prev_off,
                    existing_length: prev_len,
                });
            }
        }
        // rawbuffer.py:115-118: insert new entry.
        self.offsets.insert(insert_pos, offset);
        self.lengths.insert(insert_pos, length);
        self.descrs.insert(insert_pos, descr);
        self.values.insert(insert_pos, value);
        Ok(())
    }

    /// rawbuffer.py:120: read_value(offset, length, descr).
    pub fn read_value(
        &self,
        offset: i64,
        length: usize,
        descr: &DescrRef,
    ) -> Result<OpRef, RawBufferError> {
        for i in 0..self.offsets.len() {
            if self.offsets[i] == offset {
                if self.lengths[i] != length || !Self::descrs_are_compatible(descr, &self.descrs[i])
                {
                    return Err(RawBufferError::IncompatibleRead {
                        offset,
                        read_length: length,
                        write_length: self.lengths[i],
                    });
                }
                return Ok(self.values[i]);
            }
        }
        Err(RawBufferError::UninitializedRead { offset, length })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an int ArrayDescr for tests (base_size=0, item_size=8, Int).
    fn int_descr() -> DescrRef {
        majit_ir::descr::make_array_descr(0, 8, majit_ir::Type::Int)
    }

    /// Create an int ArrayDescr with specified item_size for tests.
    fn int_descr_sz(item_size: usize) -> DescrRef {
        majit_ir::descr::make_array_descr(0, item_size, majit_ir::Type::Int)
    }

    fn make_buf(_size: usize) -> RawBuffer {
        RawBuffer::new()
    }

    #[test]
    fn rawbuffer_write_and_read() {
        let mut buf = make_buf(32);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d.clone(), OpRef::int_op(10)).unwrap();
        buf.write_value(8, 4, d4.clone(), OpRef::int_op(20))
            .unwrap();
        buf.write_value(16, 8, d.clone(), OpRef::int_op(30))
            .unwrap();

        assert_eq!(buf.read_value(0, 8, &d).unwrap(), OpRef::int_op(10));
        assert_eq!(buf.read_value(8, 4, &d4).unwrap(), OpRef::int_op(20));
        assert_eq!(buf.read_value(16, 8, &d).unwrap(), OpRef::int_op(30));
    }

    #[test]
    fn rawbuffer_update_same_offset() {
        let mut buf = make_buf(16);
        let d = int_descr();
        buf.write_value(0, 8, d.clone(), OpRef::int_op(10)).unwrap();
        buf.write_value(0, 8, d.clone(), OpRef::int_op(99)).unwrap();

        assert_eq!(buf.read_value(0, 8, &d).unwrap(), OpRef::int_op(99));
        assert_eq!(buf.offsets().len(), 1);
    }

    #[test]
    fn rawbuffer_overlap_next() {
        let mut buf = make_buf(32);
        let d = int_descr();
        buf.write_value(8, 8, d.clone(), OpRef::int_op(10)).unwrap();
        // Write at offset 4 with length 8 overlaps [8, 16)
        let err = buf
            .write_value(4, 8, d.clone(), OpRef::int_op(20))
            .unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_overlap_prev() {
        let mut buf = make_buf(32);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d.clone(), OpRef::int_op(10)).unwrap();
        // Write at offset 4 overlaps with [0, 8)
        let err = buf.write_value(4, 4, d4, OpRef::int_op(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_incompatible_length_at_same_offset() {
        let mut buf = make_buf(16);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d, OpRef::int_op(10)).unwrap();
        let err = buf.write_value(0, 4, d4, OpRef::int_op(20)).unwrap_err();
        assert!(matches!(err, RawBufferError::OverlappingWrite { .. }));
    }

    #[test]
    fn rawbuffer_uninitialized_read() {
        let buf = make_buf(16);
        let d = int_descr();
        let err = buf.read_value(0, 8, &d).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::UninitializedRead {
                offset: 0,
                length: 8
            }
        );
    }

    #[test]
    fn rawbuffer_incompatible_read_length() {
        let mut buf = make_buf(16);
        let d = int_descr();
        let d4 = int_descr_sz(4);
        buf.write_value(0, 8, d, OpRef::int_op(10)).unwrap();
        let err = buf.read_value(0, 4, &d4).unwrap_err();
        assert_eq!(
            err,
            RawBufferError::IncompatibleRead {
                offset: 0,
                read_length: 4,
                write_length: 8,
            }
        );
    }

    #[test]
    fn rawbuffer_sorted_insertion() {
        let mut buf = make_buf(32);
        let d4 = int_descr_sz(4);
        buf.write_value(16, 4, d4.clone(), OpRef::int_op(30))
            .unwrap();
        buf.write_value(0, 4, d4.clone(), OpRef::int_op(10))
            .unwrap();
        buf.write_value(8, 4, d4.clone(), OpRef::int_op(20))
            .unwrap();

        // Entries should be sorted by offset
        assert_eq!(buf.offsets()[0], 0);
        assert_eq!(buf.offsets()[1], 8);
        assert_eq!(buf.offsets()[2], 16);
    }

    /// test_rawbuffer.py:66 test_unpack_descrs
    /// Two different descr objects with same (basesize, itemsize, signed)
    /// must be compatible. Different signed-ness must be incompatible.
    #[test]
    fn test_unpack_descrs() {
        use majit_ir::descr::SimpleArrayDescr;

        // ArrayS_8_1 and ArrayS_8_2: same (base=0, item=8, signed=true)
        let array_s_8_1: DescrRef = std::sync::Arc::new(SimpleArrayDescr::with_flag(
            0,
            0,
            8,
            0,
            majit_ir::Type::Int,
            majit_ir::descr::ArrayFlag::Signed,
        ));
        let array_s_8_2: DescrRef = std::sync::Arc::new(SimpleArrayDescr::with_flag(
            1,
            0,
            8,
            0,
            majit_ir::Type::Int,
            majit_ir::descr::ArrayFlag::Signed,
        ));
        // ArrayU_8: same size but unsigned
        let array_u_8: DescrRef = std::sync::Arc::new(SimpleArrayDescr::with_flag(
            2,
            0,
            8,
            0,
            majit_ir::Type::Int,
            majit_ir::descr::ArrayFlag::Unsigned,
        ));

        assert!(!std::sync::Arc::ptr_eq(&array_s_8_1, &array_s_8_2));

        let mut buf = make_buf(16);

        // Write with ArrayS_8_1
        buf.write_value(0, 4, array_s_8_1.clone(), OpRef::int_op(10))
            .unwrap();

        // Read with same descr
        assert_eq!(
            buf.read_value(0, 4, &array_s_8_1).unwrap(),
            OpRef::int_op(10)
        );
        // Read with non-identical but compatible descr
        assert_eq!(
            buf.read_value(0, 4, &array_s_8_2).unwrap(),
            OpRef::int_op(10)
        );

        // Overwrite with non-identical compatible descr
        buf.write_value(0, 4, array_s_8_2.clone(), OpRef::int_op(20))
            .unwrap();
        assert_eq!(
            buf.read_value(0, 4, &array_s_8_1).unwrap(),
            OpRef::int_op(20)
        );

        // Incompatible descr (unsigned) must fail
        assert!(buf.read_value(0, 4, &array_u_8).is_err());
        assert!(buf.write_value(0, 4, array_u_8, OpRef::int_op(30)).is_err());
    }
}
