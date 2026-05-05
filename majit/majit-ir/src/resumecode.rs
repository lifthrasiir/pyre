//! Resume bytecode encoding/decoding.
//!
//! Direct port of rpython/jit/metainterp/resumecode.py.
//!
//! Encoding: variable-length integers with zigzag encoding.
//! - 7-bit:  0xxxxxxx
//! - 14-bit: 1xxxxxxx 0xxxxxxx
//! - 21-bit: 1xxxxxxx 1xxxxxxx xxxxxxxx

/// resumecode.py: append_numbering(lst, item)
pub fn encode_varint(buf: &mut Vec<u8>, value: i32) {
    let mut item = (value as i64) * 2;
    if item < 0 {
        item = -1 - item;
    }
    assert!(item >= 0);
    let item = item as u32;

    if item < (1 << 7) {
        buf.push(item as u8);
    } else if item < (1 << 14) {
        buf.push((item | 0x80) as u8);
        buf.push((item >> 7) as u8);
    } else {
        assert!(item < (1 << 16), "resumecode item too large: {item}");
        buf.push((item | 0x80) as u8);
        buf.push(((item >> 7) | 0x80) as u8);
        buf.push((item >> 14) as u8);
    }
}

/// resumecode.py: numb_next_item(numb, index)
///
/// line-by-line port. Does not bounds-check: upstream contract requires
/// the buffer to contain a complete varint at `index`. A truncated
/// buffer is a bug in resume data generation and should panic loudly
/// via the standard slice indexing rather than silently returning 0.
pub fn decode_varint(buf: &[u8], index: usize) -> (i32, usize) {
    let mut value = buf[index] as i64;
    let mut index = index + 1;

    if value & (1 << 7) != 0 {
        value &= (1 << 7) - 1;
        value |= (buf[index] as i64) << 7;
        index += 1;
        if value & (1 << 14) != 0 {
            value &= (1 << 14) - 1;
            value |= (buf[index] as i64) << 14;
            index += 1;
        }
    }

    if value & 1 != 0 {
        value = -1 - value;
    }
    value >>= 1;

    (value as i32, index)
}

/// resumecode.py:70-73 numb_next_n_items — skip `size` items without
/// returning values. Used by decoders that advance over a subsection.
pub fn numb_next_n_items(buf: &[u8], size: usize, mut index: usize) -> usize {
    for _ in 0..size {
        let (_, new_index) = decode_varint(buf, index);
        index = new_index;
    }
    index
}

/// resumecode.py:110-114 create_numbering(l) — module-level helper:
/// build a single Writer from the list and return its encoded buffer.
pub fn create_numbering(items: &[i32]) -> Vec<u8> {
    let mut w = Writer::new(items.len());
    for &item in items {
        w.append_int(item as i64);
    }
    w.create_numbering()
}

/// resumecode.py: unpack_numbering(numb)
pub fn unpack_all(buf: &[u8]) -> Vec<i32> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let (next, new_i) = decode_varint(buf, i);
        result.push(next);
        i = new_i;
    }
    result
}

/// resumecode.py: Writer
pub struct Writer {
    pub current: Vec<i32>,
}

impl Writer {
    pub fn new(size_hint: usize) -> Self {
        Writer {
            current: Vec::with_capacity(size_hint),
        }
    }

    /// resumecode.py: append_short
    pub fn append_short(&mut self, item: i32) {
        self.current.push(item);
    }

    /// resumecode.py:90-93 append_int — `short = rffi.cast(rffi.SHORT, item);
    /// assert rffi.cast(lltype.Signed, short) == item`. The upstream
    /// signature is "any int" (RPython `Signed` ≈ machine word); accepting
    /// `i64` here lets the range check apply to the *original* caller value
    /// rather than a silently-truncated copy.
    #[track_caller]
    pub fn append_int(&mut self, item: i64) {
        let short = item as i16;
        assert!(
            short as i64 == item,
            "append_int: value {item} out of i16 range"
        );
        self.append_short(short as i32);
    }

    /// resumecode.py: create_numbering
    pub fn create_numbering(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.current.len() * 3);
        for &item in &self.current {
            encode_varint(&mut buf, item);
        }
        buf
    }

    /// resumecode.py: patch_current_size
    pub fn patch_current_size(&mut self, index: usize) {
        self.current[index] = self.current.len() as i32;
    }

    /// resumecode.py: patch
    pub fn patch(&mut self, index: usize, item: i32) {
        self.current[index] = item;
    }
}

/// resumecode.py: Reader
pub struct Reader<'a> {
    code: &'a [u8],
    pub cur_pos: usize,
    pub items_read: usize,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Reader {
            code,
            cur_pos: 0,
            items_read: 0,
        }
    }

    /// resumecode.py: next_item
    pub fn next_item(&mut self) -> i32 {
        let (result, new_pos) = decode_varint(self.code, self.cur_pos);
        self.cur_pos = new_pos;
        self.items_read += 1;
        result
    }

    /// resumecode.py: peek
    pub fn peek(&self) -> i32 {
        let (result, _) = decode_varint(self.code, self.cur_pos);
        result
    }

    /// resumecode.py: jump — skip n items forward
    pub fn jump(&mut self, size: usize) {
        for _ in 0..size {
            let (_, new_pos) = decode_varint(self.code, self.cur_pos);
            self.cur_pos = new_pos;
        }
        self.items_read += size;
    }

    pub fn has_more(&self) -> bool {
        self.cur_pos < self.code.len()
    }
}
