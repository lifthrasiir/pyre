/// counter.py: JitCounter — float-based 5-way associative timetable.
///
/// Direct port of rpython/jit/metainterp/counter.py.
/// Uses f32 time values (0.0 to 1.0) instead of integer counts.
/// tick(hash, increment) adds increment; fires when >= 1.0.
/// 5-way associative cache indexed by _get_index(hash), matched by
/// _get_subhash(hash). MRU promotion via _swap.

/// counter.py:82 DEFAULT_SIZE = 2048
pub const DEFAULT_SIZE: usize = 2048;

/// counter.py:12-13 ENTRY: 5 (f32 time, u16 subhash) pairs per bucket.
const ASSOCIATIVITY: usize = 5;

/// counter.py:8 UINT32MAX = 2 ** 32 - 1
const UINT32MAX: u64 = 0xFFFF_FFFF;

/// One timetable entry: 5-way associative (time, subhash) pairs.
/// counter.py:11-13 ENTRY struct.
#[derive(Clone)]
struct Entry {
    /// counter.py: times — f32 timing values, 0.0 to 1.0.
    times: [f32; ASSOCIATIVITY],
    /// counter.py: subhashes — lower 16 bits of the hash.
    subhashes: [u16; ASSOCIATIVITY],
}

impl Default for Entry {
    fn default() -> Self {
        Entry {
            times: [0.0; ASSOCIATIVITY],
            subhashes: [0; ASSOCIATIVITY],
        }
    }
}

/// counter.py:16 JitCounter
pub struct JitCounter {
    /// counter.py:86 size
    size: usize,
    /// counter.py:87 shift
    shift: u32,
    /// counter.py:97 timetable
    timetable: Vec<Entry>,
    /// counter.py:100 _nexthash
    _nexthash: u64,
    /// counter.py:264 decay_by_mult — f64 (Python float).
    decay_by_mult: f64,
}

impl JitCounter {
    /// counter.py:84-100 __init__(self, size=DEFAULT_SIZE, translator=None)
    pub fn new(size: usize) -> Self {
        let mut shift = 16u32;
        while (UINT32MAX >> shift) != (size as u64 - 1) {
            shift += 1;
            assert!(shift < 999, "size is not a power of two <= 2**16");
        }
        JitCounter {
            size,
            shift,
            timetable: vec![Entry::default(); size],
            _nexthash: 0,
            decay_by_mult: 1.0,
        }
    }

    /// counter.py:122-126 compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        if threshold == 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py:128-136 _get_index
    #[inline(always)]
    fn _get_index(&self, hash: u64) -> usize {
        let hash32 = hash as u32 as u64;
        (hash32 >> self.shift) as usize
    }

    /// counter.py:138-140 _get_subhash
    #[inline(always)]
    fn _get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py:142-153 fetch_next_hash
    pub fn fetch_next_hash(&mut self) -> u64 {
        let result = self._nexthash;
        self._nexthash =
            result.wrapping_add(1 | (1u64 << self.shift) | (1u64 << (self.shift - 16)));
        result
    }

    /// counter.py:155-166 _swap
    #[inline(always)]
    fn _swap(entry: &mut Entry, n: usize) -> usize {
        if entry.times[n] > entry.times[n + 1] {
            n + 1
        } else {
            entry.times.swap(n, n + 1);
            entry.subhashes.swap(n, n + 1);
            n
        }
    }

    /// counter.py:168-183 _tick_slowpath
    fn _tick_slowpath(entry: &mut Entry, subhash: u16) -> usize {
        if entry.subhashes[1] == subhash {
            Self::_swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            Self::_swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            Self::_swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            Self::_swap(entry, 3)
        } else {
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        }
    }

    /// PRE-EXISTING-ADAPTATION: no RPython counterpart. Read-only peek
    /// used by warmstate's cold fast path to avoid GreenKey allocation.
    pub fn would_tick_fire(&self, hash: u64, increment: f64) -> bool {
        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                return entry.times[i] as f64 + increment >= 1.0;
            }
        }
        increment >= 1.0
    }

    /// counter.py:185-202 tick(self, hash, increment)
    #[inline(always)]
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];

        let n = if entry.subhashes[0] == subhash {
            0
        } else {
            Self::_tick_slowpath(entry, subhash)
        };

        // counter.py:194: counter = float(p_entry.times[n]) + increment
        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            // counter.py:196: p_entry.times[n] = r_singlefloat(counter)
            entry.times[n] = counter as f32;
            false
        } else {
            // counter.py:199-200: self.reset(hash); return True
            self.reset(hash);
            true
        }
    }

    /// counter.py:204-230 change_current_fraction(hash, new_fraction)
    pub fn change_current_fraction(&mut self, hash: u64, new_fraction: f64) {
        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];

        let mut n = 0;
        while n < 4 && entry.subhashes[n] != subhash && entry.times[n] != 0.0 {
            n += 1;
        }
        while n > 0 {
            n -= 1;
            entry.subhashes[n + 1] = entry.subhashes[n];
            entry.times[n + 1] = entry.times[n];
        }
        entry.subhashes[0] = subhash;
        entry.times[0] = new_fraction as f32;
    }

    /// counter.py:232-237 reset(hash)
    pub fn reset(&mut self, hash: u64) {
        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                entry.times[i] = 0.0;
            }
        }
    }

    /// PRE-EXISTING-ADAPTATION: no RPython equivalent.
    /// Zero all timetable entries.
    pub fn reset_all(&mut self) {
        for entry in &mut self.timetable {
            *entry = Entry::default();
        }
    }

    /// counter.py:258-264 set_decay(decay)
    pub fn set_decay(&mut self, decay: i32) {
        let clamped = decay.clamp(0, 1000);
        self.decay_by_mult = 1.0_f64 - (clamped as f64 * 0.001);
    }

    /// counter.py:266-278 decay_all_counters()
    pub fn decay_all_counters(&mut self) {
        let mult = self.decay_by_mult;
        for entry in &mut self.timetable {
            for time in &mut entry.times {
                *time = (*time as f64 * mult) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_counting() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        assert!(!counter.tick(42, increment));
        assert!(!counter.tick(42, increment));
        assert!(counter.tick(42, increment));
    }

    #[test]
    fn test_different_hashes() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let shift = counter.shift;
        let h1 = 1u64 << shift;
        let h2 = 2u64 << shift;
        assert!(!counter.tick(h1, increment));
        assert!(!counter.tick(h2, increment));
        assert!(!counter.tick(h1, increment));
        assert!(counter.tick(h1, increment));
        assert!(!counter.tick(h2, increment));
    }

    #[test]
    fn test_reset() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let h = 1u64 << counter.shift;
        counter.tick(h, increment);
        counter.tick(h, increment);
        counter.reset(h);
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_decay() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(10);
        let h = 1u64 << counter.shift;
        for _ in 0..8 {
            counter.tick(h, increment);
        }
        // default decay_by_mult = 1.0 (no decay). Set decay first.
        counter.set_decay(40); // decay_by_mult = 0.96
        // time ≈ 8 * (1/10) = 0.8, decay by 0.96 → 0.768
        counter.decay_all_counters();
        // Verify via a tick that doesn't fire (need ~0.232 more to reach 1.0)
        let index = counter._get_index(h);
        let subhash = JitCounter::_get_subhash(h);
        let entry = &counter.timetable[index];
        let mut time = 0.0f32;
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                time = entry.times[i];
                break;
            }
        }
        assert!(time > 0.7 && time < 0.8, "time={}", time);
    }

    #[test]
    fn test_auto_reset_on_fire() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let h = 1u64 << counter.shift;
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_fetch_next_hash() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let h1 = counter.fetch_next_hash();
        let h2 = counter.fetch_next_hash();
        assert_ne!(h1, h2);
        assert_ne!(counter._get_index(h1), counter._get_index(h2));
    }

    #[test]
    fn test_change_current_fraction() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(100);
        let h = 1u64 << counter.shift;
        counter.change_current_fraction(h, 0.98);
        // 0.98 + ~0.01 = ~0.99, not enough; two more ticks → ~1.0
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_size_parameter() {
        let counter = JitCounter::new(1024);
        assert_eq!(counter.size, 1024);
        // 0xFFFFFFFF >> shift = 1023 → shift = 22
        assert_eq!(counter.shift, 22);
    }
}

/// counter.py:309 DeterministicJitCounter — test-only, NOT_RPYTHON.
///
/// RPython: subclasses JitCounter, overrides _get_index to return the
/// raw hash (identity — no collision), uses a defaultdict timetable.
/// Rust: uses a VecAssoc<u64, Entry> to mirror the defaultdict approach.
pub struct DeterministicJitCounter {
    entries: majit_ir::vec_assoc::VecAssoc<u64, Entry>,
}

impl DeterministicJitCounter {
    /// counter.py:310-315 DeterministicJitCounter.__init__
    pub fn new() -> Self {
        DeterministicJitCounter {
            entries: majit_ir::vec_assoc::VecAssoc::new(),
        }
    }

    /// counter.py:318-319 _get_index — identity (no hash collision).
    #[inline(always)]
    fn _get_index(hash: u64) -> u64 {
        hash
    }

    /// counter.py:138-140 _get_subhash
    #[inline(always)]
    fn _get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py:122-126 compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        if threshold == 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py:185-202 tick — same logic but using identity _get_index.
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        let key = Self::_get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = self.entries.entry_or_insert_with(key, Entry::default);

        let n = if entry.subhashes[0] == subhash {
            0
        } else if entry.subhashes[1] == subhash {
            JitCounter::_swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            JitCounter::_swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            JitCounter::_swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            JitCounter::_swap(entry, 3)
        } else {
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        };

        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            entry.times[n] = counter as f32;
            false
        } else {
            self.reset(hash);
            true
        }
    }

    /// counter.py:232-237 reset
    pub fn reset(&mut self, hash: u64) {
        let key = Self::_get_index(hash);
        let subhash = Self::_get_subhash(hash);
        if let Some(entry) = self.entries.get_mut(&key) {
            for i in 0..ASSOCIATIVITY {
                if entry.subhashes[i] == subhash {
                    entry.times[i] = 0.0;
                }
            }
        }
    }

    /// counter.py:322 decay_all_counters — no-op for deterministic counter.
    pub fn decay_all_counters(&mut self) {}

    /// counter.py:326-327 _clear_all
    pub fn _clear_all(&mut self) {
        self.entries.clear();
    }
}
