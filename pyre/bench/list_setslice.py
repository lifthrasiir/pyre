# Benchmark: integer list setslice (per-strategy ops)
# Exercises W_ListObject slice assignment: lst[a:b] = [...] on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + new_array(3, ArrayS 8).
# On main, there was no setslice op (Object-only fallback).
# On this branch, setslice stays in Integer strategy when new items are plain ints.
#
# NOTE: kept at module level intentionally — wrapping in def main() lets the
# JIT fire and exposes a TypeError("list indices must be integers, not tuple")
# panic in opcode_ops.rs:178 (slice assignment lowering bug). Re-wrap once
# that's fixed.

N = 200000

lst = [0] * 10
i = 0
while i < N:
    lst[2:5] = [i, i + 1, i + 2]
    i = i + 1
print(lst)
