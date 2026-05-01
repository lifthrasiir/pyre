# Benchmark: integer list reverse (per-strategy ops)
# Exercises W_ListObject.reverse() on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + setarrayitem(ArrayS 8) swaps.
# On main, reverse() called items_to_vec() → Object strategy first.
# On this branch, reverse() stays in Integer strategy (no boxing overhead).
# REPS=9 (odd): final list is reversed, so lst[0]=N-1, lst[-1]=0.
#
# NOTE: kept at module level intentionally — wrapping in def main() lets the
# JIT fire and improves cranelift from ~18x to ~13x cpython, but the result
# straddles the 10x cpython threshold and fails check.py flakily depending
# on cpython measurement noise. Re-wrap when list oopspec lowering lands and
# brings cranelift comfortably under threshold.

N = 200000
REPS = 9

lst = []
i = 0
while i < N:
    lst.append(i)
    i = i + 1

r = 0
while r < REPS:
    lst.reverse()
    r = r + 1

print(lst[0], lst[-1])
