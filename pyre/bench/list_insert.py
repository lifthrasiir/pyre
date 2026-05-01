# Benchmark: integer list insert at front (per-strategy ops)
# Exercises W_ListObject.insert() on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + ll_arraymove + setarrayitem(ArrayS 8).
# On main, first insert() called items_to_vec() → Object strategy;
# on this branch, stays Integer throughout (no boxing overhead).
# insert(0, x) is O(n) per call → total O(n^2); small N is intentional.
#
# NOTE: kept at module level intentionally — wrapping in def main() lets the
# JIT fire and exposes a dynasm-side wrong-output bug. Re-wrap once that's
# fixed.

N = 50000

lst = []
i = 0
while i < N:
    lst.insert(0, i)
    i = i + 1
print(lst[0], lst[-1], len(lst))
