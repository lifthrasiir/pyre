# Benchmark: integer list reverse (per-strategy ops)
# Exercises W_ListObject.reverse() on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + setarrayitem(ArrayS 8) swaps.
# On main, reverse() called items_to_vec() → Object strategy first.
# On this branch, reverse() stays in Integer strategy (no boxing overhead).
# REPS=9 (odd): final list is reversed, so lst[0]=N-1, lst[-1]=0.
#
def main():
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


main()
