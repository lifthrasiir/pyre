# Benchmark: integer list pop/append loop (per-strategy ops)
# Exercises W_ListObject.pop_end() / append() on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + ArrayS 8 ops only.
# On main, first pop() called items_to_vec() → Object strategy;
# on this branch, stays Integer throughout (no boxing overhead).

N = 300000


def main():
    lst = [0, 1, 2, 3, 4]
    i = 0
    while i < N:
        lst.append(i)
        lst.pop()
        i = i + 1
    print(len(lst), lst[0])


main()
