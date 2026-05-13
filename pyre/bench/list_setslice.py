# Benchmark: integer list setslice (per-strategy ops)
# Exercises W_ListObject slice assignment: lst[a:b] = [...] on Integer strategy.
# PYPYLOG confirms: guard_class(IntegerListStrategy) + new_array(3, ArrayS 8).
# On main, there was no setslice op (Object-only fallback).
# On this branch, setslice stays in Integer strategy when new items are plain ints.
#
N = 200000

def main():
    lst = [0] * 10
    i = 0
    while i < N:
        lst[2:5] = [i, i + 1, i + 2]
        i = i + 1
    print(lst)

main()
