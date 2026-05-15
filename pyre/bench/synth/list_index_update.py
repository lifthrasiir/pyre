N = 900000


def main():
    xs = [0] * 32
    i = 0
    while i < N:
        j = i & 31
        xs[j] = xs[j] + i
        xs[(j + 7) & 31] = xs[(j + 7) & 31] - j
        i = i + 1
    print(sum(xs))


main()

