N = 700000


def main():
    xs = []
    i = 0
    acc = 0
    while i < N:
        xs.append(i)
        if len(xs) > 32:
            acc = acc + xs.pop(0)
        if (i & 7) == 0:
            xs.append(i + 3)
            acc = acc - xs.pop()
        i = i + 1
    print(acc + len(xs) + sum(xs))


main()

