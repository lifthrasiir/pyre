N = 120000


def main():
    i = 0
    acc = 0
    while i < N:
        xs = [j * j for j in range(8) if ((j + i) & 1) == 0]
        d = {j: j + i for j in xs}
        acc = acc + sum(xs) + len(d)
        i = i + 1
    print(acc)


main()

