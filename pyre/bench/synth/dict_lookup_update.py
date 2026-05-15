N = 500000


def main():
    d = {}
    i = 0
    while i < 128:
        d[i] = i * 3
        i = i + 1
    i = 0
    acc = 0
    while i < N:
        k = i & 127
        acc = acc + d[k]
        d[k] = d[k] + 1
        i = i + 1
    print(acc + d[17] + len(d))


main()

