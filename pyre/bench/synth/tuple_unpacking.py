N = 900000


def pair(i):
    return i, i + 1


def main():
    i = 0
    acc = 0
    while i < N:
        a, b = pair(i)
        c, d, e = (b, a + b, a - b)
        acc = acc + c + d + e
        i = i + 1
    print(acc)


main()

