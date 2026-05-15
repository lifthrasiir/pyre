N = 2000000


def main():
    i = 1
    acc = 0
    while i < N:
        acc = acc + i
        acc = acc ^ (i << 1)
        acc = acc - (i >> 2)
        acc = acc + (i % 97)
        i = i + 1
    print(acc)


main()

