N = 120000


def gen(n):
    i = 0
    while i < n:
        yield i * 2 + 1
        i = i + 1


def main():
    i = 0
    acc = 0
    while i < N:
        for x in gen(6):
            acc = acc + x + (i & 3)
        i = i + 1
    print(acc)


main()

