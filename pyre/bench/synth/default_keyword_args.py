N = 500000


def f(a, b=3, c=5):
    return a + b * 2 - c


def main():
    i = 0
    acc = 0
    while i < N:
        acc = acc + f(i)
        acc = acc + f(i, c=7)
        acc = acc + f(i, b=11, c=13)
        i = i + 1
    print(acc)


main()

