N = 800000


def add3(a, b, c):
    return a + b + c


def mix(a, b):
    if a & 1:
        return add3(a, b, 7)
    return add3(b, a, -3)


def main():
    i = 0
    acc = 0
    while i < N:
        acc = acc + mix(i, acc & 255)
        i = i + 1
    print(acc)


main()

