N = 500000


def make_adder(k):
    def inner(x):
        return x + k
    return inner


def main():
    add5 = make_adder(5)
    add9 = make_adder(9)
    i = 0
    acc = 0
    while i < N:
        acc = acc + add5(i)
        acc = acc - add9(i // 2)
        i = i + 1
    print(acc)


main()

