N = 250000


def may_fail(i):
    if (i & 31) == 0:
        raise ValueError(i)
    return i & 7


def main():
    i = 0
    acc = 0
    while i < N:
        try:
            acc = acc + may_fail(i)
        except ValueError as e:
            acc = acc - (e.args[0] & 15)
        finally:
            acc = acc + 1
        i = i + 1
    print(acc)


main()

