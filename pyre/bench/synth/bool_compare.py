N = 1500000


def main():
    i = 0
    acc = 0
    while i < N:
        a = i % 17
        b = i % 31
        if (a < b and b != 0) or a == 13:
            acc = acc + a
        else:
            acc = acc - b
        i = i + 1
    print(acc)


main()

