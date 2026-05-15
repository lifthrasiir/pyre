N = 700000


def main():
    s = set()
    i = 0
    while i < 256:
        s.add(i * 3)
        i = i + 1
    i = 0
    acc = 0
    while i < N:
        if (i % 1024) in s:
            acc = acc + i
        else:
            acc = acc - (i & 15)
        i = i + 1
    print(acc + len(s))


main()

