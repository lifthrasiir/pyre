N = 350000


def main():
    data = b"abcdefghijklmnopqrstuvwxyz"
    i = 0
    acc = 0
    while i < N:
        b = data[i % len(data)]
        piece = data[(i & 7):(i & 7) + 5]
        acc = acc + b + len(piece)
        i = i + 1
    print(acc)


main()

