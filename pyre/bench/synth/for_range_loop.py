N = 600000


def main():
    acc = 0
    for i in range(N):
        acc = acc + (i % 19)
    for i in range(10, 200000, 3):
        acc = acc - (i % 11)
    print(acc)


main()

