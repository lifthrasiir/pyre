import math

N = 400000


def main():
    i = 0
    acc = 0
    while i < N:
        x = math.sqrt((i & 255) + 1)
        acc = acc + int(x * 1000.0)
        i = i + 1
    print(acc)


main()

