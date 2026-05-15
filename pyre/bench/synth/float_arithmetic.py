N = 800000


def main():
    i = 0
    x = 1.0
    y = 0.25
    while i < N:
        x = x + y
        y = y * 1.000001 + 0.000003
        x = x / 1.000002
        if x > 1000.0:
            x = x - 999.5
        i = i + 1
    print(int(x * 1000.0) + int(y * 1000.0))


main()

