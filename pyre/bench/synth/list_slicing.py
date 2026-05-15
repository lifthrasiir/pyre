N = 180000


def main():
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    i = 0
    while i < N:
        xs[2:5] = [i & 255, (i + 1) & 255, (i + 2) & 255]
        ys = xs[1:6]
        xs[0] = ys[2]
        i = i + 1
    print(sum(xs))


main()

