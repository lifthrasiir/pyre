N = 75000


class Accumulator:
    def __init__(self, x):
        self.x = x

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.x = self.x + 1
        return False


def main():
    i = 0
    acc = 0
    while i < N:
        with Accumulator(i & 15) as obj:
            acc = acc + obj.x
        acc = acc + obj.x
        i = i + 1
    print(acc)


main()
