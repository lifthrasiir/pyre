N = 600000


class Counter:
    def __init__(self, base):
        self.value = base

    def add(self, x):
        self.value = self.value + x
        return self.value


def main():
    c = Counter(3)
    i = 0
    acc = 0
    while i < N:
        acc = acc + c.add(i & 7)
        i = i + 1
    print(acc + c.value)


main()

