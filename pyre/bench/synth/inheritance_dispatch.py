N = 500000


class Base:
    def value(self, x):
        return x + 1


class Left(Base):
    def value(self, x):
        return x + 3


class Right(Base):
    def value(self, x):
        return x - 5


def main():
    objs = [Base(), Left(), Right()]
    i = 0
    acc = 0
    while i < N:
        acc = acc + objs[i % 3].value(i)
        i = i + 1
    print(acc)


main()

