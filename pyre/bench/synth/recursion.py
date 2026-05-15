def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def main():
    i = 0
    acc = 0
    while i < 28:
        acc = acc + fib(18)
        i = i + 1
    print(acc)


main()

