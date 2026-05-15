N = 250000


def main():
    words = ["alpha", "beta", "gamma", "delta"]
    i = 0
    acc = 0
    while i < N:
        s = words[i & 3]
        t = s + ":" + str(i & 255)
        if t.startswith("a") or t.endswith("7"):
            acc = acc + len(t)
        else:
            acc = acc - len(s)
        i = i + 1
    print(acc)


main()

