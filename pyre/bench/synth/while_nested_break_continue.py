N = 60000


def main():
    i = 0
    acc = 0
    while i < N:
        j = 0
        while j < 12:
            if j == 5:
                j = j + 1
                continue
            if j == 10:
                break
            acc = acc + ((i + j) % 23)
            j = j + 1
        i = i + 1
    print(acc)


main()

