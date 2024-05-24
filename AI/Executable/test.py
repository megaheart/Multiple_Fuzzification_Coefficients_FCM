import multiprocessing

def calc_stuff(data):
    i, j = data
    return i * j

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        for result in pool.imap_unordered(calc_stuff, [(1, 2), (3, 4), (5, 6)]):
            print(result)
