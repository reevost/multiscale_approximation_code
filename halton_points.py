import numpy as np


def fill_distance(points_array, domain_points):
    maximum = 0
    # point = np.array([])
    for y in domain_points:
        minimum = min(np.sqrt(np.sum((points_array-y)**2, axis=1)))
        if minimum > maximum:
            # point = y
            maximum = minimum
    return maximum


def separation_distance(points_array):
    minimum = np.inf
    for i in np.arange(len(points_array)):
        temp_array = np.delete(points_array, i, axis=0)
        m = min(np.sqrt(np.sum((temp_array-points_array[i])**2, axis=1)))
        if m < minimum:
            minimum = m
    return minimum


def next_prime():
    def is_prime(num):
        # "Checks if num is a prime value"
        for i in range(2, int(num**0.5)+1):
            if(num % i) == 0:
                return False
        return True

    prime = 3
    while 1:
        if is_prime(prime):
            yield prime
        prime += 2


def van_der_corput(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc


def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([van_der_corput(i, base) for i in range(size)])
    return seq  # return the list [x, y, z, ...] where x/y/z/.. are the list of the associated axis coordinates
    # easy to display with  plt.scatter(x,y)
