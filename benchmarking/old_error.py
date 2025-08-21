# =========================================================================================================#
# =========================================================================================================#
# 	Numerical Differentiation Coefficients and Computation
# =========================================================================================================#
# 	Compute coefficients for various orders of derivatives specified by either
# 	a chain length (npt) or an order of error big_O(eo)
# =========================================================================================================#
# =========================================================================================================#
import math
import itertools
from math import factorial as fact


def afun(x, sup):
    result, lx, b1 = 0, len(x), ''
    for i in range(0, lx):
        b1 += str(i)
    b2 = 1
    G = list(itertools.combinations(b1, lx - sup))
    for i in range(0, len(G)):
        b2 = 1
        for j in range(0, len(G[i])):
            b2 *= x[int(G[i][j])]
        result += b2
    return result


def Del(x, i, j):
    return x[int(j)] - x[int(i)]


def Del2(x, y, i, j):
    return (y[j] - y[i])


def errorO(x, indx, do, eo, npt_guess):
    Error, npt1 = eo + 1, npt_guess - 1
    while Error > eo:
        try:
            npt1 = npt1 + 1
            lx, K, N, hold = int(npt1 + 1), int(npt1 / 2), int(len(x) - 1), []
            fac1, fac2, fac3 = 0, 1., 0
            fac1 = 1. / fact(npt1 + 1)
            # factor = 0
            if indx < K:
                factor = 0
            if indx >= K and indx <= N - K - 1:
                factor = indx - K
            if indx >= N - K:
                factor = N - npt1
            for k in range(0, lx):
                if k == indx - factor:
                    None
                else:
                    fac2 *= 1.0 * Del(x, k + factor, indx)
            for j in range(0, lx):
                delta, prod = [], 1
                for k in range(0, lx):
                    if k == j or k == indx - factor:
                        None
                    else:
                        prod *= 1.0 * Del(x, j + factor, k + factor)
                if j == indx - factor:
                    hold.append(0)
                else:
                    hold.append(((1.0 * Del(x, j + factor, indx)) ** (npt1 - 1)) / prod)
            fac3 = sum(hold)
            Error = abs(fac1 * fac2 * fac3)
        except OverflowError:
            print("Overflow error encountered!")
            #			print '\tThis error value of %s requires %s points!\n\tA total of 10 points used instead at x = %s.' % (eo,npt1,x[indx])
            Error = 0
    return npt1, Error
