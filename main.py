import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange


def get_neighbor(i, j, N):
    r = lambda: np.random.randint(-1,2)
    return (i+r()) % N, (j+r()) % N

def get_threshold(i, j, lattice):
    val = lattice[i,j]
    return np.sum(lattice==val) / lattice.size

def main():
    N = 128
    tmax = 10000
    alpha = 2.5e-5
    strategy_num = 50

    lattice = np.random.randint(strategy_num, size=(N,N))
    strats = {}
    strat_nums = []

    for index, s in np.ndenumerate(lattice):
        strats[index] = set([s])

    for t in trange(tmax):
        cur = lattice.copy()

        i,j = np.random.randint(N, size=2)
        if np.random.random() < alpha:
            cur[i,j] = strategy_num
            strategy_num += 1
        else:
            k,l = get_neighbor(i,j,N)
            val = cur[k,l].copy()
            thres = get_threshold(k,l,cur)
            if np.random.random() < thres and not val in strats[(i,j)]:
                cur[i,j] = val
        lattice = cur

        snum = np.unique(cur).shape[0]
        strat_nums.append(snum)

    plt.subplot(211)
    plt.imshow(lattice, interpolation='nearest')
    plt.colorbar()
    plt.subplot(212)
    plt.plot(strat_nums)
    plt.xlabel(r'$t$')
    plt.ylabel('#strategies')
    plt.savefig('result.pdf')

if __name__ == '__main__':
    main()
