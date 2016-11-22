import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import trange


def get_neighbor(i, N):
    x,y = i
    r = lambda: np.random.randint(-1,2)
    return (x+r()) % N, (y+r()) % N

def get_threshold(i, lattice):
    val = lattice[i]
    return np.sum(lattice==val) / lattice.size

def main():
    N = 128
    tmax = 100000
    alpha = 25e-6
    strategy_num = 50

    lattice = np.random.randint(strategy_num, size=(N,N))
    strats = {} # strategy history of each node
    strat_nums = []
    snapshots = {}

    get = lambda: tuple(np.random.randint(N, size=2)) # get index of random node in system

    for index, s in np.ndenumerate(lattice):
        strats[index] = set([s])

    for t in trange(tmax):
        i = get()
        j = get_neighbor(i, N)

        thres = get_threshold(j, lattice)
        if np.random.random() < thres and not lattice[j] in strats[i]:
            lattice[i] = lattice[j]
            strats[i].add(lattice[i])

        if np.random.random() < alpha:
            k = get()
            lattice[k] = strategy_num
            strats[k].add(lattice[k])
            strategy_num += 1

        # book-keeping
        snum = np.unique(lattice).size
        strat_nums.append(snum)

        if t % (int(tmax/5)) == 0:
            snapshots[t] = lattice.copy()

    # plotting
    gs = mpl.gridspec.GridSpec(2, 5)

    for i, (t, lattice) in enumerate(sorted(snapshots.items())):
        ax = plt.subplot(gs[0, i])
        ax.imshow(lattice, interpolation='nearest')
        ax.set_title(r'$t={}$'.format(t))
        ax.tick_params(axis='both', which='both', labelsize=5)

    ax = plt.subplot(gs[1, :])
    ax.plot(strat_nums)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('#strategies')

    plt.savefig('result.pdf')

if __name__ == '__main__':
    main()
