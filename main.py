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
    tmax = 80000
    alpha = 25e-6

    lattice = np.zeros((N,N))
    strategy_num = np.unique(lattice).size

    strats = {} # strategy history of each node
    strat_nums = []
    snapshots = {}

    get_random = lambda: tuple(np.random.randint(N, size=2)) # get index of random node in system

    for index, s in np.ndenumerate(lattice):
        strats[index] = set([s])

    for t in trange(tmax*N**2):
        i = get_random()
        j = get_neighbor(i, N)

        thres = get_threshold(j, lattice) # n_j / N^2
        if np.random.random() < thres and not lattice[j] in strats[i]:
            lattice[i] = lattice[j]
            strats[i].add(lattice[i])

        if np.random.random() < alpha:
            k = get_random()
            lattice[k] = strategy_num
            strats[k].add(lattice[k])
            strategy_num += 1

        # book-keeping
        snum = np.unique(lattice).size
        strat_nums.append((t/N**2, snum))

        if t % (int(tmax/5)) == 0:
            snapshots[t/N**2] = lattice.copy()

    # plotting
    gs = mpl.gridspec.GridSpec(2, 5)

    for i, (t, lattice) in enumerate(sorted(snapshots.items())):
        ax = plt.subplot(gs[0, i])
        ax.imshow(lattice, interpolation='nearest')
        ax.set_title(r'$t={}$'.format(t), fontsize=10)
        ax.tick_params(axis='both', which='both', labelsize=5)

    ax = plt.subplot(gs[1, :])
    ax.plot(*zip(*strat_nums))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('#strategies')

    plt.savefig('result.pdf')

if __name__ == '__main__':
    main()
