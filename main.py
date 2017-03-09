"""
Simulate dynamics on the lattice
"""


import sys
import pickle, random
import itertools

import numpy as np
import networkx as nx

from tqdm import trange

from graph_utils import generate_graph


def get_neighbor(i, graph):
    neighs = graph.neighbors(i)
    return random.choice(neighs)

def get_threshold(i, lattice):
    val = lattice[i]
    return np.sum(lattice==val) / lattice.size

def simulate(resolution=40000, fname='results/data.dat'):
    """ Simulate Bornholdt/Sneppen model
    """
    N = 128
    tmax = resolution*2#80000
    alpha = 25e-6
    p = .5

    freq = int(tmax / resolution)
    print(f'Saving data every {freq*N**2} time steps, resulting in {int(tmax/(freq*N**2))} ({int(tmax/freq)}) data points')

    lattice = np.zeros((N,N))
    strategy_num = np.unique(lattice).size

    graph = generate_graph(N, p)

    strats = {} # strategy history of each node
    snapshots = []

    get_random = lambda: tuple(np.random.randint(N, size=2)) # get index of random node in system

    for index, s in np.ndenumerate(lattice):
        strats[index] = set([s])

    # simulate system
    for t in trange(tmax*N**2):
        i = get_random()
        j = get_neighbor(i, graph)

        thres = get_threshold(j, lattice) # n_j / N^2
        if np.random.random() < thres and not lattice[j] in strats[i]:
            lattice[i] = lattice[j]
            strats[i].add(lattice[i])

        if np.random.random() < alpha:
            k = get_random()
            lattice[k] = strategy_num
            strats[k].add(lattice[k])
            strategy_num += 1

        if t % int(freq*N**2) == 0:
            snapshots.append((t, lattice.copy()))

    # save result
    with open(fname, 'wb') as fd:
        pickle.dump({
            'snapshots': snapshots,
            'graph': graph,
            'config': {
                'N': N,
                'p': p,
                'tmax': tmax,
                'alpha': alpha
            }
        }, fd)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output data file>')
        exit(-1)

    simulate(fname=sys.argv[1])
