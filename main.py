import sys
import pickle, random
import itertools

import numpy as np
import networkx as nx

from tqdm import trange


def get_neighbor(i, graph):
    neighs = graph.neighbors(i)
    return random.choice(neighs)

def get_threshold(i, lattice):
    val = lattice[i]
    return np.sum(lattice==val) / lattice.size

def generate_graph(N, scale_free=True):
    # generate lattice
    grid_graph = nx.Graph()
    for x,y in itertools.product(range(N), repeat=2):
        for dx,dy in itertools.product(range(-1,2), repeat=2):
            grid_graph.add_edge((x,y), ((x+dx)%N,(y+dy)%N))

    # generate scale-free graph
    if scale_free:
        ba_graph = nx.barabasi_albert_graph(N**2, 1)
    else:
        ba_graph = np.zeros((N**2,N**2))

    # combine all graphs
    mat = np.logical_or(
        nx.to_numpy_matrix(grid_graph),
        nx.to_numpy_matrix(ba_graph))
    graph = nx.from_numpy_matrix(mat)

    # reset node names
    mapping = dict(zip(graph.nodes(), grid_graph.nodes()))
    nx.relabel_nodes(graph, mapping, copy=False)

    return graph

def simulate(resolution=40000, fname='results/data.dat'):
    """ Simulate Bornholdt/Sneppen model
    """
    N = 128
    tmax = resolution*2#80000
    alpha = 25e-6

    freq = int(tmax / resolution)

    lattice = np.zeros((N,N))
    strategy_num = np.unique(lattice).size

    graph = generate_graph(N)

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
                'tmax': tmax,
                'alpha': alpha
            }
        }, fd)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('Usage: {} [data file]'.format(sys.argv[0]))
        exit(-1)

    if len(sys.argv) == 1:
        simulate()
    else:
        simulate(fname=sys.argv[1])
