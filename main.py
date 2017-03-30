"""
Simulate dynamics on the lattice
"""


import sys
import json, time
import itertools

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

from joblib import Parallel, delayed, cpu_count

from graph_utils import generate_graph
from simulation.libsim import Simulator


def convert_matrix(matrix_graph):
    """ Convert adjacency matrix to data structure usable in C++
    """
    source, target = np.where(matrix_graph==1)
    graph_repr = [[] for _ in range(matrix_graph.shape[0])]
    for i, j in zip(source, target):
        graph_repr[i].append(int(j))
    return graph_repr

def simulate(p=0, alpha=25e-6, resolution=40000, fname='results/data.json'):
    """ Simulate Bornholdt/Sneppen model
    """
    N = 32 #128
    tmax = resolution*2 #80000

    freq = int(tmax / resolution) #1/N**2
    print(f'Simulating with p={p}, alpha={alpha} ({fname})')
    print(f'Saving data every {freq*N**2} time steps, resulting in {int(tmax/(freq*N**2))} ({int(tmax/freq)}) data points')

    node_list = [(i//N, i%N) for i in range(N**2)]
    graph = generate_graph(N, p)

    graph_mat = nx.to_numpy_matrix(graph, nodelist=node_list)
    graph_repr = convert_matrix(graph_mat)

    sim = Simulator(N, p, alpha, tmax, freq, fname, graph_repr)

    start = time.time()
    snapshot_num = sim.run()
    end = time.time()
    assert snapshot_num == int(tmax/freq), snapshot_num

    print(f' > Runtime: {end-start:.0f}s')

    # save graph
    with open(fname) as fd:
        data = json.load(fd)
    data['graph'] = json_graph.adjacency_data(graph)
    with open(fname, 'w') as fd:
        json.dump(data, fd)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output data file (data{{p}}{{alpha}})>')
        exit(-1)

    p_vals = [0, .5, 1]
    alpha_vals = [4e-4, 2.5e-6, 1e-7]

    core_num = int(4/5 * cpu_count())
    Parallel(n_jobs=core_num)(
        delayed(simulate)(
            p=p, alpha=alpha,
            fname=sys.argv[1].format(p=p, alpha=alpha)
        ) for p, alpha in itertools.product(p_vals, alpha_vals))
