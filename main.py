"""
Simulate dynamics on the lattice
"""


import sys
import json, time
import itertools

import networkx as nx

from graph_utils import generate_graph
from simulation.libsim import Simulator


def simulate(p=0, alpha=25e-6, resolution=40000, fname='results/data.json'):
    """ Simulate Bornholdt/Sneppen model
    """
    N = 32 #128
    tmax = resolution*2 #80000

    freq = int(tmax / resolution) #1/N**2
    print(f'Saving data every {freq*N**2} time steps, resulting in {int(tmax/(freq*N**2))} ({int(tmax/freq)}) data points')

    graph_mat = nx.to_numpy_matrix(generate_graph(N, p))
    sim = Simulator(N, p, alpha, tmax, freq, fname, graph_mat.tolist())

    start = time.time()
    snapshot_num = sim.run()
    end = time.time()
    assert snapshot_num == int(tmax/freq), snapshot_num

    print(f' > Runtime: {end-start:.0f}s')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output data file (data{{p}}{{alpha}})>')
        exit(-1)

    p_vals = [0, .5, 1]
    alpha_vals = [4e-4, 2.5e-6, 1e-7]

    for p, alpha in itertools.product(p_vals, alpha_vals):
        fname = sys.argv[1].format(p=p, alpha=alpha)
        print(f'Simulating with p={p}, alpha={alpha} ({fname})')
        simulate(p=p, alpha=alpha, fname=fname)
