"""
Interpolate between lattice and scale-free graph
"""

import os
import pickle
import itertools

import numpy as np
import networkx as nx

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm


def plot_graph(graph, ax):
    """ Plot given graph
    """
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    pos = dict(zip(graph, graph))

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=2, linewidths=.2)
    #nx.draw_networkx_labels(graph, pos, ax=ax, font_size=2)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=.2, linewidths=.2)

def plot_degree_distribution(graph, ax, log=True, **kwargs):
    """ Plot degree distribution of given graph
    """
    degs = zip(range(len(graph)), nx.degree_histogram(graph))
    degs = list(filter(lambda x: x[0]>0 and x[1]>0, degs)) # filter 0

    ax.plot(*zip(*degs), 'o-', **kwargs)

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_title('Degree distribution')
    ax.set_xlabel('degree')
    ax.set_ylabel('count')

def generate_graph(N, p):
    """ Combine lattice and scale-free graph.
        With $p$ choose entry from lattice, with $1-p$ from SF graph
    """

    # generate lattice
    grid_graph = nx.Graph()
    for x,y in itertools.product(range(N), repeat=2):
        for dx,dy in itertools.product(range(-1,2), repeat=2):
            grid_graph.add_edge((x,y), ((x+dx)%N,(y+dy)%N))

    # generate scale-free graph
    ba_graph = nx.barabasi_albert_graph(N**2, 1)

    # combine all graphs
    grid_mat = nx.to_numpy_matrix(grid_graph)
    ba_mat = nx.to_numpy_matrix(ba_graph)
    assert grid_mat.shape == ba_mat.shape == (N**2, N**2), (grid_mat.shape, ba_mat.shape)

    res_mat = -np.ones((N**2, N**2))
    for i,j in np.ndindex(res_mat.shape):
        res_mat[i,j] = ba_mat[i,j] if np.random.random() < p else grid_mat[i,j]
    assert (res_mat != -1).all(), res_mat

    res_graph = nx.from_numpy_matrix(res_mat)

    # reset node names
    mapping = dict(zip(res_graph.nodes(), grid_graph.nodes()))
    nx.relabel_nodes(res_graph, mapping, copy=False)

    return res_graph

def main(N=50, fname='cache/test_graphs.pkl'):
    p_vals = [1., .9, .8, .5, .2, 0.] #np.r_[1, 1-np.logspace(-1, 0, 4), .2]

    # generate graphs
    if not os.path.exists(fname):
        graph_list = []
        for p in tqdm(p_vals):
            graph = generate_graph(N, p)
            graph_list.append((p, graph))
        with open(fname, 'wb') as fd:
            pickle.dump(graph_list, fd)
    else:
        print(f'Cached {fname}')
        with open(fname, 'rb') as fd:
            graph_list = pickle.load(fd)

    # aggregated plot
    plt.figure()

    colors = itertools.cycle([e['color'] for e in list(plt.style.library['classic']['axes.prop_cycle'])])
    for p, graph in tqdm(graph_list):
        plot_degree_distribution(
            graph, plt.gca(),
            label=rf'$p={p:.2}$', color=next(colors))

    plt.legend(loc='best')
    plt.savefig('images/degree_distr_overview.pdf')

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    main()
