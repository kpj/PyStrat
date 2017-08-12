"""
Interpolate between lattice and scale-free graph
"""

import os
import pickle
import itertools

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from joblib import Parallel, delayed, cpu_count


def graph_from_json(json_dict):
    """ Get networkx graph from json string
    """
    # convert nodes from list to tuple (make hashable)
    for e in json_dict['nodes']:
        if isinstance(e['id'], list):
            e['id'] = tuple(e['id'])
    for row in json_dict['adjacency']:
        for e in row:
            if isinstance(e['id'], list):
                e['id'] = tuple(e['id'])

    graph = json_graph.adjacency_graph(json_dict)
    return graph

def plot_graph(graph, ax_orig=None, fname_app=''):
    """ Plot given graph
    """
    if ax_orig is None:
        ax = plt.gca()
    else:
        ax = ax_orig

    if isinstance(graph, dict):
        graph = graph_from_json(graph)

    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    pos = dict(zip(graph, graph))

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=2, linewidths=.2)
    #nx.draw_networkx_labels(graph, pos, ax=ax, font_size=2)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=.2, linewidths=.2)

    if ax_orig is None:
        plt.tight_layout()
        plt.savefig(f'images/graph{fname_app}.pdf')
        plt.close()

def plot_degree_distribution(graph_list, ax, log=True, **kwargs):
    """ Plot degree distribution of given graph
    """
    assert all(len(g)==len(graph_list[0]) for g in graph_list)

    # bin degrees
    all_degrees = np.array([d for degs in map(nx.degree, graph_list) for d in degs.values()])

    bins = np.logspace(0, np.log10(max(all_degrees)), num=20)
    hist, bin_edges = np.histogram(all_degrees, bins)

    # extract non-zero entries
    nonzero_idx = np.where(hist>0)[0]
    deg_mids = np.array([bin_edges[i] for i in range(len(bin_edges)-1)])

    # plot data
    ax.plot(deg_mids[nonzero_idx], hist[nonzero_idx], '.-', **kwargs)

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel(r'$degree$')
    ax.set_ylabel(r'count')

def generate_graph(N, p):
    """ Combine lattice and scale-free graph.
        With $1-p$ choose entry from lattice, with $p$ from SF graph
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

def main(N=50, reps=100, fname='cache/test_graphs.pkl'):
    """ Generate degree-distribution overiew of interpolated graphs
    """
    p_vals = [1., .9, .8, .5, .2, 0.]

    # generate graphs
    core_num = int(cpu_count() * 4/5)
    if not os.path.exists(fname):
        graph_list = []
        for p in tqdm(p_vals):
            graphs = Parallel(n_jobs=core_num)(delayed(generate_graph)(N, p) for _ in trange(reps))
            graph_list.append((p, graphs))
        with open(fname, 'wb') as fd:
            pickle.dump(graph_list, fd)
    else:
        print(f'Cached {fname}')
        with open(fname, 'rb') as fd:
            graph_list = pickle.load(fd)

    # aggregated plot
    plt.figure()

    colors = itertools.cycle([e['color'] for e in list(plt.style.library['classic']['axes.prop_cycle'])])
    for p, graphs in tqdm(graph_list):
        plot_degree_distribution(
            graphs, plt.gca(),
            label=rf'$p={p:.2}$', color=next(colors))

    plt.legend(loc='best')
    plt.savefig('images/degree_distr_overview.pdf')

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    main()
