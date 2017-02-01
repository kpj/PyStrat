"""
More elaborate plots based on the data
"""

import sys
import pickle
import collections

import numpy as np
import networkx as nx

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

from graph_utils import plot_graph


def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(np.ceil(i * length / num))]

def overview_plot(data):
    """ Plot #strategies over time and system snapshots
    """
    N = data['config']['N']

    # compute statistics
    strat_nums = []
    for t, lattice in data['snapshots']:
        snum = np.unique(lattice).size
        strat_nums.append((t/N**2, snum))

    snapshots = list(takespread(data['snapshots'], 5))

    # plotting
    gs = mpl.gridspec.GridSpec(2, len(snapshots))

    for i, (t, lattice) in enumerate(sorted(snapshots)):
        ax = plt.subplot(gs[0, i])
        ax.imshow(lattice, interpolation='nearest')
        ax.set_title(rf'$t={int(t/N**2):d}$', fontsize=10)
        ax.tick_params(axis='both', which='both', labelsize=5)

    ax = plt.subplot(gs[1, :])
    ax.plot(*zip(*strat_nums))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('#strategies')

    plt.savefig('images/result.pdf')

def site_distribution(data):
    """ Figure 4 of paper
    """
    N = data['config']['N']
    alpha = data['config']['alpha']

    # aggregate data
    counts = collections.defaultdict(set)
    for t, lattice in tqdm(data['snapshots']):
        for strat in range(int(np.max(lattice))+1):
            raw = np.where(lattice==strat)
            res = set([idx for idx in zip(*raw)])
            counts[strat].update(res) #= counts[strat].union(res)
    counts = dict(counts)

    sites = []
    for strat, coords in counts.items():
        sites.append(len(coords))

    binning = np.bincount(sites)
    scale = np.array(sites)**-2.5

    # plot data
    plt.figure()

    plt.loglog(binning, 'o', label=rf'$\alpha={alpha}$')
    plt.loglog(sites, scale, label=r'$s^{-2.5}$')

    plt.title('Sites with certain number of ideas distribution')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$n$')
    plt.legend(loc='best')

    plt.savefig('images/site_distribution.pdf')

def waiting_times(data):
    """ Figure 3 of paper
    """
    N = data['config']['N']

    # find dominant strategy at each point in time
    get_dom_strat = lambda lat: np.argmax(np.bincount(lat.ravel().astype(np.int64)))
    dom_strats = np.asarray(list(map(lambda e: get_dom_strat(e[1]), data['snapshots'])))

    # detect dominant strategy changes (and durations)
    indices = np.where(dom_strats[:-1] != dom_strats[1:])[0]

    durations = np.r_[indices[0], np.diff(indices)[:-1]].astype(float)
    #durations /= N**2

    # plot result
    plt.figure()

    sns.distplot(durations)

    plt.title('Distribution of waiting times')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'freq')

    plt.savefig('images/waiting_times.pdf')

def dominant_states(data):
    """ Figure 2 of paper
    """
    N = data['config']['N']

    # compute statistics
    ts = []
    strats = collections.defaultdict(list)
    max_strat = int(np.max(data['snapshots'][-1][1]))
    for t, lattice in tqdm(data['snapshots']):
        bins = np.bincount(lattice.ravel().astype(np.int64))
        #dom_strats = np.argsort(bins)[::-1]

        ts.append(t / N**2)
        for s in range(max_strat):#dom_strats:
            freq = np.sum(lattice == s) / N**2
            strats[s].append(freq)
    strats = dict(strats)

    # plot
    plt.figure()

    for s, vals in strats.items():
        plt.plot(ts, vals, label=s)

    plt.title('Strategy cluster sizes')
    plt.xlabel(r'$t$')
    plt.ylabel('relative cluster size')

    plt.savefig('images/dominant_states.pdf')

def main(fname):
    with open(fname, 'rb') as fd:
        data = pickle.load(fd)
    print('Parsing', len(data['snapshots']), 'entries')

    plot_graph(data['graph'])
    overview_plot(data)
    site_distribution(data)
    waiting_times(data)
    dominant_states(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <data file>')
        exit(-1)

    main(sys.argv[1])
