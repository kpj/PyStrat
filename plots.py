"""
More elaborate plots based on the data
"""

import os
import sys
import json
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

def overview_plot(data, spread_freq=5, fname_app=''):
    """ Plot #strategies over time and system snapshots
    """
    N = data['config']['N']

    # compute statistics
    strat_nums = []
    for t, lattice in zip(data['snapshot_times'], data['snapshots']):
        snum = np.unique(lattice).size
        strat_nums.append((t/N**2, snum))

    snapshots = list(takespread(data['snapshots'], spread_freq))
    snapshot_times = list(takespread(data['snapshot_times'], spread_freq))
    assert len(snapshots) == len(snapshot_times)

    # plotting
    gs = mpl.gridspec.GridSpec(2, len(snapshots))

    for i, (t, lattice) in enumerate(sorted(zip(snapshot_times, snapshots))):
        ax = plt.subplot(gs[0, i])
        ax.imshow(lattice, interpolation='nearest')
        ax.set_title(rf'$t={int(t/N**2):d}$', fontsize=10)
        ax.tick_params(axis='both', which='both', labelsize=5)

    ax = plt.subplot(gs[1, :])
    ax.plot(*zip(*strat_nums))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('#strategies')

    plt.savefig(f'images/result{fname_app}.pdf')

def site_distribution(data, fname_app=''):
    """ Figure 4 of paper
    """
    N = data['config']['N']
    alpha = data['config']['alpha']

    # aggregate data
    counts = collections.defaultdict(set)
    for lattice in tqdm(data['snapshots']):
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

    plt.savefig(f'images/site_distribution{fname_app}.pdf')

def get_dominant_strategy(lattice, num=1):
    """ Given a lattice, return most common strategies
    """
    lattice_1d = np.asarray(lattice).ravel().astype(int)
    bc = np.bincount(lattice_1d)

    if num == 1:
        return np.argmax(bc)
    else:
        nz_len = (bc != 0).sum()
        return np.r_[np.argsort(bc)[::-1][:min(num, nz_len)], -np.ones(max(0, num-nz_len))]

def get_domain_durations(series):
    """ Compute lengths of dominant strategies
    """
    series = np.asarray(series)
    indices = np.where(series[:-1] != series[1:])[0]
    diff = np.diff(indices)
    return np.r_[indices[0]+1, diff, series.size-indices[-1]-1].astype(float)

def waiting_times(all_data):
    """ Figure 3 of paper
    """
    print('Computing waiting times')
    result = {}
    for data in all_data:
        N = data['config']['N']
        p = data['config']['p']
        alpha = data['config']['alpha']
        print(f'p = {p}, alpha = {alpha}')

        # find dominant strategy at each point in time
        print(' > Finding dominant strategies')
        dom_strats = np.asarray(list(map(lambda e: get_dominant_strategy(e), data['snapshots'])))
        print(f'  >> Found {np.unique(dom_strats).size} unique strategies')

        # detect dominant strategy changes (and durations)
        print(' > Computing durations')
        durations = get_domain_durations(dom_strats)
        durations /= N**2
        print(f'  >> Found {durations.size} durations')

        # store result
        assert (p,alpha) not in result
        result[(p,alpha)] = durations

    # plot result
    print(' > Plotting')
    plt.figure()
    for (p, alpha), durations in result.items():
        sns.distplot(durations, kde=False, label=rf'$p={p},\alpha={alpha}$')

    plt.title('Distribution of waiting times')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'count')
    plt.legend(loc='best')

    plt.savefig('images/waiting_times.pdf')

def dominant_states(data, fname_app=''):
    """ Figure 2 of paper
    """
    N = data['config']['N']

    # compute statistics
    ts = []
    strats = collections.defaultdict(list)
    max_strat = int(np.max(data['snapshots'][-1]))
    for t, lattice in tqdm(zip(data['snapshot_times'], data['snapshots'])):
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

    plt.savefig(f'images/dominant_states{fname_app}.pdf')

def main(fnames):
    all_data = []
    for fname in fnames:
        with open(fname, 'rb') as fd:
            data = json.load(fd)
            data['snapshots'] = [np.reshape(s, (data['config']['N'], data['config']['N'])) for s in data['snapshots']]

        all_data.append(data)
        print(f'[{fname}] Parsing {len(data["snapshots"])} entries')

        f_app = os.path.basename(fname)
        #plot_graph(data['graph'])
        overview_plot(data, fname_app=f'_{f_app}')
        #site_distribution(data, fname_app=f'_{f_app}')
        #dominant_states(data, fname_app=f'_{f_app}')

    waiting_times(all_data)

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <data file>...')
        exit(-1)

    main(sys.argv[1:])
