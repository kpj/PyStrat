"""
More elaborate plots based on the data
"""

import os
import sys
import json
import collections

import numpy as np
import pandas as pd
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
        ax.imshow(
            lattice, interpolation='nearest',
            cmap=mpl.colors.ListedColormap(sns.color_palette('husl')))
        ax.set_title(rf'$t={int(t/N**2):d}$')
        ax.tick_params(
            axis='both', which='both',
            bottom='off', top='off', right='off', left='off',
            labelbottom='off', labelleft='off')

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
        for strat in np.unique(lattice):
            raw = np.where(lattice==strat)
            res = set([idx for idx in zip(*raw)])
            counts[strat].update(res) #= counts[strat].union(res)
    counts = dict(counts)

    sites = []
    df_tmp = {'strategy': [], 'coordinate': []}
    for strat, coords in counts.items():
        sites.append(len(coords))

        for co in coords:
            df_tmp['strategy'].append(strat)
            df_tmp['coordinate'].append(co)

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

    return pd.DataFrame(df_tmp)

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
    """ Figure 3 of paper and more
    """
    print('Computing waiting times')
    result = {'p': [], 'alpha': [], 'durations': []}
    for data in all_data:
        N = data['config']['N']
        p = data['config']['p']
        alpha = data['config']['alpha']
        print(f'p = {p}, alpha = {alpha}')

        # find dominant strategy at each point in time
        print(' > Finding dominant strategies')
        dom_strats = np.asarray(list(map(lambda e: get_dominant_strategy(e), data['snapshots'])))
        print(f'  >> Found {np.unique(dom_strats).size} unique strategies')

        if np.unique(dom_strats).size <= 1:
            print(' >> Skipping')
            continue

        # detect dominant strategy changes (and durations)
        print(' > Computing durations')
        durations = get_domain_durations(dom_strats)
        durations /= N**2
        print(f'  >> Found {durations.size} durations')

        # store result
        result['p'].extend([p]*len(durations))
        result['alpha'].extend([alpha]*len(durations))
        result['durations'].extend(durations)

    df = pd.DataFrame(result)

    # plot w-time distributions
    print(' > Plotting')
    for p in df['p'].unique():
        sub = df[df['p']==p]

        plt.figure()
        for alpha, group in sub.groupby(['alpha']):
            sns.distplot(
                group['durations'],
                kde=False, label=rf'$\alpha={alpha}$')

        plt.title(rf'Distribution of waiting times ($p={p}$)')
        plt.xlabel(r'$\Delta t$')
        plt.ylabel(r'count')
        plt.legend(loc='best')

        plt.savefig(f'images/waiting_times_p{p}.pdf')

    ## plot wtd dependence on parameters
    plt.figure()
    sns.boxplot(x='alpha', y='durations', hue='p', data=df)
    plt.savefig('images/waiting_times_vs_alpha.pdf')

    return df

def dominant_states(data, fname_app=''):
    """ Figure 2 of paper
    """
    N = data['config']['N']

    # compute statistics
    ts = []
    dom_strats = []
    for t, lattice in tqdm(zip(data['snapshot_times'], data['snapshots']), total=len(data['snapshot_times'])):
        bins = np.bincount(lattice.ravel().astype(int))
        ts.append(t / N**2)
        dom_strats.append(np.max(bins) / N**2)

    # plot
    plt.figure()

    plt.plot(ts, dom_strats)

    plt.title('Strategy cluster sizes')
    plt.xlabel(r'$t$')
    plt.ylabel('relative cluster size')

    plt.savefig(f'images/dominant_states{fname_app}.pdf')

    return pd.DataFrame({'time': ts, 'dominant strategy count': dom_strats})

def main(fnames):
    all_data = []
    for fname in fnames:
        with open(fname, 'r') as fd:
            data = json.load(fd)
            data['snapshots'] = [np.reshape(s, (data['config']['N'], data['config']['N'])) for s in data['snapshots']]

        all_data.append(data)
        print(f'[{fname}] Parsing {len(data["snapshots"])} entries')

        # compute and plot results
        f_app = os.path.basename(fname)
        plot_graph(data['graph'], fname_app=f'_{f_app}')
        overview_plot(data, fname_app=f'_{f_app}')

        df_sd = site_distribution(data, fname_app=f'_{f_app}')
        df_ds = dominant_states(data, fname_app=f'_{f_app}')

        # cache data
        df_sd.to_csv(f'cache/site_distribution_{f_app}.csv')
        df_ds.to_csv(f'cache/dominant_states_{f_app}.csv')

    df_wt = waiting_times(all_data)
    df_wt.to_csv(f'cache/waiting_times_{f_app}.csv')

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <data file>...')
        exit(-1)

    main(sys.argv[1:])
