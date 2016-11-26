"""
More elaborate plots based on the data
"""

import sys
import pickle
import collections

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def site_distribution(data):
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

    plt.loglog(binning, 'o', label=r'$\alpha={}$'.format(alpha))
    plt.loglog(sites, scale, label=r'$s^{-2.5}$')

    plt.xlabel(r'$s$')
    plt.ylabel(r'$n$')
    plt.legend(loc='best')

    plt.savefig('images/site_distribution.pdf')

def main(fname):
    with open(fname, 'rb') as fd:
        data = pickle.load(fd)

    site_distribution(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <data file>'.format(sys.argv[0]))
        exit(-1)

    main(sys.argv[1])
