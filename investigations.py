"""
Conduct further experiments
"""

import os

import numpy as np
import pandas as pd

import seaborn as sns
from tqdm import tqdm


def read_data(prefix):
    """ Read cached data
    """
    data = []
    for entry in os.scandir('cache'):
        if not entry.name.startswith(prefix):
            continue

        print(f'Parsing {entry.name}')
        typ1, typ2, N, p, alpha_rest = entry.name.split('_')

        N = int(N[1:])
        p = float(p)
        alpha = float('.'.join(alpha_rest.split('.')[:-2]))
        df = pd.read_csv(entry.path, index_col=0)

        data.append({
            'config': {
                'N': N,
                'p': p,
                'alpha': alpha,
            },
            'df': df
        })
    return data

def fit_slope(data, fname=None):
    """ Fit loglog slope to given data
    """
    def reject_outliers(data, m=2):
        """ Throw away "special entries"
        """
        #return data[abs(data - np.mean(data)) < m * np.std(data)]
        return data[data<1e3]

    # histogram without binning
    hist_nobin = np.bincount(data)
    hist_nobin_x = np.where(hist_nobin>0)[0]
    hist_nobin_x_fitdata = reject_outliers(hist_nobin_x)

    # histogram with binning
    data = reject_outliers(np.asarray(data))
    bins = np.logspace(0, np.log10(max(data)), num=20)

    hist_bin, bin_edges = np.histogram(data, bins)
    hist_bin = hist_bin.astype(float)
    hist_bin /= np.diff(bin_edges)

    hist_bin_x = np.where(hist_bin>0)[0]
    deg_mids = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])

    # linear fit to loglog data
    fit = np.polyfit(np.log10(deg_mids[hist_bin_x]), np.log10(hist_bin[hist_bin_x]), 1)

    if fname is not None:
        pol_obj = np.poly1d(fit)

        sns.plt.figure()

        sns.plt.plot(
            hist_nobin_x,
            hist_nobin[hist_nobin_x],
            'o', zorder=-1, label='site dist')
        sns.plt.plot(
            hist_nobin_x,
            np.array(hist_nobin_x)**-2.5,
            label=r'$s^{-2.5}$')
        sns.plt.plot(
            deg_mids[hist_bin_x],
            10**pol_obj(np.log10(deg_mids[hist_bin_x])),
            zorder=-2, label='fit')
        sns.plt.plot(
            deg_mids[hist_bin_x],
            hist_bin[hist_bin_x],
            '.-', zorder=2, label='binned data')
        sns.plt.scatter(
            hist_nobin_x_fitdata,
            hist_nobin[hist_nobin_x_fitdata],
            marker='*', color='yellow', s=10,
            zorder=1, label='considered for binning')

        sns.plt.xscale('log')
        sns.plt.yscale('log')

        sns.plt.title(rf'Slope: ${fit[0]:.3}$')
        sns.plt.legend(loc='best')

        sns.plt.savefig(fname)
        sns.plt.close()

    return fit[0]

def site_distribution_slope():
    """ Check how slope varies
    """
    # read data
    data = read_data('site_distribution')

    # compute slopes
    result = {'p': [], 'alpha': [], 'slope': []}
    for e in tqdm(data):
        cur = e['df']

        count_dist = cur.groupby('strategy').count()['coordinate'].tolist()
        slope = fit_slope(count_dist, fname=f'images/site_distribution_fit_N{e["config"]["N"]}_p{e["config"]["p"]}_a{e["config"]["alpha"]}.pdf')

        result['p'].append(e['config']['p'])
        result['alpha'].append(e['config']['alpha'])
        result['slope'].append(slope)
    df = pd.DataFrame(result)

    # plot result
    sns.plt.figure()
    sns.barplot(x='alpha', y='slope', hue='p', data=df)
    sns.plt.savefig('images/site_distribution_slope.pdf')

def main():
    site_distribution_slope()

if __name__ == '__main__':
    sns.set_style('white')
    sns.plt.style.use('seaborn-poster')

    main()
