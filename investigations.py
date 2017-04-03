"""
Conduct further experiments
"""

import os

import numpy as np
import pandas as pd

import seaborn as sns


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
        #return data[abs(data - np.mean(data)) < m * np.std(data)]
        return data[data<1e3]

    # histogram without binning
    binning = np.bincount(data)
    nonzero_xvals = np.where(binning>0)[0]
    nonzero_xvals = reject_outliers(nonzero_xvals) # throw away "special entries"
    nonzero_binning = binning[nonzero_xvals]

    # histogram with binning
    data = reject_outliers(np.asarray(data))
    bins = np.logspace(0, np.log10(max(data)), num=20)
    hist, bin_edges = np.histogram(data, bins)
    hist = hist.astype(float)
    hist /= np.diff(bin_edges)
    nonzero_idx = np.where(hist>0)[0]
    deg_mids = np.array([bin_edges[i] for i in range(len(bin_edges)-1)])

    # linear fit to loglog data
    #fit = np.polyfit(np.log10(nonzero_xvals), np.log10(nonzero_binning), 1)
    fit = np.polyfit(np.log10(deg_mids[nonzero_idx]), np.log10(hist[nonzero_idx]), 1)

    if fname is not None:
        pol_obj = np.poly1d(fit)
        x_vals = np.arange(1, binning.shape[0]+1)

        sns.plt.figure()

        sns.plt.plot(
            np.where(binning>0)[0],
            binning[binning>0],
            'o', zorder=-1, label='site dist')
        sns.plt.plot(
            x_vals,
            np.array(x_vals)**-2.5,
            label=r'$s^{-2.5}$')
        sns.plt.plot(
            nonzero_xvals,
            10**pol_obj(np.log10(nonzero_xvals)),
            zorder=-2, label='fit')
        sns.plt.plot(
            deg_mids[nonzero_idx],
            hist[nonzero_idx],
            '.-', zorder=2, label='binned data')
        sns.plt.scatter(
            nonzero_xvals,
            nonzero_binning,
            marker='*', color='yellow', s=10,
            zorder=1, label='considered for binning')

        sns.plt.xscale('log')
        sns.plt.yscale('log')

        sns.plt.title(rf'Slope: ${fit[0]:.3}$')
        sns.plt.legend(loc='best')

        #sns.plt.show()
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
    for e in data:
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
