import numpy as np
import matplotlib.pyplot as plt

def create_plot(x, y, path: str, format_='-', title='', xlab='', ylab='') -> None:
    """Plot data x and y in a scatter plot and save
    to the provided path.
    
    Args:
        - x: data for x axis (list or array)
        - y: data for y axis (list or array)
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        
    No return value.    
    """
    plt.figure(figsize=(8,8), clear=True)
    if y is None:
        plt.plot(x, format_, color='dodgerblue')
    else:
        plt.plot(x, y, format_, color='dodgerblue')
    plt.title(title, size=24)
    plt.xlabel(xlab, size=18)
    plt.ylabel(ylab, size=18)
    plt.savefig(path)
    plt.close('all')
    return


def create_hist(x, bins, path: str, title='', xlab='', ylab='',
                log_=False, t=None, y=None, y2=None, stacked=True) -> None:
    """Plot tweet data x in a histogram and save
    to the provided path. Optionally plots a line
    (or two) over the histogram. Returns the histogram
    as a tuple of bin values and bins.
    
    Args:
        - x: data for x axis (list of three arrays with reply, retweet
        and quote timestamps in the same scale as the bins argument).
        - bins: integer or list/array that specifies the number of
        bins to use or the bins limits.
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        - t: x-values of overlaid line
        - y: y-values of overlaid line (first order system)
        - y2: y-values for second overlaid line (second order system)
        - stacked: plots histograms on top of each other when True
        
    Returns:
        - n: values of the histogram bins
        - bs: bins generated or given
    """
    
    f, ax = plt.subplots(figsize=(8,8), clear=True)
    if title != '':
        plt.title(title, size=24)
    n, bs, _ = ax.hist(x, bins,color=['violet', 'palegreen','cornflowerblue'], log=log_,
                        histtype='bar', stacked=stacked, label=['replies', 'retweets', 'quotes'])
    if t is not None and y is not None:
            if y2 is not None:
                ax.plot(t, y, color='red', label='first order system')
                ax.plot(t, y2, '--', color='midnightblue', label='second order system')
            else:
                ax.plot(t, y, color='midnightblue')
    plt.xlabel(xlab, fontsize=16)
    plt.ylabel(ylab, fontsize=16)
    ax.legend(fontsize=16)
    plt.savefig(path)
    plt.close('all')
    return n, bs


def create_hist_scatter(x, bins, path: str, title='', xlab='', ylab='', log_=False, t=None,
                        y=None, scatter_y=None, root_flw=None, peaks_x=None, peaks_y=None,
                        max_flw=None, max_t=None, threshold=None) -> None:
    """Plot tweet data x (replies, retweets and quotes) in
    a histogram and save to the provided path. Optionally
    plots a line over the histogram, and the followers of
    the users interacting.
    
    Args:
        - x: data for x axis (list or array)
        - bins: integer or list/array that specifies the number of
        bins to use or the bins limits.
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        - t: x-values of overlaid line
        - y: y-values of overlaid line
        
    Returns:
        - n: values of the histogram bins
        - bs: bins generated or given
    """
    f, ax = plt.subplots(figsize=(8,8), clear=True)
    cl = ['violet', 'palegreen','cornflowerblue'] if len(x) == 3 else ['violet', 'palegreen']
    n, bs, _ = plt.hist(x, bins, color=cl, log=log_,
                        histtype='bar', stacked=True,
                        label=['replies', 'retweets', 'quotes'])
    plt.title(title, size=24)
    plt.xlabel(xlab, size=16)
    plt.ylabel(ylab, size=16)
    
    if t is not None and y is not None:
        plt.plot(t, y, color='red') # 'midnightblue', 'lightpink'

    if not peaks_x is None:
        plt.scatter(peaks_x, peaks_y, label='detected peaks', s=55, marker='X', color='sienna')
    
    if scatter_y is not None:
        ax2 = ax.twinx()
        ax2.scatter(x[0], scatter_y[0], s=10, alpha=0.7, color='black', marker='x', label='reply followers')
        ax2.scatter(x[1], scatter_y[1], s=10, alpha=0.5, color='red', marker='^', label='retweet followers')
        ax2.scatter(x[2], scatter_y[2], s=10, alpha=0.3, color='seagreen', marker='*', label='quote followers')
        ax2.scatter([0],[root_flw], s=20, color='red')
        ax2.set_ylabel('follower count', rotation=270, fontsize=12)

        if not max_flw is None:
            print('flws:', max_flw)
            ax2.scatter(max_t, max_flw, label='outliers', s=55, facecolor='none', edgecolor='deeppink')
            if threshold:
                ax2.plot([0,bins[-1]], [threshold, threshold], label='threshold', color='deeppink')

    plt.legend(fontsize=14)    
    plt.savefig(path) 
    plt.close('all')
    return n, bs


def create_loglog_hist(x, n_bins, path: str, title='', xlab='', ylab='', density_=True) -> None:
    """Plot data x in a histogram in log-log scale
    and save to the provided path.
    
    Args:
        - x: data for x axis (list or array)
        - n_bins: integer that specifies the number of bins
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        - density: plots density of bins when set to
        True (recommended since bins are not equally
        spaced), else uses raw counts.
        
    No return value.    
    """
    assert min(x) > 0, 'data cannot have negative values when plotting loglog scale'
    bins_ = np.concatenate((np.zeros(1), np.logspace(0, np.log10(max(x) + 1), num=n_bins, endpoint=True, base=10.0, dtype=None, axis=0)))
    plt.figure(figsize=(8,8), clear=True)
    plt.hist(x, bins_, color='palegreen', log=True, density=density_)
    plt.title(title, size=24)
    plt.xlabel(xlab+' (log scale)', size=18)
    plt.ylabel(ylab+' (log scale)', size=18)
    plt.xscale('log')
    plt.savefig(path)
    plt.close('all')
    return


def create_ccdf(data, path: str, title='', xlab='', ylab='', loglog=False) -> None:
    """Plot the complementary cumulative density
    function (optionally in log-log scale).
    
    Args:
        - data: data to plot
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        - loglog: boolean indicating whether
        the ccdf is plotted in loglog-scale
    
    """
    sorted_x = np.sort(data)
    n = len(data)
    ccdf = [1 - i/n for i in range(1, n+1)]
    
    plt.figure(figsize=(8,8), clear=True)
    plt.plot(sorted_x, ccdf, color='palegreen')
    plt.title(title, size=24)
    if loglog:
        assert sorted_x[0] > 0, 'data cannot have negative values when plotting loglog scale'
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlab+' (log scale)', size=18)
        plt.ylabel(ylab+' (log scale)', size=18)
    else:
        plt.xlabel(xlab, size=18)
        plt.ylabel(ylab, size=18)
        plt.ylim(-0.1,1.1)
    plt.savefig(path)
    plt.close('all')
    return
