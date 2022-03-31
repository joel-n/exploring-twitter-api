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
                log_=False, t=None, y=None, y2=None) -> None:
    """Plot data x and y in a histogram and save
    to the provided path. Optionally plots a line
    (or two) over the histogram.
    
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
        - y2: y-values for second overlaid line
        
    Returns:
        - n: values of the histogram bins
        - bs: bins generated or given
    """
    plt.figure(figsize=(8,8), clear=True)
    n, bs, _ = plt.hist(x, bins,color=['violet', 'palegreen'], log=log_,
                        histtype='bar', stacked=True, label=['replies', 'retweets']) 
    plt.title(title, size=24)
    plt.xlabel(xlab, size=18)
    plt.ylabel(ylab, size=18)
    plt.legend(fontsize=16)
    if t is not None and y is not None:
        if y2 is not None:
            plt.plot(t, y, color='red')
            plt.plot(t, y2, '--', color='midnightblue')
        else:
            plt.plot(t, y, color='midnightblue')
            
    plt.savefig(path) 
    plt.close('all')
    return n, bs


def create_hist_scatter(x, bins, path: str, title='', xlab='', ylab='',
                        log_=False, t=None, y=None,
                        scatter_y=None, root_flw=None) -> None:
    """Plot data x and y in a histogram and save
    to the provided path. Optionally plots a line
    over the histogram.
    
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
    n, bs, _ = plt.hist(x, bins, color='palegreen', log=log_)
    plt.title(title, size=24)
    plt.xlabel(xlab, size=18)
    plt.ylabel(ylab, size=18)
    
    if t is not None and y is not None:
        plt.plot(t, y, color='red') # 'midnightblue', 'lightpink'
    
    if scatter_y is not None:
        ax2 = ax.twinx()
        ax2.scatter(x, scatter_y, s=15, alpha=0.5, color='teal')
        ax2.scatter([0],[root_flw], s=20, color='red')
        ax2.set_ylabel('follower count')
    
    plt.savefig(path) 
    plt.close('all')
    return n, bs


def create_loglog_hist(x, n_bins, path: str, title='', xlab='', ylab='') -> None:
    """Plot data x and y in a histogram in log-log scale
    and save to the provided path.
    
    Args:
        - x: data for x axis (list or array)
        - n_bins: integer that specifies the number of bins
        - path: path to save figure to
        - title: figure title
        - xlab: x-axis label
        - ylab: y-axis label
        
    No return value.    
    """
    
    bins_ = np.concatenate((np.zeros(1), np.logspace(0, np.log10(max(x) + 1), num=n_bins, endpoint=True, base=10.0, dtype=None, axis=0)))
    plt.figure(figsize=(8,8), clear=True)
    plt.hist(x, bins_, color='palegreen', log=True)
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
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlab+' (log scale)', size=18)
        plt.ylabel(ylab+' (log scale)', size=18)
    else:
        plt.xlabel(xlab, size=18)
        plt.ylabel(ylab, size=18)
    plt.savefig(path)
    plt.close('all')
    return
