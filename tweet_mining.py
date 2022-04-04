import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
import time as pytime
import networkx as nx
from file_manip import * # file I/O
from plot_tools import * # plotting functions
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# import seaborn as sns # plotting tools


def print_dict(d: dict, indent: str) -> None:
    """Prints the keys of a dictionary to visualize the
    content of e.g., attributed of a returned tweet.
    
    Not all tweet objects contain the same attributed.
    
    """
    if isinstance(d, dict):
        for key in d:
            print(indent, key)
            print_dict(d[key], indent+'-')
    elif isinstance(d, list):
        for l in d:
            print(indent+' list: [')
            if isinstance(l, dict):
                print_dict(l, indent+'\t-')
            print(indent+'\t]')
    if indent == ' ':
        print(" ")
        
    return
 

def remove_missing_rt_files() -> None:
    """Checks for retweet files that should contain something,
    but are empty. Puts the missing files in another folder.
    
    No args or return value. Results are logged.
    """
    
    for f in os.listdir('retweets'):
        c_id = f.split('.')[0]
        size = os.stat(f'retweets/{f}').st_size
        root = read_file(f'root_tweets/{c_id}_root.jsonl')[0]
        n_rts = root['public_metrics']['retweet_count']

        if size == 0 and n_rts != 0:
            os.rename(f'retweets/{f}', f'missing_rts/{f}')
            logging.info(f'moved retweets/{f} to missing_rts/{f}')
        else:
            logging.info(f'retweets/{f} of size {size} (with {n_rts} retweets) not moved')


def compute_relative_time(t0: str, t: str) -> float:
    """Returns the time offset in hours of a time point t
    relative to a reference t0.
    
    Args:
        - t0: reference time point (in format %Y-%m-%dT%H:%M:%S.000Z)
        - t: time point of interest (in format %Y-%m-%dT%H:%M:%S.000Z)
    
    Returns:
        - dt: time offset in hours
    """
    t0 = datetime.datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S.000Z')
    t = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.000Z')
    
    # timedelta objects have attributes days, seconds and microseconds
    t_delta = t-t0
    dt = (86400*t_delta.days + t_delta.seconds)/3600
    return dt


def auto_correlate(x, symmetric=True):
    """Returns the autocorrelation of x: ac[x,k] = sum_n x[n+k]x[n]
    
    Args:
        - x: series to correlate
        - symmetric: boolean indicating whether or not to return
        the whole (symmetric) result, or just the right half.
        
    Returns:
        - ac: The result of the auto correlation
    
    """
    ac = np.correlate(x, x, 'full')
    if not symmetric:
        ac = ac[len(x)-1:]
        ac = ac/ac[0]
    else:
        ac = ac/ac[len(x)-1]
    return ac


def mean_square_error_fo(bin_values, beta: float, lambda_: float) -> float:
    """Computes the mean square error of a first order
    linear model with parameters beta and lambda_, with
    data discretized in steps of 1 unit of time.
    
    Args:
        - bin_values: bin values of the engagement histogram
        - beta: parameter modelling the initial response to
        the tweet. Should be positive
        - lambda_: decay constant which should take values
        less than 1.
        
    Returns:
        - MSE: the mean square error
    """
    x_hat = beta
    MSE = 0
    for x in bin_values:
        MSE += (x-x_hat)**2
        x_hat *= (1-lambda_)
    MSE = MSE / len(bin_values)
    return MSE


def eval_error(time, model_engagement, true_engagement):
    """Computes the mean square error, and the ratio between the sum of
    squared residuals and the sum of the squared signal in continuous time
    
    Args:
        - time: time vector for the observations
        - true_engagement: engagement at times
        specified in time vector
        - lambda_: (optimal) decay constant
        - beta_: parameter modelling the initial
        response to the tweet
        
    Returns:
        - MSE: mean square error
        - RSS_frac
        - model_engagement: 
    """
    res = model_engagement - true_engagement
    sq_residual = np.square(res)
    sq_signal = np.square(true_engagement)
    
    MSE = np.mean(sq_residual)
    RSS_frac = np.sum(sq_residual)/np.sum(sq_signal)
    
    #plt.hist(res, bins=200)
    #plt.show()
    
    return MSE, RSS_frac


def exponential(x_, lambda_, beta_):
    """Exponential decay with parameters lambda and beta."""
    return beta_*np.exp(-lambda_*x_)


def power(x_, lambda_, beta_):
    """Power law."""
    return beta_*(np.power(np.array(x_, dtype=float), -lambda_))


def estimate_decay_parameters(time, engagement):
    """Returns an estimate of the parameters of the
    exponential decay function of engagement over time.
    
    Args:
        - time: time vector
        - engagement: engagement at times specified
        in time vector
        
    Returns:
        - popt: optimal parameters
    """
    # lower and upper bounds on lambda and beta:
    # 1e-4 <= lambda <= 1e3, and 1 <= beta <= 1e9.
    bounds_ = ([1e-4, 1], [1e3, 1e9])
    init = [1, 1]
    
    # Optimization: Trust Region Reflective algorithm 'trf'
    # Levenberg-Marquardt ('lm') does not handle parameter bounds
    method_ = 'trf'
    
    popt, _ = curve_fit(exponential, time, engagement, p0=init,
                     bounds=bounds_, method=method_)
    return popt


def delayed_biexponential(x_, alpha, beta, gamma, eta, sigma, delay):
    """Solution to second order system. Assumes that the time bins
    are spaced 1 hour apart (tau_d is the delay given in hours).
    
    Note that gamma and alpha are positive, whereas they are negative
    in the original equation system.
    """
    #tau_dn = int(tau_d*100)
    #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
    #return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 
    delay=int(delay)
    delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
    delayed[:delay] = 0
    return eta*np.exp(-alpha*x_) + delayed


def biexponential(x_, alpha, beta, gamma, rho):
    """Solution to system
    dx1/dt = alpha*x1(t) + beta*x2(t)
    dx2/dt = gamma*x2(t) + rho*d(t)
    """
    return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))


def estimate_biexponential(time, engagement, loss_='linear'):
    
    def biexponential_opt(x_, alpha, beta, d_gamma, rho):
        """Solution to system
        dx1/dt = alpha*x1(t) + beta*x2(t)
        dx2/dt = gamma*x2(t) + rho*d(t)
        """
        gamma = alpha + d_gamma
        return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))

    method_ = 'trf'
    bounds_ = ([1e-5, 1e-2, 1e-5, 1e-3],
               [1e3,  1e9,  1e3,  1e9])
    init = [1e-1, 1, 1e-1, 1]
    popt, _ = curve_fit(biexponential_opt, time, engagement,
                        p0=init, bounds=bounds_, method=method_, loss=loss_)
    a, b, d_g, r = popt[0], popt[1], popt[2], popt[3]
    g = a + d_g
    return a, b, g, r


def estimate_second_order(time, engagement, delay, loss_='linear'):
    """Fit a second order model of type
    dx1/dt = ax1 + bx2 + nd(t) + sd(t-td)
    dx2/dt = gx2 +     + rd(t-td)
    Since r and b are coupled we set r=1
    
    Args:
        - time: time vector
        - engagement: engagement at times specified
        in time vector
        
    Returns:
        - a, b, g, n, s: model parameters
    """
    
    def biexponential_free_delay(x_, d_alpha, beta, gamma, eta, sigma, delay):
        """Solution to second order system. Assumes that the time bins
        are spaced 1 hour apart (tau_d is the delay given in hours).

        Note that gamma and alpha are positive, whereas they are negative
        in the original equation system.
        """
        #tau_dn = int(tau_d*100)
        #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
        #return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 
        delay_n=int(delay)
        alpha = gamma + d_alpha
        delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
        delayed[:delay_n] = 0
        return eta*np.exp(-alpha*x_) + delayed
    
    def biexponential_fixed_delay(x_, d_alpha, beta, gamma, eta, sigma):
        """Solution to second order system. Assumes that the time bins
        are spaced 1 hour apart (delay is given by encapsulating function hours).

        Note that gamma and alpha are positive, whereas they are negative
        in the original equation system.
        """
        #tau_dn = int(tau_d*100)
        #u_delay = np.concatenate((np.zeros(tau_dn), np.ones(len(x_)-tau_dn)))
        alpha = gamma + d_alpha
        delayed = (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-delay)) - (beta/(gamma-alpha))*np.exp(-gamma*(x_-delay))
        delayed[:delay] = 0
        return eta*np.exp(-alpha*x_) + delayed
        # return eta*np.exp(-alpha*x_) + (sigma+(beta/(gamma-alpha)))*np.exp(-alpha*(x_-tau_d))*u_delay - (beta/(gamma-alpha))*np.exp(-gamma*(x_-tau_d))*u_delay 

    method_ = 'trf'
    if delay != 0:
        bounds_ = ([1e-5, 1e-2, 1e-5, 1,   1e-5],
                   [1e3, 1e9,   1e3,  1e9, 1e6])
        init = [1e-1, 1, 1e-1, 2, 1]
        popt, _ = curve_fit(biexponential_fixed_delay, time, engagement,
                            p0=init, bounds=bounds_, method=method_, loss=loss_)
        d_a, b, g, n, s, td = popt[0], popt[1], popt[2], popt[3], popt[4], 0
        
    else:
        tau_max = int(time[-1]-time[0])
        # instead of fitting gamma, use auxillary parameter c=alpha-gamma, which is
        # always less than 0 (if alpha and gamma are defined negative) and describes the
        # discrepancy between the types of decay. Using gamma < alpha < 0 will give an error.
        bounds_ = ([1e-5, 1e-2, 1e-5, 1, 1e-5, 0],
                   [1e3, 1e9,   1e3,  1e9, 1e6, tau_max])
        init = [1e-1, 1, 1e-1, 2, 1, int(0.5*tau_max)]

        # Linear fit (alt. L1 fit with 'soft_l1')
        popt, _ = curve_fit(biexponential_free_delay, time, engagement,
                            p0=init, bounds=bounds_, method=method_, loss=loss_)
        d_a, b, g, n, s, td = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    
    a = g + d_a
    return a, b, g, n, s, td


def peak_detection(e, lag, threshold, influence):
    """Peak detection algorithm using deviation from moving median to
    find peaks in histograms.
    
    From: J.P.G. van Brakel. Robust peak detection algorithm (using z-scores), 2019.
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtimetimeseries-data.
    
    Args:
        - e: engagement histogram values
        - lags: size of the moving window
        - threshold: number of standard deviations required to trigger detection
        - influence: influence of the next sample to the mean
    
    Returns:
        - A dictionary that contains the peak detection signal: with elements
        in {1-,0,1}, indicating a negativ peak, no peak, or positive peak; the
        moving median, and the moving standard deviation.
    """
    
    signals = np.zeros(len(e))
    filteredE = np.array(e)
    avgFilter = [0]*len(e)
    stdFilter = [0]*len(e)
    avgFilter[lag - 1] = np.mean(e[0:lag])
    stdFilter[lag - 1] = np.std(e[0:lag])
    
    # For each data point
    for i in range(lag, len(e) - 1):
        
        # Detect a deviation from the median +/- threshold*std
        if abs(e[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if e[i] > avgFilter[i-1]:
                signals[i] = 1 # positive peak
            else:
                signals[i] = -1 # negative peak

            # Less influence from peak data
            filteredE[i] = influence * e[i] + (1 - influence) * filteredE[i-1]
        else:
            # No peak detected, no less influence
            signals[i] = 0
            filteredE[i] = e[i]
        
        avgFilter[i] = np.mean(filteredE[(i-lag):i])
        stdFilter[i] = np.std(filteredE[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter),
                lag=lag,
                threshold=threshold,
                influence=influence)


def filter_peaks(result, y, peak_threshold: float, adj_dist=1):
    """Filters out the desired peaks from the result of the peak
    detection algorithm. Returns the peak coordinates.
    
    Args:
        - result: the result dictionary of the peak detection algorithm
        - y: engagement histogram values
        - peak_threshold: fraction of global maximum a peak must attain
        in order not to be filtered out
        - adj_dist: max distance between adjacent peaks; these will be
        reduced to the highest peak
        
    Returns:
        - peaks_x: x-coordinate for the relevant peaks
        - peaks_y: y-coordinate for the relevant peaks
    """
    
    xx, yy= [], []
    
    assert peak_threshold >= 0 and peak_threshold <= 1
    glob_max = max(y)
    
    # Filter out peaks below the global threshold
    for i,r in enumerate(result["signals"]):
        if r==1 and y[i] > peak_threshold*glob_max:
            xx.append(i+1)
            yy.append(y[i])

    #print(xx, yy)
    peaks_x, peaks_y = [], []
    
    prev_x = xx[0]
    segment_x, segment_y = [xx[0]], [yy[0]]
    curr_max, curr_max_y = xx[0], yy[0]

    # Filter out adjacent peaks
    for i in range(1, len(xx)):
        if xx[i] > prev_x + adj_dist:
            #print(f'{xx[i]}!={prev_x+adj_dist}; peak = ({curr_max},{curr_max_y})')
            peaks_x.append(curr_max)
            peaks_y.append(curr_max_y)
            
            segment_y = [yy[i]]
            segment_x = [xx[i]]
            curr_max = xx[i]
            curr_max_y = yy[i]
        else:
            if curr_max_y < yy[i]:
                curr_max = xx[i]
                curr_max_y = yy[i]
            segment_x.append(xx[i])
            segment_y.append(yy[i])
            
        prev_x = xx[i]
            
    peaks_x.append(curr_max)
    peaks_y.append(curr_max_y)
    
    valleys = []
    for i in range(1,len(peaks_x)):
        if peaks_x[i] > peaks_x[i-1] + 3:
            valleys.append((peaks_x[i-1],peaks_x[i]))
    #print(valleys)
    #print(peaks_x, peaks_y)
    
    return peaks_x, peaks_y # xx, yy


def plot_peak_detection(y, result, conv_id):
    """Plots the peaks from the raw result of the detection algorithm.
    The result data is filtered before plotting.
    
    Args:
        - y: the engagement histogram values
        - result: result dictionary of the peak detection algorithm
        - conv_id: conversation id
        
    Returns:
        - type_: the type of engagement graph, first position
        indicates delay (1X) or no delay (0X), the second position
        indicates a single peak (0), double peak (2), bump (3), or
        multiple peaks (4).
        - first_peak: bin number of the first peak (i.e., delay).
        Chosen as the largest of the first two peaks, or -1 in
        the odd case that there is no peak.
    """
    
    peaks_x, peaks_y = filter_peaks(result, y, peak_threshold=0.15, adj_dist=2)
    
    # DECISION: We use the heuristic to pick the largest of the two first peaks
    type_ = 0
    if len(peaks_x) != 0:
        if peaks_x[0] <= result['lag']+1:
            type_ += 0 # NOP
        else:
            type_ += 10
        
        if len(peaks_x) == 1:
            first_peak = peaks_x[0] - result['lag'] - 1
            second_peak = -1
        else:
            if peaks_y[0] > peaks_y[1]:
                first_peak = peaks_x[0] - result['lag'] - 1
                second_peak = peaks_x[1] - result['lag'] - 1
            else:
                first_peak = peaks_x[1] - result['lag'] - 1
                if len(peaks_x) > 2:
                    second_peak = peaks_x[2] - result['lag'] - 1
                else:
                    second_peak = -1
        
        if (len(peaks_x) == 2) and (min(peaks_y[0], peaks_y[1]) > 0.45*max(peaks_y[0], peaks_y[1])):
            type_ += 1
        elif len(peaks_x) == 2:
            type_ += 2
        elif len(peaks_x) > 2:
            type_ += 3
    else:
        first_peak = -1
        second_peak = -1
        type_ = 4
    
    
    plt.figure(figsize=(8,8), clear=True)
    #plt.subplot(211)
    plt.plot(np.arange(1, len(y)+1), y, color='navy', lw=2)

    plt.plot(np.arange(1, len(y)+1),
               result["avgFilter"], '--', color='gold', lw=1)

    plt.plot(peaks_x, peaks_y, 'o', color='red')

    plt.plot(np.arange(1, len(y)+1),
               result["avgFilter"] + result['threshold'] * result["stdFilter"], color="sienna", lw=0.8)

    plt.plot(np.arange(1, len(y)+1),
               result["avgFilter"] - result['threshold'] * result["stdFilter"], color="sienna", lw=0.8)

    #plt.subplot(212)
    suppressed = [i for i,r in enumerate(result['signals']) if r > 0]
    supp_y = -1*np.ones(len(suppressed))
    plt.plot(suppressed, supp_y, 'o', color='salmon')
    #plt.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
    #plt.ylim(-1.5, 1.5)
    plt.savefig(f'sampled_conversations_graphs/peak_detection/{type_}/{conv_id}_peaks2.png')
    plt.close('all')

    return type_, first_peak, second_peak


def get_file_paths(conv_id: str):
    """Returns the file paths to the conversation root, replies, and retweets
    for a given conversation ID.
    
    Args:
        - conv_id: conversation ID
        
    Returns:
        - root_path, conv_path, retw_path: file paths
    """
    
    root_path = f'root_tweets/{conv_id}_root.jsonl'
    conv_path = f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'
    retw_path = f'retweets/{conv_id}.jsonl'
    
    return root_path, conv_path, retw_path


def process_engagement_times(conv_ids: list[str], delta_sec: float) -> None:
    """Retrieves the engagement times of retweets and replies.
    Only considers conversations where root, conversation, and
    retweet files exist.
    
    Args:
        - conv_ids: list of conversation IDs
        - delta_sec: the time step length in seconds
        of the time series discretization in seconds
        
    No return value.
    """

    out_conv_id = []
    sufficient, missing_data, too_few, opt_failed = 0, 0, 0, 0
    
    # Iterate with generator from filter(all_files_exist, conv_ids)?
    for conv_id in conv_ids:
        root_path, conv_path, retw_path = get_file_paths(conv_id)
        
        # Check that root, conversation and retweet files exist.
        if os.path.isfile(root_path) and os.path.isfile(conv_path) and os.path.isfile(retw_path):
            root = read_file(root_path)[0]
            out_conv_id.append(conv_id)
        else:
            missing_data += 1
            continue
        
        """If plotting the number of followers of those who interact by retweeting
        skip = False
        for rt in file_generator(retw_path):
            if not 'public_metrics' in rt['author']:
                skip = True
            break
        if skip:
            missing_data += 1
            continue
        """
        
        engagement_time, followers = [], []
        rep_time, rt_time = [], []
        replying_users = set()
        
        for re in file_generator(conv_path):
            re_t = compute_relative_time(root['created_at'], re['created_at'])
            engagement_time.append(re_t)
            rep_time.append(re_t)
            replying_users.add(re['author_id'])
            #followers.append(re['author']['public_metrics']['followers_count'])
        
        n_replies = len(engagement_time)
        
        for rt in file_generator(retw_path): 
            rt_t = compute_relative_time(root['created_at'], rt['created_at'])
            engagement_time.append(rt_t)
            rt_time.append(rt_t)
            #followers.append(-1)
            #followers.append(rt['author']['public_metrics']['followers_count'])
        
        # TODO/DECISION: Ignore the conversations that have fewer than 50 replies/retweets,
        # or perhaps only engagament in the first X minutes
        tot_eng = len(engagement_time)
        n_api_retweets = root['public_metrics']['retweet_count']
        
        # if tot_eng < 50:
        if (tot_eng < 50) or ((tot_eng-n_replies) <= 1e-5*n_api_retweets):
            too_few += 1
            continue
        else:
            sufficient += 1
        
        
        reply_ratio = n_replies / tot_eng
        
        # Bin and estimate engagement curve (time series with increments of delta_t)
        mx = np.max(engagement_time)    # max time in hours
        delta_h = delta_sec/3600        # delta_t in hours
        n_bins = int((mx//delta_h) + 2) # Add 2 to compensate for integer division and endpoint in linspace
        bins = np.linspace(start=0, stop=n_bins*delta_h, num=n_bins, endpoint=False, retstep=False, dtype=None, axis=0)
        
        n, bs = np.histogram(engagement_time, bins)
        
        lag = 10
        threshold = 1.5
        influence = 0.8
        # rng = np.random.default_rng(123)
        rd = np.zeros(lag) # np.abs(rng.standard_normal(lag))
        time_series = np.concatenate((rd,n))#/max(n)
        result = peak_detection(time_series, lag=lag, threshold=threshold, influence=influence)
        type_, first_peak, second_peak = plot_peak_detection(time_series, result, conv_id, plot=False)
        
        root_followers = root['author']['public_metrics']['followers_count']
        n_api_replies = root['public_metrics']['reply_count']
        
        #acE = auto_correlate(n, symmetric=False)
        #create_plot(x=acE, y=None, path=f'sampled_conversations_graphs/engagement_correlation/{conv_id}_corr.png',
        #            format_='-', title='Autocorrelation of engagement', xlab=f'time (dt={delta_sec} seconds)', ylab='correlation')
        
        t = (bins[1:] + bins[:-1])/2 # Time vector (center of bins, can also use bins[:-1])
        
        try:
            # Take the delay into account (TODO: also in the MSE computation)
            #if type_ >= 10:
            if first_peak > 0:
                t = t[first_peak:] - delta_h*first_peak # = t[:-first_peak]?
                n = n[first_peak:]
            else:
                first_peak = 0
                

            """Fit only bins until 97% of engagement is reached, useful when using log loss
            cum_eng = np.cumsum(n)
            stop_ix = -1
            for i,e in enumerate(cum_eng):
                if e > cum_eng[-1]*0.97:
                    stop_ix = i
                    break
            
            n = n[:stop_ix]
            t = t[:stop_ix]
            """
            
            a, b, g, eta, s = 'None', 'None', 'None', 'None', 'None'
            lambda_, beta_ = 'None', 'None'
            a, b, g, rho = estimate_biexponential(time=t, engagement=n, loss_='linear')
            model_eng = biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            
            """
            if second_peak == -1:
                # Set delay to zero in second order system to get a biexponential solution
                #opt_params = estimate_decay_parameters(t, n, loss_='linear', f_scale_=1.0)
                #lambda_, beta_ = opt_params[0], opt_params[1]
                #lambda_, beta_ = estimate_lin_decay(t, n) # log loss
                
                a, b, g, eta, s, delay = estimate_second_order(time=t, engagement=n, delay=0, loss_='linear')
                model_eng = delayed_biexponential(t, alpha=a, beta=b, gamma=g, eta=eta, sigma=s, delay=delay)
                #model_eng = exponential(t, lambda_, beta_)
                #a, b, g, eta, s = 'None', 'None', 'None', 'None', 'None'
            else:
                delay = second_peak - first_peak
                a, b, g, eta, s, delay = estimate_second_order(time=t, engagement=n, delay=delay, loss_='linear')
                delay=int(delay)
                model_eng = delayed_biexponential(t, alpha=a, beta=b, gamma=g, eta=eta, sigma=s, delay=delay)
                lambda_, beta_ = 'None', 'None'
            """
            MSE, RSS_frac = eval_error(t, model_eng, n)
            
            result_file = 'parameter_estimations/estimations_delay_biexp-no-impulse-delay_L2.txt'
            write_text(file_name=result_file,
                       text=f'{conv_id},{root_followers},{len(replying_users)},{type_},{first_peak},{second_peak},{lambda_},{beta_},{a},{b},{g},{rho},{MSE},{RSS_frac},{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies}')
                       #text=f'{conv_id},{root_followers},{len(replying_users)},{type_},{first_peak},{second_peak},{lambda_},{beta_},{a},{b},{g},{eta},{s},{MSE},{RSS_frac},{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies}')
                       #text=f'{conv_id},{root_followers},{len(replying_users)},{type_},{first_peak},{lambda_},{beta_},{MSE},{RSS_frac},{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies}')
            
            graph_path_line = f'sampled_conversations_graphs/engagement_histograms/{conv_id}_repl_retw_ds{int(delta_sec)}_delay_stacked_biexp_L2.png'
            _, _ = create_hist([rep_time, rt_time], bins, path=graph_path_line,
                               #engagement_time, bins, path=graph_path_line,
                               title='Engagement over time (replies and RTs)',
                               xlab='time (h)', ylab='counts (replies and retweets)',
                               log_=False, overlay_line=True, t=t+delta_h*first_peak, y=model_eng)
            
            #_, _ = create_hist_scatter(engagement_time, bins, path=f'sampled_conversations_graphs/engagement_histograms/interactor_followers/{conv_id}_engag_ds{int(delta_sec)}_type3_flws.png',
            #                           title='Engagement over time (replies and RTs)', xlab='time (h)', ylab='counts (replies and retweets)', log_=False,
            #                           overlay_line=True, t=t+delta_h*first_peak, y=model_eng, scatter_y=followers, root_flw=root['author']['public_metrics']['followers_count'])
            
        except Exception as e:
            logging.warning(e)
            logging.info(f'optimization failed for conversation {conv_id}: replies:{n_replies}, reply_ratio:{reply_ratio}')
            sufficient -= 1
            opt_failed += 1
        
    print('Plots: {}, missing data: {}, too few data points: {}, optimization failed: {}'.format(sufficient, missing_data, too_few, opt_failed))
    
    return


def undirected_message_tree(conv_id: str) -> nx.Graph:
    """Returns an undirected graph (tree) that represents the
    messages in a conversation. The graph is empty if any of
    the root or conversation files are missing. The graph is
    not guaranteed to be connected.
    
    Args:
        - conv_id: a single conversation ID
        
    Returns:
        - G: a conversation graph
    """
    
    root_path = f'root_tweets/{conv_id}_root.jsonl'
    conv_path = f'sampled_conversations/{conv_id}_conversation-tweets.jsonl'
    
    G = nx.Graph()
    
    if not os.path.isfile(root_path) or not os.path.isfile(conv_path):
        return G
    
    for re in file_generator(conv_path):
        for ref_tw in re['referenced_tweets']:
            if ref_tw['type'] == 'replied_to':
                G.add_edge(re['id'], ref_tw['id'])
                break
    return G


def compute_radii(conv_ids: list[str]) -> list[int]:
    """Computes and saves the radii of conversations graphs,
    i.e., the maximum depth of the conversation tree. Ignores
    the conversations that fewer than 90% of nodes connected
    to the root. The result is stored in graph_radius.txt in
    the parameter_estimations folder.
    
    Args:
        - conv_ids: list of conversation IDs
        
    Returns:
        - rad: vector of conversation radii
    """
    rad = []
    connected = 0
    for c_id in conv_ids:
                        
        G = undirected_message_tree(c_id)
        
        if not G.has_node(c_id):
            continue
        
        n_nodes = len(list(G.nodes))
        # Allow for unconnected graphs if the component that contains
        # the root tweet has more than 90% of all nodes.
        if n_nodes > 0:
            if nx.is_connected(G):
                r = nx.radius(G)
                rad.append(r)
                write_text('parameter_estimations/graph_radius.txt', f'{c_id},{nx.radius(G)}')
                connected += 1
            else:
                nodes_largest_cc = nx.node_connected_component(G, c_id)
                if len(nodes_largest_cc) >= 0.9*n_nodes:
                    r = nx.radius(G.subgraph(nodes_largest_cc))
                    rad.append(r)
                    write_text('parameter_estimations/graph_radius.txt', f'{c_id},{r}')
                    
    print("{} conversations graphs were not sufficiently connected. Obtained results from {} conversations, of which {} were completely connected.".format(len(conv_ids)-len(rad), len(rad), connected))
    return rad


def retweet_metrics(conv_ids: list[str]) -> None:
    """Computes retweet metrics and print to a file.
    Plots retweets over time from the initial post.
    Saves images to 'sampled_conversation_graphs' folder.
    
    NOTE: Implicitly assumes that get_root_tweets() has
    been called through get_retweets_of() for these
    conversation IDs.
    
    TODO (suggestions): Save retweet times to files, and
    use this information in another function.
    
    Args:
        - conv_ids: list of conversation IDs (that have retweet
        files in the 'retweet' folder, and root files in the
        'root_tweet' folder).
        
    Metrics:
    - Retweets over time; histogram/pdf/cdf (per root)
    - Number of retweets; histogram (for all roots)
    - Retweets vs root followers; scatter (for all roots)
    - Final engagement time; histogram (for all roots)
    - Final engagement time v. followers; scatter (for all roots)
    - Final engagement time v. retweets; scatter (for all roots)
        
    No return value.
    """
    
    n_retweets = []
    root_followers = []
    final_rt_time = []
    
    for conv_id in conv_ids:
        root = read_file(f'root_tweets/{conv_id}_root.jsonl')[0]
        retweets = read_file(f'retweets/{conv_id}.jsonl')

        n_retweets.append(len(retweets))
        root_followers.append(root['author']['public_metrics']['followers_count'])
        
        engagement_time = []
        # Root is poster first, no further adjustments to time needed
        t0 = datetime.datetime.strptime(root['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
        for rt in retweets:
            time = datetime.datetime.strptime(rt['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
            dt = time-t0
            engagement_time.append((86400*dt.days + dt.seconds)/3600)

        engagement_time = np.sort(engagement_time)
        
        if(len(engagement_time)) > 0:
            final_rt_time.append(engagement_time[-1])
        else:
            final_rt_time.append(0)
            # DECISION: return if no retweets?
            # return
        
        
        create_hist(engagement_time, bins=50, path=f'sampled_conversations_graphs/{conv_id}_retweets.svg',
                title='Engagement times', xlab='time (h)', ylab='counts')

    create_plot(n_retweets, root_followers, format_='o', path='sampled_conversations_graphs/retweets_vs_followers.svg',
                title='Retweets as a function of followers', xlab='followers', ylab='retweets')
    create_plot(final_rt_time, root_followers, format_='o', path='sampled_conversations_graphs/followers_vs_final_time.svg',
                title='Final engagement time', xlab='final retweet time (h)', ylab='number of followers')
    create_plot(final_rt_time, n_retweets, format_='o', path='sampled_conversations_graphs/retweets_vs_final_time.svg',
                title='Final engagement time', xlab='final retweet time (h)', ylab='number of retweets')
    create_hist(final_rt_time, bins=50, path='sampled_conversations_graphs/final_time_distribution.svg',
                title='Final engagement time distribution', xlab='final retweet time (h)', ylab='counts')
    create_hist(n_retweets, bins=50, path='sampled_conversations_graphs/retweets_distribution.svg',
                title='Retweets distribution', xlab='retweets', ylab='counts')
    
    d = {'conv_id':conv_ids,
         'n_retweets':n_retweets,
         'root_followers':root_followers,
         'final_rt_time':final_rt_time}
    
    pd.DataFrame(data=d).to_csv('rt_data.csv',index=False)
    
    return


def reply_metrics(conv_ids: list[str]):
    """Computes reply metrics for a conversation and
    saves to folder.
    
    NOTE: Assumes that get_root_tweets() has been
    called for these conversation IDs. This is
    automatically fulfilled through calling
    get_retweets_of() with the conversation IDs
    as argument. get_root_tweets() also be called
    separately.
    
    Args:
        - conv_ids: list of conversation IDs to compute
        metrics for
        
    No return value.
    """

    n_authors_conv = []
    n_replies_conv = []
    n_deg_out_conv = []
    t_final_reply = []
    out_conv_id = []
    
    for conv_id in conv_ids: 
        
        root_path = f'root_tweets/{conv_id}_root.jsonl'
        if os.path.isfile(root_path):
            root = read_file(root_path)[0]
            out_conv_id.append(conv_id)
        else:
            continue

        conversation = read_file(f'sampled_conversations/{conv_id}')
        
        conv_dict, engagement_time, n_authors = get_conversation_dict(conversation, root)
        
        n_authors_conv.append(n_authors)
        n_replies_conv.append(len(conv_dict))
        t_final_reply.append(max(engagement_time))
        
        deg_out = 0
        for tweetID in conv_dict:
            # If root author replies to someone else [TODO: control for author]
            if conv_dict[tweetID]['author_id'] == root['author_id'] and conv_dict[tweetID]['referenced_author_id'] != root['author_id']:
                deg_out += 1
        
        n_deg_out_conv.append(deg_out)
        
    # plot figures
    t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    create_plot(n_authors_conv, n_replies_conv, format_='o',
                path=f'sampled_conversations_graphs/authors_replies_{t_query}.svg',
                title='Authors and replies', xlab='number of authors', ylab='number of replies')
    create_plot(n_authors_conv, t_final_reply, format_='o',
                path=f'sampled_conversations_graphs/authors_t_final_{t_query}.svg',
                title='Authors and time for last reply', xlab='number of authors',
                ylab='time of final reply (h)')
    create_hist(n_replies_conv, bins=50,
                path=f'sampled_conversations_graphs/replies_distribution_{t_query}.svg',
                title='Replies in conversations', xlab='number of replies', ylab='counts')
    create_hist(n_authors_conv, bins=50,
                path=f'sampled_conversations_graphs/authors_distribution_{t_query}.svg',
                title='Authors in conversations', xlab='number of authors', ylab='counts')
    
    return n_authors_conv, n_replies_conv, n_deg_out_conv, t_final_reply, out_conv_id
        

def get_conversation_dict(conv_tweets: list[dict], root: dict) -> tuple[dict, list, int]:
    """Returns a dictionary containing information on
    the tweets in the conversation along with a list
    of their engagement times in hours after tweet zero.
    Dictionaries in Python >3.7 are ordered.
    
    Args:
        - conv_tweets: a list with tweet .json objects, 
        obtained from e.g., read_file()
        
    Returns:
        - conv_dict: a dictionary mapping tweet ids to tweet info
        - engagement_time: list of times in hours
        - n_authors: number of authors participating in the conversation
    """
    author_set = set()
    conv_dict = {}
    time_stamps = []
    for tw in conv_tweets:
        ref_id = None
        ref_auth_id = None
        try:
            for ref_tw in tw['referenced_tweets']:
                if ref_tw['type'] == 'replied_to':
                    ref_id = ref_tw['id']
                    ref_auth_id = ref_tw['author_id']
                    break
        except Exception as e:
            logging.warning(e)

        author_set.add(tw['author_id'])
        conv_dict[tw['id']] = {'author_id':      tw['author_id'],
                               'author_name':    tw['author']['name'],
                               'time_stamp':     tw['created_at'],
                               'public_metrics': tw['public_metrics'],
                               'referenced_id':  ref_id,
                               'referenced_author_id': ref_auth_id}
                             # 'text_snippet':tw['text'][0:4],
        
        time_stamps.append(tw['created_at'])
        
    engagement_time = []
    t0 = datetime.datetime.strptime(root['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
    for ts in time_stamps:
        time = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.000Z')
        dt = time-t0
        engagement_time.append((86400*dt.days + dt.seconds)/3600)

    # Needed if we use the results of conv. query and not root tweet time
    #for i in range(len(engagement_time)):
    #    engagement_time[i] = engagement_time[i] - engagement_time[-1]
        
    n_authors = len(author_set)
        
    return conv_dict, engagement_time, n_authors
        

def get_conversation_dict(conv_tweets: list[dict], root: dict) -> tuple[dict, list, int]:
    """Returns a dictionary containing information on
    the tweets in the conversation along with a list
    of their engagement times in hours after tweet zero.
    Dictionaries in Python >3.7 are ordered.
    
    Args:
        - conv_tweets: a list with tweet .json objects, 
        obtained from e.g., read_file()
        
    Returns:
        - conv_dict: a dictionary mapping tweet ids to tweet info
        - engagement_time: list of times in hours
        - n_authors: number of authors participating in the conversation
    
    """
    author_set = set()
    conv_dict = {}
    time_stamps = []
    for tw in conv_tweets:
        ref_id = None
        ref_auth_id = None
        try:
            for ref_tw in tw['referenced_tweets']:
                if ref_tw['type'] == 'replied_to':
                    ref_id = ref_tw['id']
                    ref_auth_id = ref_tw['author_id']
                    break
        except Exception as e:
            logging.warning(e)

        author_set.add(tw['author_id'])
        conv_dict[tw['id']] = {'author_id':      tw['author_id'],
                               'author_name':    tw['author']['name'],
                               'time_stamp':     tw['created_at'],
                               'public_metrics': tw['public_metrics'],
                               'referenced_id':  ref_id,
                               'referenced_author_id': ref_auth_id}
                             # 'text_snippet':tw['text'][0:4],
        
        time_stamps.append(tw['created_at'])
        
    engagement_time = []
    t0 = datetime.datetime.strptime(root['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
    for ts in time_stamps:
        time = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.000Z')
        dt = time-t0
        engagement_time.append((86400*dt.days + dt.seconds)/3600)

    # Needed if we use the results of conv. query and not root tweet time
    #for i in range(len(engagement_time)):
    #    engagement_time[i] = engagement_time[i] - engagement_time[-1]
        
    n_authors = len(author_set)
        
    return conv_dict, engagement_time, n_authors


def create_conversation_network(conv_dict, engagement_time):
    """Returns a directed networkx graph of the conversation network
    where the user participating in the conversation are the nodes.
    
    Args:
        - conv_dict: a conversation dictionary, the
        output of get_conversation_dict().
    
    Returns:
        - DiG: a directed graph based on the conversation dictionary
        The nodes have author and time as an attribute.
    """
    DiG = nx.DiGraph()
    for i, key in enumerate(conv_dict):
        DiG.add_node(conv_dict[key]['author_id'], author=conv_dict[key]['author_name'], time=engagement_time[i],
                     rts=conv_dict[key]['public_metrics']['retweet_count'],
                     likes=conv_dict[key]['public_metrics']['like_count'])

    for key in conv_dict:
        if conv_dict[key]['referenced_author_id'] is not None:
            DiG.add_edge(conv_dict[key]['author_id'], conv_dict[key]['referenced_author_id'])
        
        
    """Fix the problem with some tweets getting no time:
    
    # e.g. by 
    for n in nx.nodes(DiG):
        try:
            print(DiG.nodes[n]['time'])
        except:
            print(f'no time for {n}')
            DiG.nodes[n]['time'] = 0
            print("new time", DiG.nodes[n]['time'])

    """      
    return DiG


def create_conversation_network_tree(conv_dict, engagement_time):
    """Returns a directed networkx graph of the conversation network
    where the messages are nodes.
    
    Args:
        - conv_dict: a conversation dictionary, the
        output of get_conversation_dict().
    
    Returns:
        - DiG: a directed graph based on the conversation dictionary
        The nodes have author and time as an attribute.
    """
    DiG = nx.DiGraph()
    for i, key in enumerate(conv_dict):
        DiG.add_node(key, author=conv_dict[key]['author_name'], time=engagement_time[i],
                     rts=conv_dict[key]['public_metrics']['retweet_count'],
                     likes=conv_dict[key]['public_metrics']['like_count'])

    for key in conv_dict:
        DiG.add_edge(key, conv_dict[key]['referenced_id'])
            
    """Fix the problem with some tweets getting no time:
    
    # e.g. by 
    for n in nx.nodes(DiG):
        try:
            print(DiG.nodes[n]['time'])
        except:
            print(f'no time for {n}')
            DiG.nodes[n]['time'] = 0
            print("new time", DiG.nodes[n]['time'])

    """         
    return DiG


def assign_time_attributes(graph):
    times = []
    for n in nx.nodes(graph):
        try:
            times.append(graph.nodes[n]['time'])
        except:
            graph.nodes[n]['time'] = 0
            times.append(graph.nodes[n]['time'])
    return times


def plot_engagement(engagement_time, save_path=None):
    max_ = max(engagement_time)
    n_bins = max(10, int(max_*6)) # make one bin per 10 minutes
    
    plt.figure(figsize=(9,9), clear=True)
    n, bs, _ = plt.hist(engagement_time, bins=n_bins)
    plt.title('Replies after posting')
    plt.xlabel('hours')
    plt.ylabel('replies')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    n_norm = n/np.sum(n)
    cdf = np.cumsum(n_norm)
    cdf = np.concatenate((np.zeros(1), cdf)) # same length as bins
    plt.figure(figsize=(9,9), clear=True)
    plt.plot(bs, cdf, '-')
    plt.title('Engagement CDF')
    plt.xlabel('hours')
    if save_path:
        plt.savefig(f'{save_path[:-4]}_cdf.svg')
    else:
        plt.show()
    plt.close()

    return


def get_downloaded_user_followers(folder_path: str):
    """Return a list of the user IDs of which we
    have retrieved followers.

    Args:
        - folder_path: path to the files of followers

    Returns:
        - user_ids: list of user IDs for which we have
        retrieved followers for.
    
    """
    file_paths = os.listdir(folder_path)
    user_ids = []
    for filename in file_paths:
        user_id = filename.split('_')[0]
        user_ids.append(user_id)
    return user_ids 


def build_partial_network(gexf_name, followers=True, following=True):
    """Create a .gexf network for the followers and followings of
    the set of users contained in the follower_ids and following_id
    folders.
    
    Args:
        - gexf_name: name of the output file (excluding '.gexf')
        
    No return value.
    """
    dir_graph = nx.DiGraph()
    
    # Get followers
    if followers:
        file_paths = os.listdir('follower_ids')
        for filename in file_paths:

            dest_id = filename.split('_')[0]

            with open(f'follower_ids/{filename}', mode='r', encoding="utf8") as file:
                user_ids = file.readlines()

            for line in user_ids:
                id_ = line[:-1] # Remove '\n'
                dir_graph.add_edge(id_, dest_id)
                
    if following:
        file_paths = os.listdir('following_ids')
        for filename in file_paths:

            source_id = filename.split('_')[0]

            with open(f'following_ids/{filename}', mode='r', encoding="utf8") as file:
                user_ids = file.readlines()

            for line in user_ids:
                id_ = line[:-1] # Remove '\n'
                dir_graph.add_edge(source_id, id_)
                
            
    nx.readwrite.gexf.write_gexf(dir_graph, path=f'{gexf_name}.gexf', encoding='utf-8')
    return


def create_conversation_graphs(dir_, show_plot=False, plot_with_colors=True):
    """Create conversation graphs in .svg and .gexf (Gephi) formats.
    Also creates histograms of the engagement in replies over time.
    Graphs are saved in 'sampled_conversations_graphs' folder.
    
    The kamada kawai layout seems to be expensive to compute for large graphs.
    One may just as well use the -gexf-file only, but the images cannot be exported
    to .svg format.
    
    Args:
        - dir_: directory consisting of conversation files
        (.jsonl formatted files containing tweets in the conversation)
        
    No return value.
    
    """
    file_paths = os.listdir(dir_)

    for conv_file_name in file_paths:

        conv_id = conv_file_name.split('_')[0]
        graph_file_name = f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.gexf'

        conv_sample = read_file(f'sampled_conversations/{conv_file_name}')

        conv_dict, engagement_time, n_authors = get_conversation_dict(conv_sample)

        dir_graph_sample = create_conversation_network(conv_dict, engagement_time)
        
        """
        # Room for computing metrics on the graphs!
        """
        
        nx.readwrite.gexf.write_gexf(dir_graph_sample, path=graph_file_name, encoding='utf-8')

        if plot_with_colors:
            assign_time_attributes(dir_graph_sample)

            plt.figure(figsize=(13,13), clear=True)
            vmax_ = int(engagement_time[-1]+1)
            nc = [v for k,v in nx.get_node_attributes(dir_graph_sample, 'time').items()]
            nx.draw_kamada_kawai(dir_graph_sample, with_labels=False, font_weight='bold', node_color = nc, vmin=0, vmax=vmax_, cmap = plt.cm.get_cmap('rainbow'))
            plt.savefig(f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.svg')
            if show_plot:
                plt.show()
        else:
            plt.figure(figsize=(13,13), clear=True)
            nx.draw_kamada_kawai(dir_graph_sample, with_labels=False)
            plt.savefig(f'sampled_conversations_graphs/{conv_id}_conversation_graph_users.svg')
            if show_plot:
                plt.show()
        
        if show_plot:
            plot_engagement(engagement_time)
        else:
            plot_engagement(engagement_time, save_path=f'sampled_conversations_graphs/{conv_id}_engagement_time.svg')
        
        
        
def sample_num_retweets(dir_):
    """Collects the number of retweets of the original tweet
    in the conversation files located in dir_.
    
    Args:
        - dir_: directory consisting of conversation files
        (.jsonl formatted files containing tweets in the conversation)
        
    Returns:
        - RTs: A list of the number of retweets of the conversation root.
        - cid: A list corresponding to the conversation id to each number
        in RTs. (Not every conversation has an accessible root.)
    """
    
    file_paths = os.listdir(dir_)
    
    RTs = []
    cid = []
    for conv_file_name in file_paths:
        conv_id = conv_file_name.split('_')[0]
        with open(f'sampled_conversations/{conv_file_name}', 'r') as filehandle:
            for line in filehandle:
                obj = json.loads(line)
                try:
                    found = False
                    for ref_tw in obj['referenced_tweets']: # check referenced tweets
                        # check that the referenced tweet is indeed the root
                        if ref_tw['id'] == conv_id: # and ref_tw['type'] == 'replied_to' ?
                            RTs.append(ref_tw['public_metrics']['retweet_count'])
                            cid.append(conv_id)
                            found = True
                            break
                    if found: # go to next file
                        break
                except:
                    continue
                
    return RTs, cid