import os
import logging
import datetime
#import numpy as np
#import pandas as pd
import networkx as nx
from file_manip import *
from plot_tools import *
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def get_saved_conversation_ids(lower_bound: int, upper_bound: int, folder='sampled_conversations') -> list:
    """Returns a list of conversation ids between a given
    lower and upper bound from the folder of saved conversations.
    
    Args:
        - lower_bound: integer lower limit of conversation IDs to include
        - upper_bound: integer upper limit of conversation IDs to include
        - folder: folder to search for conversations in
    
    Returns:
        - conv_ids: a list of conversation IDs from the folder
    """
    
    file_paths = os.listdir(folder)
    conv_ids = []
    
    for conv_file_name in file_paths:
        conv_id = conv_file_name.split('_')[0]
        if int(conv_id) >= lower_bound and int(conv_id) <= upper_bound:
            conv_ids.append(conv_id)

    return conv_ids


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


def mean_square_error_fo(bin_values, lambda_: float, beta_: float) -> float:
    """Computes the mean square error of a first order
    linear model with parameters beta_ and lambda_, with
    data discretized in steps of 1 unit of time.
    
    Args:
        - bin_values: bin values of the engagement histogram
        - beta_: parameter modelling the initial response to
        the tweet. Should be positive
        - lambda_: decay constant which should take values
        less than 1
        
    Returns:
        - MSE: the mean square error
    """
    x_hat = beta_
    MSE = 0
    for x in bin_values:
        MSE += (x-x_hat)**2
        x_hat *= (1-lambda_)
    MSE = MSE / len(bin_values)
    return MSE


def eval_error(time, model_engagement, true_engagement) -> tuple[float, float]:
    """Computes the mean square error, and the ratio between the sum of
    squared residuals and the sum of the squared signal in continuous time
    
    Args:
        - time: time vector for the observations
        - model_engagement: predicted engagement of fitted model
        - true_engagement: engagement at times specified in time vector
        
    Returns:
        - MSE: mean square error
        - RSS_frac: ratio of residuals squared and sum_signal_squared
        - sum_signal_squared: sum of squared signal samples
    """
    res = model_engagement - true_engagement
    sq_residual = np.square(res)
    sq_signal = np.square(true_engagement)
    
    MSE = np.mean(sq_residual)
    sum_signal_squared = np.sum(sq_signal)
    RSS_frac = np.sum(sq_residual)/sum_signal_squared
    
    #plt.hist(res, bins=200)
    #plt.show()
    
    return MSE, RSS_frac, sum_signal_squared


def exponential(x_, lambda_, beta_):
    """Exponential decay with parameters lambda and beta."""
    return beta_*np.exp(-lambda_*x_)


def log_exponential(x_, lambda_, beta_):
    return np.log(exponential(x_, lambda_, beta_))


def log_exp(x_, lambda_, log_beta_):
    """Logarithm of exponential decay with parameters lambda and beta."""
    return log_beta_ - lambda_*x_


def power(x_, lambda_, beta_):
    """Power law."""
    return beta_*(np.power(np.array(x_, dtype=float), -lambda_))


def estimate_lin_decay(time, engagement):
    t, log_eng = [], []
    for i, e in enumerate(engagement):
        if e != 0:
            log_eng.append(np.log(e))
            t.append(time[i])
    
    time = np.array(t)
    log_eng = np.array(log_eng)
    
    lin_model = LinearRegression(fit_intercept=True)
    
    #log_eng = np.log(np.array(engagement)+1)
    lin_model.fit(time.reshape(-1, 1), log_eng.reshape(-1, 1))
    
    lambda_ = -lin_model.coef_[0][0]
    beta_ = np.exp(lin_model.intercept_[0])
    
    #print("coefs:", lambda_, beta_)
    
    return lambda_, beta_
    

def estimate_decay_parameters(time, engagement, loss_='linear', f_scale_=1.0):
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
    
    # Linear fit (alt. L1 fit)
    popt, _ = curve_fit(exponential, time, engagement, p0=init,
                        bounds=bounds_, method=method_, loss=loss_, f_scale=f_scale_)
    
    # Log fit
    #engagement = np.array(engagement)
    #log_eng = np.log(engagement+1e-3)
    #bounds_ = ([1e-4, 0], [1e3, 9]) # 0 <= log(beta) <= 9
    #popt, _ = curve_fit(log_exponential, time, engagement, p0=init,
    #                    bounds=bounds_, method=method_)
    
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


def u_biexponential(x_, alpha, beta, gamma, rho):
    """Solution to system
    dx1/dt = alpha*x1(t) + beta*x2(t)
    dx2/dt = gamma*x2(t) + rho*d(t)
    """
    if np.abs(gamma - alpha) < 1e-10:
        return np.exp(-alpha*(x_))*rho
    else:
        return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))


def estimate_unconstr_biexp(time, engagement):
    
    def biexponential_unconstr_opt(x_, alpha, beta, gamma, rho):
        """Solution to system
        dx1/dt = alpha*x1(t) + beta*x2(t)
        dx2/dt = gamma*x2(t) + rho*d(t)
        """
        if np.abs(gamma - alpha) < 1e-10:
            return np.exp(-alpha*(x_))*rho
        else:
            return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))
    
    method_ = 'lm'
    init = [1, 1, 1, 1]
    popt, _ = curve_fit(biexponential_unconstr_opt, time, engagement, p0=init, method=method_)
    a, b, g, r = popt[0], popt[1], popt[2], popt[3]
    return a, b, g, r


def estimate_biexponential(time, engagement, loss_='linear'):
    
    def biexponential_opt(x_, alpha, beta, d_gamma, rho):
        """Solution to system
        dx1/dt = alpha*x1(t) + beta*x2(t)
        dx2/dt = gamma*x2(t) + rho*d(t)
        """
        gamma = alpha + d_gamma
        return np.exp(-alpha*(x_))*rho*beta/(gamma-alpha) + np.exp(-gamma*(x_))*rho*(1-(beta/(gamma-alpha)))
    
    method_ = 'trf'
            # alpha, beta, d_gamma, rho
    bounds_ = ([0,    0,    1e-9, 1e-9],
               [1e9,  1e9,  1e9,  1e9])
    init = [1e-1, 1, 1e-1, 1]
    
    """Previous bounds:
    bounds_ = ([1e-5, 1e-2, 1e-5, 1e-3],
               [1e3,  1e9,  1e3,  1e9])
    init = [1e-1, 1, 1e-1, 1]
    """
    
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

    peaks_x, peaks_y = [], []
    
    prev_x = xx[0]
    segment_x, segment_y = [xx[0]], [yy[0]]
    curr_max, curr_max_y = xx[0], yy[0]

    # Filter out adjacent peaks
    for i in range(1, len(xx)):
        if xx[i] > prev_x + adj_dist:
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

    return peaks_x, peaks_y


def plot_peak_detection(y, result, conv_id, plot=True):
    """Plots the peaks from the raw result of the detection algorithm.
    The result data is filtered before plotting.
    
    Args:
        - y: the engagement histogram values
        - result: result dictionary of the peak detection algorithm
        - conv_id: conversation id
        
    Returns:
        - type_: the type of engagement graph, first position
        indicates delay (1X) or no delay (0X), the second position
        indicates a single peak (0), double peak (1), bump (2), or
        multiple peaks (3).
        - first_peak: bin number of the first peak (i.e., delay).
        Chosen as the largest of the first two peaks, or -1 in
        the odd case that there is no peak.
    """
    
    peaks_x, peaks_y = filter_peaks(result, y, peak_threshold=0.15, adj_dist=2)
    
    ### TODO: find also second peak, if it is relevant!
    
    # DECISION: We use the heuristic to pick the largest of the two first peaks
    type_ = 0
    if len(peaks_x) != 0:
        if peaks_x[0] <= result['lag']+1:
            type_ += 0
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
    
    
    if plot:
        plt.figure(figsize=(8,8))
        #plt.subplot(211)
        plt.plot(np.arange(1, len(y)+1), y, color='navy', lw=2)

        plt.plot(np.arange(1, len(y)+1),
                   result["avgFilter"], '--', color='gold', lw=1)

        plt.plot(peaks_x, peaks_y, 'o', color='red')

        plt.plot(np.arange(1, len(y)+1),
                   result["avgFilter"] + result['threshold'] * result["stdFilter"], color="sienna", lw=0.8)

        plt.plot(np.arange(1, len(y)+1),
                   result["avgFilter"] - result['threshold'] * result["stdFilter"], color="sienna", lw=0.8)

        suppressed = [i for i,r in enumerate(result['signals']) if r > 0]
        supp_y = -1*np.ones(len(suppressed))
        plt.plot(suppressed, supp_y, 'o', color='salmon')
        plt.savefig(f'sampled_conversations_graphs/peak_detection/{type_}/{conv_id}_peaks.png')
        plt.close('all')
        #plt.show()
    
    return type_, first_peak, second_peak


def get_file_paths(conv_id: str) -> tuple[str, str, str]:
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


def has_public_metrics(retweet_path: str):
    contains_metrics = False
    for rt in file_generator(retweet_path):
        if 'public_metrics' in rt['author']:
            contains_metrics = True
        break
    return contains_metrics


def create_bins(delta_sec, max_time, padding=True):
    """Creates bins spaced by delta_sec seconds for histogram generation.
    Zero bins can be added to conversations that have a maximum engagement
    time under 72 hours.
    
    Args:
        - delta_sec: bin width in seconds
        - max_time: maximum time of engagement in hours
        - padding: boolean indicating whether to pad short conversations
        with zero bins
    
    Returns:
        - bins: bins with specified width
        - delta_h: bin width in hours
    """
    if padding:
        max_time = max(max_time, 72)
    delta_h = delta_sec/3600
    n_bins = int((max_time//delta_h) + 2) # Add 2 to compensate for integer division and endpoint in linspace
    bins = np.linspace(start=0, stop=n_bins*delta_h, num=n_bins, endpoint=False, retstep=False, dtype=None, axis=0)
    return bins, delta_h


def perform_peak_detection(engagement_hist_values):
    """Performs peak detection and returns the type of the
    conversation, and the locations of the first and second
    peaks.
    
    Args:
        - engagement_hist_values: engagement histogram values
        
    Returns:
        - type_: conversation type, see plot_peak_detection() for
        details
        - first_peak: location of the first peak (bin number)
        - second_peak: location of the second peak (bin number)
    """
    
    lag_ = 10
    rd = np.zeros(lag_)
    time_series = np.concatenate((rd,engagement_hist_values))
    result = peak_detection(time_series, lag=lag_, threshold=1.5, influence=0.8)
    type_, first_peak, second_peak = plot_peak_detection(time_series, result, conv_id='', plot=False)

    return type_, first_peak, second_peak


def load_engagement(root_t, conversation_path: str, retweet_path: str, quote_path=None):
    """Reads and returns the engagement times (and followers)
    of all interactions with the conversation root.
    
    Args:
        - root_t: time of root creation (root['created_at'])
        - conversation_path: path to file containing replies
        - retweet_path: path to the retweet file
        - quote_path: (To implement) path to file containing quotes
        
        
    Returns:
        - engagement_time: time stamps of interactions
        - n_replies: number of replies in conversation,
        is also the index for the first retweet time in
        the vector: use it to divide retweet/reply times
        - followers: element i contains the number of
        followers of the user that interacted at time 
        engagement_time[i]
        - n_replying_users: number of unique users that
        replied to the conversation
    """
    engagement_time, followers = [], []
    replying_users = set()
    
    for reply in file_generator(conversation_path):
        reply_t = compute_relative_time(root_t, reply['created_at'])
        engagement_time.append(reply_t)
        replying_users.add(reply['author_id'])
        #followers.append(reply['author']['public_metrics']['followers_count'])

    n_replies = len(engagement_time)

    for retweet in file_generator(retweet_path): 
        retweet_t = compute_relative_time(root_t, retweet['created_at'])
        engagement_time.append(retweet_t)
        #followers.append(-1)
        #followers.append(retweet['author']['public_metrics']['followers_count'])
        
    if quote_path:
        for quote in file_generator(quote_path):
            quote_t = compute_relative_time(root_t, quote['created_at'])
            engagement_time.append(quote_t)
            #followers.append(quote['author']['public_metrics']['followers_count'])
    
    n_replying_users = len(replying_users)
    
    return engagement_time, n_replies, followers, n_replying_users


def fit_first_order(time, engagement_hist, delta_h, peak_detection=True):
    """Fits a first order model and returns associated statistics.
    
    Args:
        - time: time vector for the observations
        - engagement_hist: values for engagement bins
        - delta_h: bin width in hours
        
    Returns:
        - lambda_, beta_: system parameters, see estimate_decay_parameters()
        - model_eng: engagement for the model
        - type_: type of conversation, see see plot_peak_detection()
        - first_peak, second_peak: bin number for first and second peak
    """
    
    # Peak detection for the first order model
    if peak_detection:
        type_, first_peak, second_peak = perform_peak_detection(engagement_hist)
    else:
        type_, first_peak, second_peak = 'None', 0, -1

    if first_peak > 0:
        truncated_t = time[first_peak:] - delta_h*first_peak # = time[:-first_peak]?
        truncated_n = engagement_hist[first_peak:]
    else:
        truncated_t = time
        truncated_n = engagement_hist
        first_peak = 0
    
    # Fit the first order model
    opt_params = estimate_decay_parameters(truncated_t, truncated_n, loss_='linear', f_scale_=1.0)
    lambda_, beta_ = opt_params[0], opt_params[1]
    trunc_model_eng = exponential(truncated_t, lambda_, beta_)
    model_eng = np.concatenate((np.zeros(first_peak), trunc_model_eng))
    
    return lambda_, beta_, model_eng, type_, first_peak, second_peak


def process_engagement_times(conv_ids: list[str], delta_sec: float, plot_engagement=False) -> None:
    """Retrieves the engagement times of retweets and replies.
    Only considers conversations where root, conversation, and
    retweet files exist.
    
    Args:
        - conv_ids: iterator over conversation IDs
        - delta_sec: the time step length in seconds
        of the time series discretization in seconds
        
    No return value.
    """
        
    sufficient, missing_data, too_few, opt_failed = 0, 0, 0, 0
    
    # Iterate with generator from filter(all_files_exist, conv_ids)?
    for conv_id in conv_ids:
        root_path, conv_path, retw_path = get_file_paths(conv_id)
        
        # Check that root, conversation and retweet files exist.
        if os.path.isfile(root_path) and os.path.isfile(conv_path) and os.path.isfile(retw_path):
            root = read_file(root_path)[0]
        else:
            missing_data += 1
            continue
        
        """If plotting the number of followers of those who interact by retweeting
        if not has_public_metrics(retw_path):
            missing_data += 1
            continue
        """
        engagement_time, n_replies, followers, n_rep_users = load_engagement(root['created_at'], conv_path, retw_path)
        tot_eng = len(engagement_time)
        
        # For reply and retweet separation
        #if n_replies < 30 or (tot_eng-n_replies)<30:
        #    continue
        
        n_api_retweets = root['public_metrics']['retweet_count']
        # DECISION: Ignore the conversations that have fewer than 50 replies/retweets
        if (tot_eng < 50) or ((tot_eng-n_replies) <= 1e-5*n_api_retweets):
            too_few += 1
            continue
        else:
            sufficient += 1
        
        reply_ratio = n_replies / tot_eng
        
        # Bin and estimate engagement curve (time series with increments of delta_t) 
        bins, delta_h = create_bins(delta_sec, np.max(engagement_time), padding=True)
        n, _ = np.histogram(engagement_time, bins)
        
        # Retweet and reply histograms
        #reply_hist, _ = np.histogram(engagement_time[:n_replies], bins)
        #retw_hist, _ = np.histogram(engagement_time[n_replies:], bins)

        t = (bins[1:] + bins[:-1])/2 # Time vector (center of bins, can also use bins[:-1])        
        
        try:
            
            lambda_, beta_, fo_model_eng, type_, first_peak, second_peak = fit_first_order(t, n, delta_h, peak_detection=False)
            #lambda_, beta_, type_, first_peak, second_peak = -1, -1, 'None', -1, -1 # if 1st order model is skipped
            #fo_MSE, fo_RSS_frac = -1, -1
            
            # Fit the second order model
            # a, b, g, rho = estimate_biexponential(time=t, engagement=n, loss_='linear')
            #so_model_eng = biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            #a, b, g, rho = estimate_unconstr_biexp(time=t, engagement=n)
            #so_model_eng = u_biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            
            # Reply and retweet separation
            #a, b, g, rho = estimate_biexponential(time=t, engagement=reply_hist, loss_='linear')
            #so_model_eng = biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            #a2, b2, g2, rho2 = estimate_biexponential(time=t, engagement=retw_hist, loss_='linear')
            #so_model_eng2 = biexponential(t, alpha=a2, beta=b2, gamma=g2, rho=rho2)
                
            # Evaluate model errors
            fo_MSE, fo_RSS_frac, sum_signal_squared = eval_error(t, fo_model_eng, n)
            #so_MSE, so_RSS_frac, sum_signal_squared = eval_error(t, so_model_eng, n)
            #so_MSE2, so_RSS_frac2, _ = eval_error(t, so_model_eng2, n)
            
            root_followers = root['author']['public_metrics']['followers_count']
            n_api_replies = root['public_metrics']['reply_count']
            n_api_quotes = root['public_metrics']['quote_count']
            
            #result_file = f'parameter_estimations/model_evaluations.txt'
            result_file = f'parameter_estimations/model_evaluations_fo_nodelay.txt'
            #result_file = f'parameter_estimations/model_evaluations_reply_retweets.txt'
            hist_folder = 'sampled_conversations_graphs/engagement_histograms'
            
            metrics1 = f'{conv_id},{root_followers},{n_rep_users},{type_},{first_peak},{second_peak},'
            metrics2 = f'{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies},'
            #estimates = f'{lambda_},{beta_},{a},{b},{g},{rho},{fo_MSE},{fo_RSS_frac},{so_MSE},{so_RSS_frac},{sum_signal_squared}'
            #estimates2 = f'{a},{b},{g},{rho},{so_MSE},{so_RSS_frac},{a2},{b2},{g2},{rho2},{so_MSE2},{so_RSS_frac2},{sum_signal_squared}'
            estimates3 = f'{lambda_},{beta_},{fo_MSE},{fo_RSS_frac},{sum_signal_squared}'
            write_text(file_name=result_file, text=metrics1+metrics2+estimates3)
            
            # path=f'{hist_folder}/{conv_id}_first-and-second_order.png'
            # path=f'{hist_folder}/{conv_id}_second_order_reply_retweets.png'
            if plot_engagement:
                _, _ = create_hist([engagement_time[:n_replies], engagement_time[n_replies:]], bins,
                                path=f'{hist_folder}/{conv_id}_first_order_no_delay.png',
                                title='Tweet Engagement', xlab='time (h)', ylab='counts',
                                log_=False, t=t, y=fo_model_eng)#, y2=so_model_eng)
            
                # sc_path = f'sampled_conversations_graphs/engagement_histograms/interactor_followers/{conv_id}_engag_ds{int(delta_sec)}_type3_flws.png'
                #_, _ = create_hist_scatter(engagement_time, bins, path=sc_path,
                #                           title='Engagement over time (replies and RTs)', xlab='time (h)', ylab='counts (replies and retweets)', log_=False,
                #                           t=t+delta_h*first_peak, y=model_eng, scatter_y=followers, root_flw=root_followers)
            
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
