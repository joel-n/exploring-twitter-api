import re
import os
import logging
import datetime
import networkx as nx
from plot_tools import *
from file_manip import read_file, file_generator, write_text
from curve_fitting import eval_error, estimate_biexponential, estimate_decay_parameters, exponential, biexponential


def get_saved_conversation_ids(lower_bound: int, upper_bound: int, folder='sampled_conversations', delimiter='_') -> list:
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
        conv_id = conv_file_name.split(delimiter)[0]
        if int(conv_id) >= lower_bound and int(conv_id) <= upper_bound:
            conv_ids.append(conv_id)

    return conv_ids


def influencers_to_retrieve(conv_ids, max_flw_threshold, thr_pct='095'):
    """Returns the set of influencers that appear
    in the given conversations. Files with number
    of influencer followers must exist.
    
    Args:
        - conv_ids: list of conversation IDs
        - max_flw_threshold: Conversations with an
        influencer with higher follower count than this
        number will be ignored.
        - thr_pct: ending in file name (local infl. threshold)
    
    Returns:
        - infl_ids: set of influencer IDs
    """
    infl_ids = set()
    for conv_id in conv_ids:    
        skip_conv = False
        
        for nflw in file_generator(f'infl_nflws/{conv_id}_nflws_{thr_pct}.txt'):
            if nflw > max_flw_threshold:
                skip_conv = True
                break

        if skip_conv:
            continue

        for infl_id in file_generator(f'infl_ids/{conv_id}_{thr_pct}.txt'):
            if not os.path.isfile(f'infl_follower_ids/{infl_id}_followers.txt'):
                infl_ids.add(infl_id)
    return infl_ids



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
        - y: engagement histogram values (including zero-bins for lag)
        - peak_threshold: fraction of global maximum a peak must attain
        in order not to be filtered out
        - adj_dist: max distance between adjacent peaks; these will be
        reduced to the highest peak
        
    Returns:
        - peaks_x: x-coordinate for the relevant peaks
        - peaks_y: y-coordinate for the relevant peaks
    """
    
    xx, yy = [], []
    
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
            # New segment
            peaks_x.append(curr_max)
            peaks_y.append(curr_max_y)
            
            segment_y = [yy[i]]
            segment_x = [xx[i]]
            curr_max = xx[i]
            curr_max_y = yy[i]
        else:
            # Append current segment
            if curr_max_y < yy[i]:
                # Replace y_max
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
        - y: the engagement histogram values (including zero-bins for lag)
        - result: result dictionary of the peak detection algorithm
        - conv_id: conversation id
        
    Returns:
        - type_: the type of engagement graph, first position
        indicates delay (1X) or no delay (0X), the second position
        indicates a single peak (0), double peak (1), bump (2), or
        multiple peaks (3).
        - first_peak: bin number of the first peak (i.e., delay).
        NOTE: Chosen as the largest of the first two peaks, or -1 in
        the odd case that there is no peak.
    """
    
    peaks_x, peaks_y = filter_peaks(result, y, peak_threshold=0.15, adj_dist=2)
    
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
        np_peaks_x, np_peaks_y = np.array(peaks_x), np.array(peaks_y)
        for i in [len(y)-2]:
            f,ax = plt.subplots(figsize=(8,8))
            #plt.subplot(211)
            plt.plot(np.arange(-10, len(y)-10), y, color='navy', lw=3, zorder=0)

            plt.plot(np.arange(-10, len(y[:i+1])-10),
                    result["avgFilter"][:i+1], '--', color='red', lw=5, zorder=2)

            plt.scatter(np_peaks_x[np_peaks_x<i]-11, np_peaks_y[np_peaks_x<i], s=90, color='crimson', label='detected peaks', zorder=4)

            plt.fill_between(x=np.arange(-10, len(y[:i+1])-10),
                            y1=result["avgFilter"][:i+1] + result['threshold'] * result["stdFilter"][:i+1],
                            y2=result["avgFilter"][:i+1] - result['threshold'] * result["stdFilter"][:i+1],
                            lw=2, alpha=0.3, color="seagreen", zorder=1)

            suppressed = np.array([i-10 for i,r in enumerate(result['signals'][:i+1]) if r > 0])
            supp_y = -1*np.ones(len(suppressed))
            plt.scatter(suppressed[suppressed < i], supp_y[suppressed < i], s=70, edgecolor='black', color='salmon', label='ignored peaks', zorder=5)
            plt.xlim(-10,70)
            plt.ylim(-50, max(peaks_y)+50)
            ax.tick_params(labelsize=30, direction="in", which='both')
            plt.xlabel('time (h)', fontsize=30)
            plt.ylabel('interactions', fontsize=30)
            if i > 0:
                plt.legend(loc='upper right', fontsize=26)
            plt.tight_layout()
            plt.savefig(f'sampled_conversations_graphs/peak_detection/peaks{i}.svg')
            plt.close('all')
            #plt.show()
    
    peaks_x, peaks_y = np.array(peaks_x) - result['lag'] - 1, np.array(peaks_y)

    return type_, first_peak, second_peak, peaks_x, peaks_y


def get_file_paths(conv_id: str) -> tuple[str, str, str, str]:
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
    quote_path = f'quotes/{conv_id}_quotes.jsonl'
    
    return root_path, conv_path, retw_path, quote_path


def has_public_metrics(retweet_path: str) -> bool:
    contains_metrics = False
    for rt in file_generator(retweet_path):
        if 'public_metrics' in rt['author']:
            contains_metrics = True
        break
    return contains_metrics


def create_bins(delta_sec, max_time, padding=True) -> tuple[np.array, float]:
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


def perform_peak_detection(engagement_hist_values, plot=False):
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
        - peaks_x: peak time values
        - peaks_y: peak heights
    """
    
    lag_ = 10
    rd = np.zeros(lag_)
    time_series = np.concatenate((rd,engagement_hist_values))
    result = peak_detection(time_series, lag=lag_, threshold=1.5, influence=0.8)
    type_, first_peak, second_peak, peaks_x, peaks_y = plot_peak_detection(time_series, result, conv_id='', plot=plot)

    return type_, first_peak, second_peak, peaks_x, peaks_y


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
        type_, first_peak, second_peak, _, _ = perform_peak_detection(engagement_hist)
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


def process_engagement_times(conv_ids: list[str], delta_sec: float, plot_engagement=False, result_file_mod='') -> None:
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
        reply_times_path, retweet_times_path, quote_times_path = get_interaction_paths(conv_id)
        root_path = f'root_tweets/{conv_id}_root.jsonl'
        
        # Check that root, conversation and retweet files exist.
        if os.path.isfile(root_path) and os.path.isfile(reply_times_path) and os.path.isfile(quote_times_path):
            root = read_file(root_path)[0]
        else:
            missing_data += 1
            continue
        
        """If plotting the number of followers of those who interact by retweeting
        if not has_public_metrics(retw_path):
            missing_data += 1
            continue
        """

        reply_times, retweet_times, quote_times = load_precomputed_engagement_times(conv_id)
        n_replies, n_retweets, n_quotes = len(reply_times), len(retweet_times), len(quote_times)
        tot_eng = n_replies + n_retweets + n_quotes

        n_api_retweets = root['public_metrics']['retweet_count']
        # DECISION: Ignore the conversations that have fewer than 50 replies/retweets
        if (tot_eng < 50) or (n_retweets <= 0.5*n_api_retweets):
            too_few += 1
            continue
        else:
            sufficient += 1
        
        reply_ratio = n_replies / tot_eng
        
        engagement_time = np.concatenate((np.zeros(1), reply_times, retweet_times, quote_times))
        bins, delta_h = create_bins(delta_sec, np.max(engagement_time), padding=True)
        n, _ = np.histogram(engagement_time, bins)
        
        #t = (bins[1:] + bins[:-1])/2 # Time vector (center of bins, can also use bins[:-1])        
        t = bins[:-1]

        try:
            
            lambda_, beta_, fo_model_eng, type_, first_peak, second_peak = fit_first_order(t, n, delta_h, peak_detection=False)
            #lambda_, beta_, type_, first_peak, second_peak = -1, -1, 'None', -1, -1 # if 1st order model is skipped
            #fo_MSE, fo_RSS_frac = -1, -1

            # Fit the second order model
            a, b, g, rho = estimate_biexponential(time=t, engagement=n, loss_='linear')
            so_model_eng = biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            #a, b, g, rho = estimate_unconstr_biexp(time=t, engagement=n)
            #so_model_eng = u_biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)

            # Reply and retweet separation
            #a, b, g, rho = estimate_biexponential(time=t, engagement=reply_hist, loss_='linear')
            #so_model_eng = biexponential(t, alpha=a, beta=b, gamma=g, rho=rho)
            #a2, b2, g2, rho2 = estimate_biexponential(time=t, engagement=retw_hist, loss_='linear')
            #so_model_eng2 = biexponential(t, alpha=a2, beta=b2, gamma=g2, rho=rho2)
                
            # Evaluate model errors
            fo_MSE, fo_RSS_frac, sum_signal_squared = eval_error(t, fo_model_eng, n)
            so_MSE, so_RSS_frac, sum_signal_squared = eval_error(t, so_model_eng, n)
            #so_MSE2, so_RSS_frac2, _ = eval_error(t, so_model_eng2, n)
            
            root_followers = root['author']['public_metrics']['followers_count']
            n_api_replies = root['public_metrics']['reply_count']
            n_api_quotes = root['public_metrics']['quote_count']
            
            result_file = f'parameter_estimations/model_evaluations{result_file_mod}.txt'
            hist_folder = 'sampled_conversations_graphs/engagement_histograms'
            
            metrics1 = f'{conv_id},{root_followers},{type_},{first_peak},{second_peak},'
            metrics2 = f'{n_replies},{reply_ratio},{tot_eng},{n_api_retweets},{n_api_replies},'
            estimates = f'{lambda_},{beta_},{a},{b},{g},{rho},{fo_MSE},{fo_RSS_frac},{so_MSE},{so_RSS_frac},{sum_signal_squared}'
            #estimates2 = f'{a},{b},{g},{rho},{so_MSE},{so_RSS_frac},{a2},{b2},{g2},{rho2},{so_MSE2},{so_RSS_frac2},{sum_signal_squared}'
            #estimates3 = f'{lambda_},{beta_},{fo_MSE},{fo_RSS_frac},{sum_signal_squared}'
            write_text(file_name=result_file, text=metrics1+metrics2+estimates)
            
            if plot_engagement and (sufficient % 100) == 0:
                print(f'Processed {sufficient} conversations')
                _, _ = create_hist([reply_times, retweet_times, quote_times], bins,
                                path=f'{hist_folder}/{conv_id}{result_file_mod}.png',
                                title='', xlab='time (h)', ylab='counts',
                                log_=False, t=t, y=fo_model_eng, y2=so_model_eng)
            
        except Exception as e:
            logging.warning(e)
            logging.info(f'optimization failed for conversation {conv_id}: replies:{n_replies}, reply_ratio:{reply_ratio}')
            sufficient -= 1
            opt_failed += 1
        
    print('Plots: {}, missing data: {}, too few data points: {}, optimization failed: {}'.format(sufficient, missing_data, too_few, opt_failed))
    
    return


def get_interaction_paths(conv_id: str) -> tuple[str, str, str]:
    """Returns the paths to precomputed interaction times"""
    reply_times_path   = f'interaction_times/reply_times/{conv_id}.txt'
    retweet_times_path = f'interaction_times/retweet_times/{conv_id}.txt'
    quote_times_path   = f'interaction_times/quote_times/{conv_id}.txt'
    return reply_times_path, retweet_times_path, quote_times_path


def get_follower_paths(conv_id: str) -> tuple[str, str, str]:
    """Returns the paths to the number of followers for the interactions"""
    reply_followers_path   = f'interaction_followers/reply_followers/{conv_id}_rep_flw.txt'
    retweet_followers_path = f'interaction_followers/retweet_followers/{conv_id}_rt_flw.txt'
    quote_followers_path   = f'interaction_followers/quote_followers/{conv_id}_q_flw.txt'
    return reply_followers_path, retweet_followers_path, quote_followers_path


def load_precomputed_engagement_times(conv_id: str):
    """Loads the engagement data of a conversation from
    the folder interaction_times.
    
    Args:
        - conv_id: conversation ID

    Returns:
        - lists of engagement times for the
        respective interaction category
    """
    reply_times_path, retweet_times_path, quote_times_path = get_interaction_paths(conv_id)
    reply_times = read_file(reply_times_path)
    retweet_times = read_file(retweet_times_path)

    if os.path.isfile(quote_times_path):
        quote_times = read_file(quote_times_path)
    else:
        quote_times = []

    return reply_times, retweet_times, quote_times


def load_precomputed_engagement_followers(conv_id: str):
    """Loads the engagement data of a conversation from
    the folder interaction_followers.
    
    Args:
        - conv_id: conversation ID

    Returns:
        - lists of engagement follower counts for the
        respective interaction category
    """
    reply_followers_path, retweet_followers_path, quote_followers_path = get_follower_paths(conv_id)
    reply_followers = read_file(reply_followers_path)
    retweet_followers = read_file(retweet_followers_path)
    
    if os.path.isfile(quote_followers_path):
        quote_followers = read_file(quote_followers_path)
    else:
        quote_followers = []
    
    return reply_followers, retweet_followers, quote_followers


def load_conv_followers(follower_dist_path: str) -> list[int]:
    """Loads the followers of the unique users in the conversation.
    
    Args:
        - follower_dist_path: path to file that contains
        the unique follower counts
    
    Returns:
        - unique_flws: list of follower count
    """
    unique_flws = []
    for flw in file_generator(follower_dist_path):
        unique_flws.append(flw)
    return unique_flws


def compute_interactions(infl_times: np.array, interaction_times_list: tuple[np.array], window: float):
    ### TODO: If the window size is w, and influencer time is k<w, then use window size k instead
    interactions_before, interactions_after = np.zeros(len(infl_times)), np.zeros(len(infl_times))
    for interaction_times in interaction_times_list:
        for t in interaction_times:
            v  = infl_times-t
            a = np.where((v < 0) & (v > -window))
            b = np.where((v > 0) & (v <  window))
            z_a, z_b = np.zeros(len(v), dtype=int), np.zeros(len(v), dtype=int)
            z_a[a] = 1
            z_b[b] = 1
            interactions_after += z_a
            interactions_before += z_b
    return interactions_after, interactions_before


def not_space(s: str) -> bool:
    """Returns false if s is empty or
    consists of whitespace characters.
    
    Args:
        - s: string to check
    
    Returns:
        - boolean indicating if string
        fulfills the criterion
    """
    return not (s.isspace() or not s)


def handle_special(s: str) -> str:
    """Removes quotation marks and special
    characters (such as ampersands etc.) from
    the string where they would cause an error
    if passed to the API. Used when fetching
    retweets and querying on the tweet text.
    Seems to work for right-to-left languages
    as well, e.g., Arabic and Hebrew.
    
    Args:
        - s: string that contains quotation
        marks (" or '), and a hyperlink removed.
        
    Returns:
        - a string with quotation marks in the
        correct places: only between characters
        in the string, and no quotation marks at
        the end or beginning of the string.
    """
    #s_list = s.split('"')
    s_list = re.split('\'|"|&amp;|&gt;|&lt;', s)
    # TODO/Decision: Heuristic: If there is a string that has more than 20 characters in it
    # it alone may be used as a query to minimize the risk of other special characters
    # messing up the query
    return '" "'.join(filter(not_space, s_list))



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
