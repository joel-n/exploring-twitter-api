import os
from file_manip import file_generator, append_objs_to_file
from analysis_tools import get_file_paths


def make_infl_id_file(conv_id: str, threshold: float, threshold_percentile: float) -> None:
    """Writes influencer IDs and follower count to file, given a threshold
    computed in advance. Does not create a file in case no influencers are
    found.

    Args:
        - conv_id: Conversation ID
        - threshold: threshold on followers to count as an influencer
        - threshold_percentile: Percentile (divided by 100) of local/global follower
        distribution used to compute the threshold. Will be part of the file name.
    
    """
    threshold_percentile = str(threshold_percentile).replace('.', '')
    infl_id_path = f'infl_ids/{conv_id}_{threshold_percentile}.txt'
    infl_nflw_path = f'infl_nflws/{conv_id}_nflws_{threshold_percentile}.txt'
    
    if os.path.isfile(infl_id_path):
        return
    
    infl_attr = set() # set of influencer ID and number of followers, hopefully the combo is unique!
    paths = [*get_file_paths(conv_id)] # TODO: option to read files from disk E:/
    for path in paths:
        for tweet in file_generator(path):
            if tweet['author']['public_metrics']['followers_count'] > threshold:
                infl_attr.add((int(tweet['author']['id']), tweet['author']['public_metrics']['followers_count']))
                #infl_attr[tweet['author']['id']] = tweet['author']['public_metrics']['followers_count']
    
    if len(infl_attr) == 0:
        return
    
    append_objs_to_file(infl_id_path, [attr[0] for attr in infl_attr])
    append_objs_to_file(infl_nflw_path, [attr[1] for attr in infl_attr])
    return