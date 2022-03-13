"""
sample.py

Sample from the tweet stream for a specified duration
(max 900 seconds). Call 'python -m sample time', where
'time' is the sampling duration in seconds.
"""

import sys
import time
import datetime
import threading
from twarc import Twarc2
from tweet_mining import append_objs_to_file

def main():
    sampling_seconds = int(sys.argv[1])
    academic_access = False

    if academic_access:
        with open('tokens/academic_bearer_token.txt', 'r') as file:
            BEARER_TOKEN = file.read()
    else:
        with open('tokens/bearer_token.txt', 'r') as file:
            BEARER_TOKEN = file.read()

    client = Twarc2(bearer_token=BEARER_TOKEN)
    print('Client initialized.')

    # Event controlled sampling
    thread_stop = threading.Event()
    thread_stop.clear()

    t_query = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'sampled_tweets_{t_query}.jsonl'

    def sample():
        # Samples from the Twitter API stream. Used in a Thread.
        # No arguments or return values.
        
        append_objs_to_file(file_name, client.sample(event=thread_stop))

    # Begin streaming tweets
    sample_thread = threading.Thread(target=sample)
    # sample_thread = threading.Thread(target=sample, args=[file_name]) # or args=(file_name,)
    sample_thread.daemon = True
    sample_thread.start()

     
    if sampling_seconds <= 0:
        # default to 20 sec sampling if time is not positive
        sampling_seconds = 20
    else:
        # max 15 minutes (900 sec) sampling
        sampling_seconds = min(sampling_seconds, 900)

    print(f'Sampling for {sampling_seconds} seconds.')

    sec = sampling_seconds//10
    rem_sec = sampling_seconds%10
    print(f'0/{sampling_seconds}')
    for i in range(sec):
        time.sleep(10)
        print(f'{(i+1)*10}/{sampling_seconds}')

    if rem_sec != 0:
        time.sleep(rem_sec)
        print(f'{sampling_seconds}/{sampling_seconds}')

    # Stop sampling
    thread_stop.set()

    print(f'Sampling done. File: {file_name}')


if __name__=='__main__':
    main()