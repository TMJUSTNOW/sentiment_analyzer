import json
import time

import tweepy



hashtags_ema_dict = {}
hashtags_counter_dict = {}
disp_time = 100         # in seconds
start = int(time.time())

class TweepyListener(tweepy.StreamListener):
    def on_data(self, encoded_data):

        decoded_data = json.loads(encoded_data)
        global start

        # update counter
        update_counter(data=decoded_data)

        # if tweet is Quoted add Quoted hash tags
        if 'quoted_status' in decoded_data:
            update_counter(data=decoded_data['quoted_status'])

        if (int(time.time()) - start) > disp_time:   # show Result if elapsed time is greater than display time
            # Find ema(exponential moving average) and display top trends
            find_ewma()
            start = int(time.time())

        return True

def update_counter(data):
    '''
    Updates a dictionary containing hash tag as key and number of occurrence as value
    :param data: Decoded Twitter API data
    '''
    try:
        hash_tag_list = []
        if 'extended_tweet' in data:
            if data['extended_tweet']['entities']['hashtags']:
                hash_tag_list = data['extended_tweet']['entities']['hashtags']
        else:
            if data['entities']['hashtags']:
                hash_tag_list = data['entities']['hashtags']

        if hash_tag_list:
            for each_dict in hash_tag_list:
                if each_dict['text'] in hashtags_counter_dict:  # if hash key present in counter update it
                    hashtags_counter_dict[each_dict['text']] += 1
                else:
                    hashtags_counter_dict[each_dict['text']] = 1
    except Exception as exc:
        print('Exception while updating counter: {0}'.format(exc))

def find_ewma():
    '''
    Keep track of Exponential Weighted Moving Average of each tag and display top few
    '''

    # Find exponential moving average of counter dict
    weight_factor = 0.6
    global hashtags_counter_dict

    try:
        key_not_in_counter = list(hashtags_ema_dict.keys())
        for each_key in hashtags_counter_dict:
            if each_key in hashtags_ema_dict:
                key_not_in_counter.remove(each_key)
                # calculate exponential moving average
                # giving more weight to present value
                hashtags_ema_dict[each_key] = (1 - weight_factor) * hashtags_ema_dict[each_key] + weight_factor * \
                                                                                                  hashtags_counter_dict[
                                                                                                      each_key]
            else:
                # Assume ema[0] = y[0]
                hashtags_ema_dict[each_key] = hashtags_counter_dict[each_key]

        # calculating ema for key which not present in counter. Assume their values as zero
        for each_key in key_not_in_counter:
            hashtags_ema_dict[each_key] *= (1 - weight_factor)

        # Reset Counter
        hashtags_counter_dict = {}

        # Sorting and displaying top 5 item by ema value
        top_hashtags = sorted(hashtags_ema_dict, key=lambda x: hashtags_ema_dict[x], reverse=True)[:5]
        disp_string = ''
        for val in top_hashtags:
            disp_string += '\n' + '#' + val
        print("Top trending Hash Tags now are: {0}".format(disp_string))
    except Exception as exc:
        print('Exception occurred while calculating EWMA: {0}'.format(exc))


if __name__ == '__main__':
    # api_key = input('Provide Consumer Key: ')
    # api_secret_key = input('Provide Consumer Secret key: ')
    # api_access_key = input('Provide Access Key: ')
    # api_access_secret_key = input('Provide Access Secret Key: ')

    api_key = 'BMaRbtElbiTtZiV8B21yD5nAa'
    api_secret_key = 'nYoxVIHJsjhHDpNCESXkKiTOBgrGs4O34QkBtDDAjlshKFaSNs'
    api_access_key = '884591331779031040-qGeQFpCrHGaJFnhCKk4BUIBLn0cWhr1'
    api_access_secret_key = 'QeV2ppw3R71hNvA1zwS5Tmz0t6OXedOF6ma0VAlVLNMUW'
    title = input('Provide a input to Search: ')
    print('Trending Tags are shown in every {0}sec\n'.format(disp_time))
    try:
        auth = tweepy.OAuthHandler(api_key, api_secret_key)
        auth.set_access_token(api_access_key, api_access_secret_key)
        overwrite_listener_obj = TweepyListener()
        stream_api = tweepy.Stream(auth=auth, listener=overwrite_listener_obj)
        stream_api.filter(track=[title])
    except Exception as exc:
        print('Exception occurred during Authenticating or searching: {0}'.format(exc))
