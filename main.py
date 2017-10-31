#!usr/bin/env python

import csv
from tweepy import OAuthHandler, API

"""
Purpose: Download tweets from interesting feeds. classify them into interesting and not interesting

Use this to train a natural language model to classify future tweets.

Later try to classify automatically into separate categories

At the very end search large twitter space to look for fitting tweets across whole space with trained model


TODO:
Access the twitter feeds

Write the tweets to csv... or sql

read them out from sql again

"""


class TweetDownloader(object):
    """
    Class written to download tweets from all feeds specified and then save them in a csv file... or sql (make it sql for fun)

    currently planned format:
    twitter_feed | Tweet | Date | Interesting? (Used to train classifier) | (Optional) Keywords automatically added
    """


    def __init__(self, _feedfile='feeds.csv', _keyfile='keys.csv'):
        key_dict = self.read_keys(_keyfile)

        self.ckey = key_dict['ckey']
        self.csecret = key_dict['csecret']
        self.atoken = key_dict['atoken']
        self.asecret = key_dict['asecret']

        self.feeds = self.read_feeds(_feedfile)


    def read_feeds(self, fname):
        """
        reads the user_names for the feeds from CSV file and returns list

        Argument:
        fname : String : Path to file

        Returns:
        results : list
        """
        results = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                results.append(row[0])

        return results

    def read_keys(self, fname):
        """
        reads the keys from CSV file and returns dictionary

        Argument:
        fname : String : Path to file

        Returns:
        results : dict : key -> value
        """
        keys = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                keys.append(row)

        results = {}
        for key, value in keys:
            results[key] = value

        return results

    def download_all_tweets(self, feed):
        auth = OAuthHandler(self.ckey, self.csecret)
        auth.set_access_token(self.atoken, self.asecret)
        api = API(auth)

        # from
        # https://stackoverflow.com/questions/45867934/is-it-possible-to-download-tweets-and-retweets-of-10000-users-with-tweepy-throug
        alltweets = []
        new_tweets = api.user_timeline(screen_name = feed, count=200) 

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        #keep grabbing tweets until there are no tweets left to grab
        #while new_tweets:       
        #    try:
        #        new_tweets = api.user_timeline(screen_name = feed,count=200,max_id=oldest)
        #    except tweepy.TweepError:
        #        time.sleep(60 * 15)
        #        continue

        return alltweets

if __name__ == '__main__':
    tw = TweetDownloader()
    tweets = tw.download_all_tweets(tw.feeds[0])
    print(tweets[0])
