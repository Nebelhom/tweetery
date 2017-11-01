#!usr/bin/env python

import csv
import time
import tweepy
#from tweepy import OAuthHandler, API

"""
Purpose: Download tweets from interesting feeds. classify them into interesting and not interesting

Use this to train a natural language model to classify future tweets.

Later try to classify automatically into separate categories

At the very end search large twitter space to look for fitting tweets across whole space with trained model


TODO:
Write the tweets to csv (DONE nearly)... or sql

read them out from sql again

"""


class TweetCollector(object):
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
            for feed, since_id in reader:
                results.append((feed, since_id))

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

    def download(self, feed, count=200, since_id=None, exclude_replies=True, include_rtf=True):
        auth = tweepy.OAuthHandler(self.ckey, self.csecret)
        auth.set_access_token(self.atoken, self.asecret)
        api = tweepy.API(auth)

        # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
        # Use as reference

        # from
        # https://gist.github.com/yanofsky/5436496
        # and
        # https://stackoverflow.com/questions/45867934/is-it-possible-to-download-tweets-and-retweets-of-10000-users-with-tweepy-throug
        alltweets = []
        if since_id:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                since_id=since_id, exclude_replies=exclude_replies,
                include_rtf=include_rtf)
        else:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                exclude_replies=exclude_replies, include_rtf=include_rtf)

        # save most recent tweets
        alltweets.extend(new_tweets)

        return alltweets

    def download_all(self, feed, exclude_replies=True, include_rtf=True):
        auth = tweepy.OAuthHandler(self.ckey, self.csecret)
        auth.set_access_token(self.atoken, self.asecret)
        api = tweepy.API(auth)

        # from
        # https://gist.github.com/yanofsky/5436496
        # and
        # https://stackoverflow.com/questions/45867934/is-it-possible-to-download-tweets-and-retweets-of-10000-users-with-tweepy-throug
        alltweets = []
        new_tweets = api.user_timeline(screen_name=feed, count=200,
            exclude_replies=exclude_replies, include_rtf=include_rtf)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            print("getting tweets before {}".format(oldest))
            
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(screen_name=feed, count=200,
            exclude_replies=exclude_replies, include_rtf=include_rtf)
            
            #save most recent tweets
            alltweets.extend(new_tweets)
            
            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1
            
            print ("...{} tweets downloaded so far".format(len(alltweets)))

        return alltweets

    def toCSV(self, tweets, feed):
        #transform the tweepy tweets into a 2D array that will populate the csv 
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in tweets]
        
        #write the csv  
        with open('{}_tweets.csv'.format(feed), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["id","created_at","text"])
            writer.writerows(outtweets)


if __name__ == '__main__':
    tw = TweetCollector()
    for feed, since_id in tw.feeds:
        if since_id == '-1':
            tweets = tw.download(feed)
        else:
            tweets = tw.download(feed, since_id=since_id)
        tw.toCSV(tweets, feed)
