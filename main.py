#!usr/bin/env python

import numpy as np
import pandas as pd
import os.path
import tweepy

# Keys may otherwise not be properly displayed
pd.set_option("display.max_colwidth", 999)


class TweetCollector(object):
    """
    TweetCollector class collects tweets from a set of pre-defined feeds based
    on a defined file (default 'feeds.csv'). A twitter app account is
    necessary for the class to work correctly. Parameters are to be defined in
    a csv file (path default 'keys.csv').

    :'keys.csv' structure:
    key,value
    ckey,WWWWWWWWWWWWWWWWWWWWWWWWW
    csecret,XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    atoken,YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    asecret,ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

    :'feeds.csv' structure:
    feed,since_id
    feed#1,-1
    feed#2,933644665072603137
    ...

    :param feedfile:        string, default 'feeds.csv' - path to csv file
                            containing information on feeds and since_id
    :param keyfile:         string, default 'keys.csv' - path to csv file
                            containing information on twitter app
    :param count:           integer, default 200
    :param exclude_replies: Boolean, default True - flag exclude tweet
                            replies or not
    :param include_rts:     Boolean, default True - flag include retweets

    Methods defined here

    get_tweets(self)
        Download and accumulate tweets from multiple feeds.

    _download(self, feed, since_id, count=200, exclude_replies=True,
                  include_rts=True)
        Download up to count tweets or tweets from since_id onwards from feed
        using the tweepy api.

    check_ending(self, fname, ending)
        Compares end of filename, fname, with ending.

    new_fname_if_need(self, fname)
        Checks if file exists and proposes new fname if needed.
        IMPORTANT: Relies on the fname having a .xxx ending

    to_CSV(self, csvname='', extend_existing=False, overwrite=False)
        Writes self.tweets to CSV format.

    save_xls(self, xlsname, tweets)
        Saves self.tweets dataframe to xls format.

    to_XLS(self, xlsname='', extend_existing=False, overwrite=False)
        Writes self.tweets to XLS or XLSX format.

    get_tweet_full_text(self, tweet)
        Finds full text of the tweet.
        Irrespective of normal length, extended or retweeted status.

    update_feeds_csv(self, fname='feeds.csv')
        Saves or updates a list of feeds in csv format to the newest status id.

    """

    def __init__(self, feedfile='feeds.csv', keyfile='keys.csv',
                 count=200, include_rts=True, exclude_replies=True):

        kd = pd.read_csv(keyfile)
        self.ckey = kd[kd['key'] == 'ckey']['value'].to_string(index=False)

        self.csecret = kd[kd['key'] == 'csecret']['value'].to_string(
            index=False)

        self.atoken = kd[kd['key'] == 'atoken']['value'].to_string(
            index=False)

        self.asecret = kd[kd['key'] == 'asecret']['value'].to_string(
            index=False)

        self.feeds = pd.read_csv(feedfile)
        self.columns = ['since_id', 'created_at', 'tweet', 'feed']

        # Used in get_tweets and _download
        self.count = count
        self.ex_repl = exclude_replies
        self.rts = include_rts

        self.tweets = self.get_tweets()

    def get_tweets(self):
        """
        Download and accumulate tweets from multiple feeds.

        :returns:   Pandas DataFrame with columns:
                    since_id    -- string of numbers corresponding to tweet
                                   status id
                    created_at  -- string in date format of posting time,
                                   e.g. 2017-12-06 16:37:49
                    tweet       -- string of tweet
                    feed        -- string of name of twitter feed
        """

        alltweets = []
        for feed, since_id in self.feeds.values:
            tweets = self._download(feed, since_id, self.count, self.ex_repl,
                                    self.rts)
            alltweets.append(tweets)

        return pd.concat(alltweets)

    def _download(self, feed, since_id, count=200, exclude_replies=True,
                  include_rts=True):
        """
        Download up to count tweets or tweets from since_id onwards from feed
        using the tweepy api.

        see also https://developer.twitter.com/en/docs/tweets/timelines/guides/
                 working-with-timelines

        Keyword arguments:
        :param feed:            string - username on twitter; here comes
                                from self.feeds dataframe
        :param since_id:        None or string of integers - latest twitter
                                status id
        :param count:           integer, default 200
        :param exclude_replies: Boolean, default True - flag exclude tweet
                                replies or not
        :param include_rts:     Boolean, default True - flag include retweets

        :returns:               Pandas DataFrame with columns:
                                since_id    -- string of numbers corresponding
                                               to tweet status id
                                created_at  -- string in date format of posting
                                               time, e.g. 2017-12-06 16:37:49
                                tweet       -- string of tweet
                                feed        -- string of name of twitter feed

        """

        auth = tweepy.OAuthHandler(self.ckey, self.csecret)
        auth.set_access_token(self.atoken, self.asecret)
        api = tweepy.API(auth)

        # from
        # https://gist.github.com/yanofsky/5436496

        alltweets = []

        if since_id == -1:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                                           exclude_replies=exclude_replies,
                                           include_rts=include_rts,
                                           tweet_mode='extended')
        else:
            new_tweets = api.user_timeline(screen_name=feed, count=count,
                                           since_id=since_id,
                                           exclude_replies=exclude_replies,
                                           include_rts=include_rts,
                                           tweet_mode='extended')

        # save most recent tweets
        alltweets.extend(new_tweets)

        # transform tweepy tweets into a 2D array that will populate the csv
        outtweets = [[tweet.id_str, tweet.created_at,
                      self.get_tweet_full_text(tweet), feed]
                     for tweet in alltweets]

        df_tweets = pd.DataFrame(data=outtweets, columns=self.columns)

        return df_tweets

    def check_ending(self, fname, ending):
        """
        Compares end of filename, fname, with ending.

        :param fname:  string of filename, e.g. example.csv
        :param ending: string of file ending, eg. csv

        :returns: Boolean

        """

        fend = fname.split('.')[-1]
        if fend.lower() != ending.lower():
            return False
        return True

    def new_fname_if_need(self, fname):
        """
        Checks if file exists and proposes new fname if needed.

        IMPORTANT: Relies on the fname having a .xxx ending

        Keyword arguments:
        :param fname: string of filename, e.g. example.csv

        :returns: string

        """

        checking = True
        i = 1
        while checking:
            if os.path.exists(fname):
                new = fname.split('.')
                new[-2] = new[-2] + str(i)
                if os.path.exists('.'.join(new)):
                    i += 1
                else:
                    return '.'.join(new)
            else:
                return fname

    def to_CSV(self, csvname='', extend_existing=False, overwrite=False):
        """
        Writes self.tweets to CSV format.

        :param csvname:         string, default '' - path or filename of the
                                CSV file.
                                If it already exists, then append number until
                                new filename created.
        :param extend_existing: Boolean, default False - If True, extends
                                existing csv file
        :param overwrite:       Boolean, default False - If True, overwrites
                                existing csv file
                                extend_existing supersedes overwrite

        :returns: No return statement

        """

        if csvname == '':
            csvname = 'Tweets.csv'

        # Add file ending if left out
        if not self.check_ending(csvname, 'csv'):
            csvname = csvname + '.csv'

        # Writes to a new file
        if not overwrite and not extend_existing:
            csvname = self.new_fname_if_need(csvname)
            print("Saving to {}".format(csvname))
            self.tweets.to_csv(csvname, index=False)

        # Extends an existing file if valid path given
        # Extend supersedes overwrite
        elif extend_existing:
            try:
                old_tweets = pd.read_csv(csvname)
                if old_tweets.columns.equals(self.tweets.columns):
                    new_tweets = pd.concat([self.tweets, old_tweets])
                    print('Extending tweets in {}...'.format(csvname))
                    new_tweets.to_csv(csvname, index=False)
                else:
                    print('Files do not have the same DataFrame columns. '
                          'Saving to different filename instead.')
                    self.to_CSV(csvname)

            except FileNotFoundError:
                print('File could not be found. Creating new file with '
                      'filename {}...'.format(csvname))
                self.tweets.to_csv(csvname, index=False)

        # Overwrites an existing file if needed
        else:
            print("Saving to {}".format(csvname))
            self.tweets.to_csv(csvname, index=False)

    def save_xls(self, xlsname, tweets):
        """
        Saves self.tweets dataframe to xls format.

        Auxiliary function for to_XLS.

        :param xlsname: string - path or filename of the XLS(X) file.
        :param tweets:  Pandas Dataframe

        :returns: no return statement

        """

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(xlsname, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        tweets.to_excel(writer, sheet_name='Sheet1', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

    def to_XLS(self, xlsname='', extend_existing=False, overwrite=False):
        """
        Writes self.tweets to XLS or XLSX format.

        :param xlsname:         string - path or filename of the XLS file.
                                If it already exists, then append number until
                                new filename created.
        :param extend_existing: boolean, default False - If True, extends
                                existing csv file
        :param overwrite:       boolean, default False - If True, overwrites
                                existing csv file
                                extend_existing supersedes overwrite

        :returns:               no return statement

        """

        if xlsname == '':
            xlsname = 'Tweets.xlsx'

        # Add file ending if left out
        if not self.check_ending(xlsname, 'xlsx') and \
                not self.check_ending(xlsname, 'xls'):
            xlsname = xlsname + '.xlsx'

        # Writes to a new file
        if not overwrite and not extend_existing:
            xlsname = self.new_fname_if_need(xlsname)
            print("Saving to {}".format(xlsname))
            self.save_xls(xlsname, self.tweets)

        # Extends an existing file if valid path given
        # Extend supersedes overwrite
        elif extend_existing:
            try:
                old_tweets = pd.read_excel(xlsname)
                if old_tweets.columns.equals(self.tweets.columns):
                    new_tweets = pd.concat([self.tweets, old_tweets])
                    print('Extending tweets in {}...'.format(xlsname))
                    self.save_xls(xlsname, new_tweets)

                else:
                    print('Files do not have the same DataFrame columns. '
                          'Saving to different filename instead.')
                    self.to_XLS(xlsname)

            except FileNotFoundError:
                print('File could not be found. Creating new file with '
                      'filename {}...'.format(xlsname))
                self.save_xls(xlsname, self.tweets)

        # Overwrites an existing file if needed
        else:
            print("Saving to {}".format(xlsname))
            self.save_xls(xlsname, self.tweets)

    def get_tweet_full_text(self, tweet):
        """
        Finds full text of the tweet.

        Irrespective of normal length, extended or retweeted status.

        :param tweet: Tweepy Status Object
                      see also http://docs.tweepy.org/en/v3.5.0/api.html

        :returns:     string

        """

        full_text = ""

        if 'retweeted_status' in tweet._json:
            full_text = tweet._json['retweeted_status']['full_text']

        elif "extended_tweet" in tweet._json and \
                "full_text" in tweet._json["extended_tweet"]:
            full_text = tweet._json["extended_tweet"]["full_text"]

            if "display_text_range" in tweet._json["extended_tweet"]:
                # Brackets to allow line split
                beginIndex = (
                    tweet._json["extended_tweet"]["display_text_range"][0])
                endIndex = (
                    tweet._json["extended_tweet"]["display_text_range"][1])
                full_text = full_text[beginIndex:endIndex]

        elif "full_text" in tweet._json:
            full_text = tweet._json["full_text"]

            if "display_text_range" in tweet._json:
                beginIndex = tweet._json["display_text_range"][0]
                endIndex = tweet._json["display_text_range"][1]
                full_text = full_text[beginIndex:endIndex]

        elif "text" in tweet._json:
            text = tweet._json["text"]
            full_text = text
        else:
            text = tweet.text
            full_text = text

        return full_text

    def update_feeds_csv(self, fname='feeds.csv'):
        """
        Saves or updates a list of feeds in csv format to the newest status id.

        :param fname: string - path or filename of the csv file.

        :returns: no return statement

        """

        feeds = self.tweets['feed'].unique()
        for f in feeds:
            since_id = self.tweets[self.tweets['feed'] == f]['since_id'].max()
            self.feeds.loc[self.feeds['feed'] == f, 'since_id'] = since_id
        self.feeds.to_csv(fname, index=False)


if __name__ == '__main__':
    tw = TweetCollector(feedfile='example_feeds.csv')
    tw.to_CSV(csvname='example_tweets.csv', overwrite=True,
              extend_existing=False)
    tw.to_XLS(xlsname='example_tweets.xlsx', overwrite=True,
              extend_existing=True)
    tw.save_feeds_csv(fname='example_feeds.csv')
