# Tweetery

TweetCollector class collects tweets from a set of pre-defined feeds using the tweepy api.

## Dependencies
* [Python3](https://www.python.org/downloads/)
* [Tweepy Library](http://www.tweepy.org/)
* [Numpy Library](http://www.numpy.org/)
* [Pandas Library](https://pandas.pydata.org/)

## Feeds
The feeds are based on a defined file. The default being 'feeds.csv' in the same folder, but can be defined according to your needs.

### Structure of feed file (csv format)

feed,since_id 
feed_1,-1 
feed_2,933644665072603137 
...

## Twitter App
A twitter app account is necessary for the class to work correctly. Parameters are to be defined in a csv file (path default 'keys.csv').

### Structure of keys file for twitter app (csv file)
As these parameters are to be kept secret, there is no key.csv file given by default.

key,value
ckey,WWWWWWWWWWWWWWWWWWWWWWWWW
csecret,XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
atoken,YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
asecret,ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

## Example code
```
# Instantiates TweetCollector Object and downloads tweets automatically
tw = TweetCollector(feedfile='example_feeds.csv')

# Saves to 'example_tweets.csv' and overwrites an existing file in the process
tw.to_CSV(csvname='example_tweets.csv', overwrite=True,
          extend_existing=False)

# Extends the existing 'example_tweets.xlsx' file or
# creates new if it does not exist
tw.to_XLS(xlsname='example_tweets.xlsx', overwrite=True,
          extend_existing=True)

# Updates the existing 'example_feeds.csv' file with the latest tweet status id
# or saves as new file.
# Next time it is called, it will only look for tweets in the feed, "after" the
# status id
tw.save_feeds_csv(fname='example_feeds.csv')
```
