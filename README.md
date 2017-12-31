# Tweetery

The Tweetery project is tweet collection and classification module.

Based on a pre-defined list of twitter feeds and timestamp of your choosing, tweets will be collected and can be classified in interesting or not based on supplied training data.

TweetCollector class collects tweets from a set of pre-defined feeds using the tweepy api.

TweetClassifier class classifies tweets based on pre-defined training data using a Logistic Regression algorithm.

Please see below for example code.

## Dependencies
* [Python3](https://www.python.org/downloads/)
* [Tweepy Library](http://www.tweepy.org/)
* [Scikit Learn Library](http://scikit-learn.org/stable/)
* [Numpy Library](http://www.numpy.org/)
* [Pandas Library](https://pandas.pydata.org/)
* [NLTK Library](http://www.nltk.org/)
* [python-docx Library](https://python-docx.readthedocs.io/en/latest/)
* [emoji Library](https://pypi.python.org/pypi/emoji/)

## Feeds
The feeds are based on a defined file. The default being 'feeds.csv' in the user_data folder, but can be defined according to your needs. It is, however, important that you place it in the '''user_data''' directory.

### Structure of feed file (csv format)
```
feed,since_id 
feed_1,-1 
feed_2,933644665072603137 
...
```

## Twitter App
A twitter app account is necessary for the class to work correctly. Parameters are to be defined in a csv file (path default 'user_data/keys.csv').

### Structure of keys file for twitter app (csv file)
As these parameters are to be kept secret, there is no key.csv file given by default. Again it is, however, important that you place it in the '''user_data''' directory.

```
key,value
ckey,WWWWWWWWWWWWWWWWWWWWWWWWW
csecret,XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## TweetCollector DataFrame Structure
TweetCollector returns a DataFrame with the following columns:

since_id    -- string of numbers corresponding
               to tweet status id
created_at  -- string in date format of posting
               time, e.g. 2017-12-06 16:37:49
tweet       -- string of tweet
feed        -- string of name of twitter feed

## TweetClassifier DataFrame Structures
### Classifiable Tweets
The TweetClassifier takes a DataFrame with the following columns:
since_id    -- string of numbers corresponding
               to tweet status id
created_at  -- string in date format of posting
               time, e.g. 2017-12-06 16:37:49
tweet       -- string of tweet
feed        -- string of name of twitter feed

In the typical use case, the structure from TweetCollector is passed in.

### Training Data
Training Data have the following minimal structure
since_id    -- string of numbers corresponding
               to tweet status id
created_at  -- string in date format of posting
               time, e.g. 2017-12-06 16:37:49
tweet       -- string of tweet
feed        -- string of name of twitter feed
interesting -- string of 0 or 1 (interesting or not)

### Return Data
TweetClassifier will return a DataFrame (or a report of it)
since_id    -- string of numbers corresponding
               to tweet status id
created_at  -- string in date format of posting
               time, e.g. 2017-12-06 16:37:49
tweet       -- string of tweet
feed        -- string of name of twitter feed
interesting -- string of 0 or 1 (interesting or not)
probability -- string of float of probability of the tweet being interesting (i.e. 1)

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

# Get tweets to be classified
clf = tw.get_tweets()

# Load training data. If Classifier is already saved, there is no need
# Pandas Library also allows for other data formats to be loaded
# Pre-requisite: Must have a pre-defined DataFrame structure as described above
df = pd.read_excel('Training Data.xlsx')

# Instantiate Tweet_Classifier
tc = Tweet_Classifier(clf, train=df)

# Save the trained classifier
tc.save_classifier()

# Predict the downloaded tweets
tc.predict()

# Save them as report
tc.save_as_doc()
tc.save_as_txt()

# Extend the existing training Data by the newly classified ones
# It may be necessary to change some things manually if classification
# was wrong
tweets = tc.get_prediction().drop('probability', axis=1)
tw.to_XLS(xlsname='Training Data.xlsx', tweets=tweets, overwrite=True,
          extend_existing=True)
```
