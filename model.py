#!usr/bin/python

import os
import os.path as osp
from pathlib import Path
import pickle
import string

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# https://pypi.python.org/pypi/emoji/
import emoji

"""
TODO:
- Save and load model in pickle or SQL
- calibrate method to make it check all models and parameters
- In create_model: check for file or calibrate model (be verbose if poss)
- Output as report

"""


class Text_Classifier():
    """
    """
    def __init__(self, clf_text, train_X, train_y):
        """
        """
        self._dest = osp.join(os.getcwd(), 'pickles')
        self._clf_path = osp.join(os.getcwd(), 'pickles', 'classifier.pkl')

        # Adjust to a property function based on ending either csv or xlsx
        self.tfidf = TfidfVectorizer(analyzer=self.text_process,
                                     ngram_range=(1, 1))
        self.clf = self.create_classifier(train_X, train_y)

        # Pandas Series or None
        self.clf_text = clf_text
        # Pandas series or None
        self.prediction = None
        # Pandas series or None
        self.proba = None
        # Pandas Dataframe of clf_text & prediction or None
        self.paired = None

    def text_process(self, mess):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        if type(mess) is not str:
            mess = str(mess)
        
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation + "–"]

        # Check if char is an emoji
        noemoji = [char for char in nopunc if char not in emoji.UNICODE_EMOJI.keys()]

        # Join the characters again to form the string.
        nopunc = ''.join(noemoji)

        # Now just remove any stopwords
        no_stop = [word for word in nopunc.split() if word.lower() not in
                   stopwords.words('english') if 'http' not in word.lower()]
        
        # Stemming
        snowball = SnowballStemmer('english')
        
        return [snowball.stem(word) for word in no_stop]

    def create_classifier(self, X, y, ignore_saved_model=False):
        """
        Returns a fully trained model ready to be used.
        TODO:
        - Integrate calibrate
        - Check if there is a file that can be loaded
        """
        if Path(self._clf_path).is_file() and not ignore_saved_model:
            model = pickle.load(open(self._clf_path, 'rb'))
            print('Classifier successfully loaded')
            return model

        else:
            print('No classifier pre-saved. Please wait while a new ',
                  'classifier is calibrated...')
            pipe = Pipeline([
                ('vect', self.tfidf),
                ('clf', LogisticRegression(C=10.0, penalty='l2'))
            ])
            pipe.fit(X, y)
            print('Classifier has been calibrated.')
            return pipe

    def calibrate(self, fname, X, y):
        """
        Take calibration data and calibrate
        Give choice, with pre-existing values set or completely from scratch 
        (later).
        Use Gridsearch for it and several classifiers for it.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=42)
        pass

    def predict(self, pref_prob=1):
        """
        pref_prob - index of preferred probability of classification
        """
        self.prediction = self.clf.predict(self.clf_text)
        self.proba = self.clf.predict_proba(self.clf_text)
        self.paired = pd.DataFrame({'text': self.clf_text,
                                    'interesting': self.prediction,
                                    'probability': self.proba[:,pref_prob]})

    def save_classifier(self):
        if self.clf is None:
            print('Cannot save NoneType. Are you sure a classifier is loaded?')
        else:
            if not osp.exists(self._dest):
                os.makedirs(self._dest)
            pickle.dump(self.clf, open(self._clf_path, 'wb'),
                              protocol=4)
            print('Classifier saved.')


if __name__ == '__main__':
    clf = pd.read_excel('example_tweets.xlsx')
    df = pd.read_excel('training_data.xlsx')
    ml = Text_Classifier(clf['tweet'], df['tweet'], df['interesting'])
    ml.save_classifier()
    ml.predict()
    print(ml.paired)
