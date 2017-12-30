#!usr/bin/python

import datetime
import os
import os.path as osp
from pathlib import Path
import string

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import docx
from docx.shared import Cm, Inches
from hyperlink import add_hyperlink

# https://pypi.python.org/pypi/emoji/
import emoji


TODAY = datetime.date.today()


class Tweet_Classifier(object):
    """
    add_clf_info - additional information on the tweet such as feed, data, since_id, etc
    """
    def __init__(self, data, train=None, ignore_saved=False):
        """
        self._dest = path to data directory
        self._clf_path = path to pickled classifier
        self._hyperparams = path to pre-saved hyperparams

        self.data - pd.DataFrame with columns since_id  created_at  tweet   feed
        """
        # Paths
        self._dest = osp.join(os.getcwd(), 'data')
        self._clf_path = osp.join(self._dest, 'classifier.pkl')
        self._hyperparams = osp.join(self._dest, 'hyperparams.pkl')

        # information fed into class
        self.data = data
        self.train_set = train

        # Adjust to a property function based on ending either csv or xlsx
        self.tfidf = TfidfVectorizer(analyzer=self.text_process,
                                     ngram_range=(1, 1))

        self.clf = self.create_classifier(train['tweet'],
                                          train['interesting'],
                                          ignore_saved_model=ignore_saved)

        # Pandas Series or None
        self.clf_text = self.data['tweet']
        # Pandas series or None
        self.prediction = None
        # Pandas series or None
        self.proba = None
        # Pandas Dataframe of all info or None
        self.assembled = None

    def create_classifier(self, X=None, y=None, ignore_saved_model=False):
        """
        Returns a fully trained model ready to be used.
        """

        if Path(self._clf_path).is_file() and not ignore_saved_model:
            model = joblib.load(open(self._clf_path, 'rb'))
            print('Classifier successfully loaded')
            return model

        else:
            print('No classifier pre-saved. Please wait while a new ',
                  'classifier is being calibrated...')

            if X is not None and y is not None:
                clf = self.calibrate(X, y)
                print('Classifier has been calibrated.')
                return clf

            else:
                print('You have not defined any training data.',
                      ' Please define:\n')
                if X is None:
                    print('X - pandas Dataframe defining data features.\n')
                if y is None:
                    print('y - pandas Data Series defining the expected',
                          ' classification.\n')
                return None

    def calibrate(self, X, y, full_calibration=False):
        """
        Take calibration data and calibrate
        Give choice, with pre-existing values set or completely from scratch
         (later).
        Use Gridsearch and several classifiers (if possible) for it.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=42)
        tfidf = TfidfVectorizer(analyzer=self.text_process)
        pipe = Pipeline([
            ('vect', tfidf),
            ('clf', LogisticRegression())
        ])

        clf = None

        if Path(self._hyperparams).is_file() and not full_calibration:
            param_grid = joblib.load(open(self._hyperparams, 'rb'))
            pipe.set_params(**param_grid)
            pipe.fit(X_train, y_train)
            clf = pipe

        else:
            param_grid = [
                {'vect__ngram_range': [(1, 1), (2, 2), (1, 2)],
                 'vect__use_idf': [False, True],
                 'clf__penalty': ['l1', 'l2'],
                 'clf__C': [0.1, 1.0, 10.0, 100.0]
                 }
            ]
            grid = GridSearchCV(pipe, param_grid, scoring='accuracy',
                                cv=10, verbose=10, n_jobs=-1)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_

        return clf

    def predict(self, pref_prob=1):
        """
        pref_prob - index of preferred probability of classification
        """

        self.prediction = self.clf.predict(self.clf_text)
        # print(self.prediction)
        self.proba = self.clf.predict_proba(self.clf_text)
        # print(self.proba)
        a = pd.DataFrame({'tweet': self.clf_text,
                          'interesting': self.prediction,
                          'probability': self.proba[:, pref_prob]})
        self.assembled = pd.merge(self.data, a, how='outer', left_on='tweet',
                                  right_on='tweet')

    def save_classifier(self):
        """
        """

        if self.clf is None:
            print('Cannot save NoneType. Are you sure a classifier is loaded?')
        else:
            if not osp.exists(self._dest):
                os.makedirs(self._dest)
            # Classifier saved
            joblib.dump(self.clf, open(self._clf_path, 'wb'),
                        protocol=4)
            print('Classifier saved.')

            # Hyperparameters saved
            joblib.dump(self.clf.get_params(), open(self._hyperparams, 'wb'),
                        protocol=4)
            print('Hyperparameters of Classifier saved.')

    def save_as_txt(self, fname='report{}.txt'.format(TODAY), cut_off=0.25):
        """
        Saves the classification outcome as a text of tweet | Relevance.
        """
        if self.prediction is None or self.proba is None or \
                self.assembled is None:
            print('No classification has taken place. Please',
                  ' re-use this method after classification has taken place')
            return

        else:
            with open(fname, 'w') as f:
                f.write('{:12s}\t{}\t{}\t{}\n'.format('Relevance', 'Text',
                                                      'URL', 'Date'))
                f.write('{:12s}\t{}\t{}\t{}\n'.format('=========', '====',
                                                      '===', '===='))
                rel = self.assembled.sort_values(by='probability',
                                                 ascending=False)

                # columns are 'since_id', 'created_at', 'tweet', 'feed',
                # 'interesting', 'probability',
                for since_id, date, tweet, feed, clf, prob in rel.values:
                    if prob >= cut_off:
                        url = 'https://twitter.com/{feed}/status/{since_id}'\
                            .format(feed=feed, since_id=since_id)
                        f.write('{:7.2f}%\t{tweet}\t{url}\t{date}\n'
                                .format(prob * 100, tweet=tweet, url=url,
                                        date=date))
            print('Report saved in {}'.format(osp.abspath(fname)))
            return

    def save_as_doc(self, fname='report{}.docx'.format(TODAY),
                    doc_title='Tweet Report{}'.format(TODAY), cut_off=0.25):
        """
        Saves the classification outcome as a text of tweet | Relevance.

        Add cols: URL and Date
        Include Hyperlinks inside tweets
        Show Feed as feed with hyperlink to tweet

        """

        def make_bold(cell):
            run = cell.paragraphs[0].runs[0]
            font = run.font
            font.bold = True
            return

        def make_hyperlink(cell, url, text, color=None, underline=True):
            p = cell.paragraphs[0]
            hyperlink = add_hyperlink(p, url, text, None, True)
            return

        def set_col_widths(table, col_widths):
            """
            Sets the widths of columns of table

            table = docx.Table() instance
            col_widths = iterable of ints (width in EMUs)
                         Helper functions like Cm() or Inches() help
            """
            for col, width in zip(table.columns, col_widths):
                col.width = width
            return

        if self.prediction is None or self.proba is None or \
                self.assembled is None:
            print('No classification has taken place. Please',
                  ' re-use this method after classification has taken place')
            return

        else:
            doc = docx.Document()

            # Create the template
            doc.add_heading(doc_title, 0)

            # Create table
            table = doc.add_table(rows=1, cols=4)
            table.autofit = False

            # Header Cells
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Relevance'
            make_bold(hdr_cells[0])

            hdr_cells[1].text = 'Tweet'
            make_bold(hdr_cells[1])

            hdr_cells[2].text = 'Feed / Link'
            make_bold(hdr_cells[2])

            hdr_cells[3].text = 'Date'
            make_bold(hdr_cells[3])

            # Body of Table
            rel = self.assembled.sort_values(by='probability',
                                             ascending=False)

            # columns are 'since_id', 'created_at', 'tweet', 'feed',
            # 'interesting', 'probability',
            for since_id, date, tweet, feed, clf, prob in rel.values:
                if prob >= cut_off:
                    url = 'https://twitter.com/{feed}/status/{since_id}'\
                        .format(feed=feed, since_id=since_id)
                    row_cells = table.add_row().cells
                    row_cells[0].text = '{:7.2f}%'.format(prob * 100)
                    row_cells[1].text = tweet
                    row_cells[2].text = ''
                    make_hyperlink(row_cells[2], url, feed)
                    row_cells[3].text = str(date)[:11]

            set_col_widths(table, (Inches(1.06), Inches(3.0), Inches(0.88),
                                   Inches(1.06)))

            doc.save(fname)
            print('Report saved in {}'.format(osp.abspath(fname)))
            return

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
        nopunc = [char for char in mess
                  if char not in string.punctuation + "–"]

        # Check if char is an emoji
        noemoji = [char for char in nopunc
                   if char not in emoji.UNICODE_EMOJI.keys()]

        # Join the characters again to form the string.
        nopunc = ''.join(noemoji)

        # Now just remove any stopwords
        no_stop = [word for word in nopunc.split() if word.lower() not in
                   stopwords.words('english') if 'http' not in word.lower()]

        # Stemming
        snowball = SnowballStemmer('english')

        return [snowball.stem(word) for word in no_stop]


if __name__ == '__main__':
    clf = pd.read_excel('example_tweets.xlsx')
    df = pd.read_excel('training_data.xlsx')
    ml = Tweet_Classifier(clf, train=df)
    # Commented out so that hyperparam.pkl does not always change on commit
    # ml.save_classifier()
    ml.predict()
    #ml.save_as_doc()
    # ml.save_as_txt()
