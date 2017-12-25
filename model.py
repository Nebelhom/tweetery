#!usr/bin/python

import pandas as pd
import numpy as np

import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split

# https://pypi.python.org/pypi/emoji/
import emoji

# Still needs
# 1. both path to classifiable tweets & to training tweets
# 2. allow text and classifier columns for both as above

class ML_Model():
	def __init__(self, fname, text_col='text', clf_col='clf'):
		# Adjust to a property function based on ending either csv or xlsx
		self.df = pd.read_excel(fname)
		self.df1 = df[['tweet', 'Interesting']]

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

if __name__ == '__main__':
	ml = ML_Model('Competitor_Tweets_checked.xlsx', text_col='tweet', clf_col='Interesting')
