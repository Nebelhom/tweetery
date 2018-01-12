#!usr/bin/python

class ZeroTweetsException(Exception):

    def __init__(self):
        """Create an instance of an exception."""
        message = ('There were no new tweets to be downloaded. ',
                   'The procedure will now be terminated...')
        Exception.__init__(self, message)
