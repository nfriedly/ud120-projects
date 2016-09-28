#!/usr/bin/python

import os
import pickle
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


def import_emails(quick=False, force=False):

    # first see if we have a cached pickel file
    if not force and not quick and os.path.isfile("your_word_data.pkl") and os.path.isfile("your_email_authors.pkl"):
        word_data = pickle.load(open("your_word_data.pkl", "r")) # todo: do I need to close these files?
        from_data = pickle.load(open("your_email_authors.pkl", "r"))
        return word_data, from_data

    from_data = []
    word_data = []

    from_sara  = open("from_sara.txt", "r")
    from_chris = open("from_chris.txt", "r")


    ### temp_counter is a way to speed up the development--there are
    ### thousands of emails from Sara and Chris, so running over all of them
    ### can take a long time
    ### temp_counter helps you only look at the first 200 emails in the list so you
    ### can iterate your modifications quicker
    temp_counter = 0


    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            ### only look at first 200 emails when developing
            ### once everything is working, remove this line to run over full dataset
            temp_counter += 1
            if (temp_counter < 100 and quick) or not quick:
                path = os.path.join('..', path[:-1])
                print path
                email = open(path, "r")

                ### use parseOutText to extract the text from the opened email
                text = parseOutText(email)

                ### use str.replace() to remove any instances of the words
                for word in ["sara", "shackleton", "chris", "germani"]:
                    text = text.replace(word, '')

                # not sure if this part is expected by the quiz, but it seems to get me closer to their target number
                text = ' '.join(text.split()) # remove double whitespaces and any beginning/end whitespace

                if text != "":
                    ### append the text to word_data
                    word_data.append(text)

                    ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
                    from_data.append( 0 if name is "sara" else 1 )

                email.close()

    print "emails processed:", temp_counter

    from_sara.close()
    from_chris.close()

    if not quick:
        pickle.dump( word_data, open("your_word_data.pkl", "w") )
        pickle.dump( from_data, open("your_email_authors.pkl", "w") )

    return word_data, from_data

word_data, from_data = import_emails(force=False,quick=False)
assert len(word_data) == len(from_data), 'lengths not equal'

print "word_data[152] = ", word_data[152] if len(word_data) >=152 else "not set"

### in Part 4, do TfIdf vectorization here

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(word_data, from_data)
print "number of feature names:", len(tfidf.get_feature_names()) # expecting 38757
print "word 34597 = ", tfidf.get_feature_names()[34597] # quiz wants "stephaniethank"