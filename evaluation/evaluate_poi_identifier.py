#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.cross_validation import train_test_split # todo: figure out the the model_selection version of this is for sklearn 0.18+

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.3)


clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
preds = clf.predict(features_test)
print "number of pois in test set", sum(labels_test)
print "total people in test set", len(labels_test)
print "number of true positives in POI predictions", sum([1 if pred == 1 and labels_test[i] == 1 else 0 for i, pred in enumerate(preds)])
print "precision score", precision_score(labels_test, preds)
print "recall score", recall_score(labels_test, preds)