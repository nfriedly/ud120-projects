#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

t0 = time()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# # all in USD
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
           'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
           'long_term_incentive', 'restricted_stock', 'director_fees']

# integer number of emails (except , 'email_address' - excluded for now)
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
           'shared_receipt_with_poi']

label_features = ['poi']
features_list = label_features
features_list.extend(financial_features)
features_list.extend(email_features)

# POI flag + top 10 features from SelectKBest (feature scores dropped fairly quickly after the 10th best)
# features_list = ['poi', 'bonus', 'salary', 'total_stock_value', 'exercised_stock_options', 'shared_receipt_with_poi', 'total_payments', 'deferred_income', 'restricted_stock', 'long_term_incentive', 'loan_advances']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
#
from sklearn.feature_selection import SelectKBest
selection = SelectKBest(k=5)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


# probably needs some scaling to be useful
from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3)
# , ("scaler", MinMaxScaler())

# creates multiple instances of another classifier (Decision Tree by default) and tweaks the values of each one to better handle hard-to-classify data
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()

# creates multiple decision trees that randomly ignore certain features or training data (?)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()

# always hits divide-by-zero errors
# from sklearn.svm import SVC

# clf = SVC()

from sklearn.pipeline import Pipeline, FeatureUnion
#combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
clf = Pipeline([("features", selection), ("clf", clf)])




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV
# param_grid = dict(features__pca__n_components=range(1,10),
#                   features__univ_select__k=range(1,10))
#
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
# clf.fit(features_train, labels_train)
# print
# print "grid search results" , clf.best_estimator_
#


# from sklearn.feature_selection import SelectKBest
# selection = SelectKBest(k=5)
# features_train = selection.fit_transform(features_train, labels_train)
# features_test = selection.transform(features_test)
#
# print
# print [[features_list[i], score] for i, score in enumerate(selection.scores_)]
# print
# print "num features: ", features_train.shape

# clf.fit(features_train, labels_train)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# print "done in %0.3fs" % (time() - t0)

dump_classifier_and_data(clf, my_dataset, features_list)

test_classifier(clf, my_dataset, features_list)