#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print(len(enron_data))

i=0
for k, person in enron_data.iteritems():
    if person["poi"]:
        i=i+1
print(i)

print(enron_data["PRENTICE JAMES"]["total_stock_value"])

print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])


print(enron_data["LAY KENNETH L"]["total_payments"])      # 103559793
print(enron_data["SKILLING JEFFREY K"]["total_payments"]) #   8682716
print(enron_data["FASTOW ANDREW S"]["total_payments"])    #   2424083

len({k: v for k, v in enron_data.iteritems() if v["salary"] != "NaN"}) # 95
len({k: v for k, v in enron_data.iteritems() if v["email_address"] != "NaN"}) # 111

len({k: v for k, v in enron_data.iteritems() if v["total_payments"] == "NaN"}) / float(len(enron_data)) # 0.14383561643835616

len({k: v for k, v in enron_data.iteritems() if v["total_payments"] == "NaN" and v["poi"]}) / float(len(enron_data)) # 0.0

len(enron_data) + 10
len({k: v for k, v in enron_data.iteritems() if v["total_payments"] == "NaN"}) + 10