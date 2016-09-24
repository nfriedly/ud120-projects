#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    for i, pred in enumerate(predictions):
        nw = net_worths[i]
        err = pred-nw
        tuple = (ages[i],nw, err)
        cleaned_data.append(tuple)

    cleaned_data = sorted(cleaned_data, key=lambda tuple: abs(tuple[2])) # sort list by absolute error
    cleaned_data = cleaned_data[0:-len(cleaned_data)/10] # chop the last 10 %

    return cleaned_data

