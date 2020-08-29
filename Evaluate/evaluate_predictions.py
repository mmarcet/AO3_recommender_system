#!/usr/bin/env python

import argparse
import numpy as np

#Calculates ranking metrics for a set of predictions

def calculate_metrics(predicted,read, K):
    """Calculates metrics for a given recommendation 
    Input: predicted -> Set of recomendations
           read -> Set of fics read
           K -> Number of items to consider
    Output: recall, precision, F1
    """
    TP = len(predicted.intersection(read))
    recall = TP / min(len(read),K)
    precision = TP / K
    if precision != 0.0 and recall != 0.0:
        F1 = 2 / ((1/precision) + (1/recall))
    else:
        F1 = 0.0
    return recall,precision,F1

def mapk(predicted, read, k):
    """
    Computes the average precision at k.
    
    Input: read -> predicted -> Set of recomendations
           read -> Set of fics read
           K -> Number of items to consider
    
    Output: The average precision at k over the input lists
    """
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in read and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score / min(len(read), k)

def evaluate_predictions(train,test,recommendations,K):
    """ Calculates the average recall@K, precision@K, F1@K and map@k for 
    each user provided.
    Input: train: The set of users and the fics each user has read in the 
                  training set
           test: The list of fics each user has read in the test set
           recommendations: List of recommendations for each user
           K: Number of recommendations to evaluate
    Output: Averages for all users of the four metrics
    """
    mainRecall, mainPrec, mainF1, mainMap = [],[],[],[]
    for user in recommendations:
        not_seen = test[user]
        recommend = recommendations[user][:K]
        if len(not_seen) != 0:
            recall,precision,F1 = calculate_metrics(set(recommend), not_seen, K)
            score = mapk(recommend, not_seen, K)
            mainRecall.append(recall)
            mainPrec.append(precision)
            mainF1.append(F1)
            mainMap.append(score)
    return np.average(mainRecall),np.average(mainPrec),np.average(mainF1),np.average(mainMap)

def load_dataset(infile):
    """ Imports list of fics that a user has read. When testing this
    should be the training file.
    Input: Name of the file that contains three columns: user\titem\trating
    Output: A dictionary where the user is the key and the value is a set
    of items the user has read.
    """
    data = {}
    with open(infile,"r") as infile:
        for line in infile:
            line = line.strip()
            if "user" in line and "item" in line:
                pass
            else:
                dades = line.split("\t")
                if dades[0] not in data:
                    data[dades[0]] = set([])
                data[dades[0]].add(dades[1])
    return data

def load_recommendations(recommFile):
    """ Loads a list of recommendations into memory.
    Input: File name of the recommendations to upload
    Output: Dictionary where the keys are the users and the values
            are a list of recommendations for said users
    """
    recomm = {}
    with open(recommFile,"r") as infile:
        for line in infile:
            line = line.strip()
            if ";" in line:
                dades = line.split("\t")
                recomm[dades[0]] = dades[1].split(";")
    return recomm
    
    
parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("-train",dest="trainFile",action="store",\
    required=True,help="Provides the training file")
parser.add_argument("-test",dest="testFile",action="store",\
    required=True,help="Provides the test file")
parser.add_argument("-recom",dest="recomFile",\
    action="store",required=True,help="File containing the recommendations")
parser.add_argument("-k",dest="numR",type=int, \
    action="store",default=20,help="Number of recommendations to evaluate")
args = parser.parse_args()

train = load_dataset(args.trainFile)
test = load_dataset(args.testFile)
recommendations = load_recommendations(args.recomFile)
K = args.numR
recall, precision, F1, map_at_k = evaluate_predictions(train,test,recommendations,K)

print("Reacall@K\tPrecision@K\tF1@K\tMap@K")
print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(recall,precision, F1, map_at_k))


