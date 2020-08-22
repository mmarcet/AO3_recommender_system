#!/usr/bin/env python

import argparse
import numpy as np

#Calculates ranking metrics for a set of predictions

def calculate_metrics(predicted,read, K):
    """Calculates metrics for a given recommendation """
    TP = len(predicted.intersection(read))
    recall = TP / min(len(read),K)
    precision = TP / K
    if precision != 0.0 and recall != 0.0:
        F1 = 2 / ((1/precision) + (1/recall))
    else:
        F1 = 0.0
    return recall,precision,F1,TP

def mapk(predicted, read, k):
    """
    Computes the average precision at k.
    
    :param read : A list of elements that are to be predicted (order doesn't matter)
    :param predicted : A list of predicted elements (order does matter)
    :param k: The maximum number of predicted elements
    
    :return The average precision at k over the input lists
    """
    
    score = 0.0    # This will store the numerator
    num_hits = 0.0 # This will store the sum of rel(i)

    for i, p in enumerate(predicted):
        if p in read and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(read), k)

def evaluate_predictions(seenAll,complete,recommendations):
    mainRecall, mainPrec, mainF1, mainMap = [],[],[],[]
    for user, seen in seenAll.items():
        not_seen_yet = complete[user].difference(seen)
        recommend = recommendations[user]
        common = set(recommend).intersection(seenAll)
        if len(not_seen_yet) != 0:
            recall,precision,F1,TP = calculate_metrics(set(recommend), not_seen_yet, len(recommend))
            score = mapk(recommend, not_seen_yet, len(recommend))
            mainRecall.append(recall)
            mainPrec.append(precision)
            mainF1.append(F1)
            mainMap.append(score)
            # ~ print(user, recall, precision, F1, TP, set(recommend).intersection(not_seen_yet))
    return np.average(mainRecall),np.average(mainPrec),np.average(mainF1),np.average(mainMap)

def load_dataset(infile):
    data = {}
    with open(infile,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            if dades[0] not in data:
                data[dades[0]] = set([])
            data[dades[0]].add(dades[1])
    return data

def get_complete_list(train, test):
    complete = {}
    for user in train:
        fics = train[user].copy()
        fics.update(test[user])
        complete[user] = fics
    return complete

def load_recommendations(recommFile):
    recomm = {}
    with open(recommFile,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            recomm[dades[1]] = dades[2].split(";")
    return recomm
    
    
parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("--trainFile",dest="trainFile",action="store",required=True,help="Provides the training file")
parser.add_argument("--testFile",dest="testFile",action="store",required=True,help="Provides the test file")
parser.add_argument("--recommendationsFile",dest="recomFile",action="store",required=True,help="File containing the recommendations")
args = parser.parse_args()

train = load_dataset(args.trainFile)
test = load_dataset(args.testFile)
complete = get_complete_list(train, test)
recommendations = load_recommendations(args.recomFile)

recallT, precisionT, F1T, map_at_kT = evaluate_predictions(train,complete,recommendations)

print("Recall@K:",recallT)
print("Precision@K:",precisionT)
print("F1@K:",F1T)
print("Map@K:",map_at_kT)
