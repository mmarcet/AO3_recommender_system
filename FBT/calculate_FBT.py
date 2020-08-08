#!/usr/bin/env python 

##Calculates FBT based on kudos

import argparse
import numpy as np

def load_kudos(fileName):
    """ Loads kudos into memory """
    kudos = {}
    with open(fileName,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split()
            kudos[dades[0]] = dades[1:]
    return kudos

def calculate_item_frequency(kudos,min_support,itemInterest):
    """ Counts the support for each item in the database and gets rid 
    for all of those that have a support below a minimum. The only 
    exception is the item of interest which will always be included """
    presence = {}
    for reader,group in kudos.items():
        for a in group:
            if a not in presence:
                presence[a] = 0
            presence[a] += 1
    valid_items = {}
    for a in presence:
        support = presence[a] / len(kudos)
        if support >= min_support:
            valid_items[a] = presence[a]
        elif a == itemInterest:
            valid_items[a] = presence[a]
    return valid_items

def calculate_Nitem_frequency(kudos,valid_items,itemInterest,num_items,min_support,min_confidence):
    presence = {}
    for reader,items in kudos.items():
        items = set(items)
        items = list(items.intersection(set(valid_items.keys())))
        if len(items) >= 2 and itemInterest in items:
            for item in items:
                if item != itemInterest:
                    group = (item,itemInterest)
                    group = frozenset(group)
                    if group not in presence:
                        presence[group] = 0
                    presence[group] += 1
    valid_pairs = {}
    for group in presence:
        support = presence[group] / len(kudos)
        confidence = presence[group] / valid_items[itemInterest]
        if support >= min_support or confidence >= min_confidence:
            valid_pairs[group] = [support,confidence]
    return valid_pairs

def load_info(infileName):
    info = {}
    with open(infileName,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            info[dades[0]] = (dades[1],dades[2])
    return info

def print_top10(items,valid_items):
    print("##### Top 10 most read")
    n = 1
    for code in items[:10]:
        print(str(n)+".- "+info[code][1]+" by "+info[code][0]+" ("+str(valid_items[code])+")")
        n+=1

parser = argparse.ArgumentParser(description="Basic recommenders")
parser.add_argument("-k",dest="kudosFile",action="store",required=True,help="File that associates users to kudos")
parser.add_argument("-u",dest="userCode",action="store",default=None,help="User for which we want to make a recommendation")
parser.add_argument("-b",dest="basicFile",action="store",required=True,help="File containing basic information")
parser.add_argument("-f",dest="ficCode",action="store",default=None,help="Fic of interest - idCode")
parser.add_argument("--min_support",dest="minSupport",action="store",type=float,default=0.01,help="Minimum ammount of support")
parser.add_argument("--min_confidence",dest="minConfidence",action="store",type=float,default=0.01,help="Minimum ammount of confidence")
args = parser.parse_args()


kudos = load_kudos(args.kudosFile)
info = load_info(args.basicFile)
itemInterest = args.ficCode
userInterest = args.userCode
valid_items = calculate_item_frequency(kudos,args.minSupport,itemInterest)
items = valid_items.keys()
items = sorted(items,key=lambda x:valid_items[x],reverse=True)

if itemInterest:
    if itemInterest in valid_items:
        valid_pairs = calculate_Nitem_frequency(kudos,valid_items,itemInterest,2,args.minSupport,args.minConfidence)
        pairs = list(valid_pairs.keys())
        pairs = sorted(pairs,key=lambda x:valid_pairs[x][1],reverse = True)
        if len(pairs) == 0:
            print("### No fanfic was associated to your chosen fic")
        else:
            print("#### Most often found together with your item")
            n = 1
            for pair in pairs[:10]:
                p1,p2 = pair
                if p1 != itemInterest:
                    code = p1
                else:
                    code = p2
                print(str(n)+".- "+info[code][1]+" by "+info[code][0]+" (support:"+str(valid_pairs[pair][0])+" confidence:"+str(valid_pairs[pair][1])+")")
                n += 1
    else:
        print("### The selected fic was not among the list")
elif userInterest:
    if userInterest not in kudos:
        print("New user detected")
        print_top10(items,valid_items)
    else:
        userItems = [x for x in kudos[userInterest] if x in valid_items]
        print(userItems)
        if len(userItems) == 0:
            print("No valid associations found for any item read by the user")
            print_top10(items,valid_items)
        else:
            confidences = {}
            for code in userItems[:5]:
                valid_pairs = calculate_Nitem_frequency(kudos,valid_items,code,2,args.minSupport,args.minConfidence)
                for pair in valid_pairs:
                    p1,p2 = pair
                    if p1 != code:
                        fic = p1
                    else:
                        fic = p2
                    if fic not in confidences:
                        confidences[fic] = []
                    confidences[fic].append(valid_pairs[pair][1])
            final_confidence = {}
            for fic in confidences:
                conf = np.average(confidences[fic])
                final_confidence[fic] = conf
            fics = list(final_confidence.keys())
            #Delete fics written by the author
            for code in fics:
                print(code,info[code][0],userInterest)
            if info[code][0] == userInterest:
                print("DLS")
            fics = [x for x in fics if info[x][0] != userInterest]
            #Delete fics that were read by the authors
            fics = [x for x in fics if x not in kudos[userInterest]]
            fics = sorted(fics, key=lambda x: final_confidence[x], reverse = True)
            n = 1
            for fic in fics[:10]:
                print(str(n)+".- "+info[fic][1]+" by "+info[fic][0]+" (average confidence:"+str(final_confidence[fic])+"; Number of times associated:"+str(len(confidences[fic]))+")")
                n += 1
